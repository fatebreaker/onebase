"""vLLM worker extension: serve any rounding variant of a model without writing a checkpoint.

    VLLM_SERVER_DEV_MODE=1 PYTHONPATH=<mlsys repo> vllm serve <base dir> \
        --worker-extension-cls mlsys.vllm_ext.VariantWorker [--enable-lora ...]
    POST /collective_rpc {"method": "apply_variant", "args": ["<json spec>"]}       then POST /reset_prefix_cache

Spec (JSON string; collective_rpc only forwards strings):
    {"scheme": "<merge.apply_scheme name>" | "none",   # "none" restores the pristine base
     "role": "base" | "rl",                            # load base_q (no adapter needed) or rl_q (needs "adapter")
     "adapter": "<PEFT adapter dir>",                  # role rl: merged into the base before rounding
     "seed": 0,                                       # stochastic rounding only; each parameter gets its own stream
     "base_other": "<dir>", "rl_from": "<dir>"}       # full-parameter pairs: pre-RL base (and RL weights) read from disk

Weights are changed IN PLACE (param.data.copy_), so captured CUDA graphs stay valid. The pristine base is kept in
pinned CPU memory and every variant starts from it. Row-wise and row-group-wise grids are invariant to vLLM's row
stacking of q/k/v and gate/up, so fused parameters are quantized as stored; the LoRA update of each original module
is placed at its rows of the fused parameter. (nvfp4's per-tensor scale is taken over the fused tensor.)
"""
from __future__ import annotations

import json
import math
import re
import time
import zlib
from pathlib import Path

import torch
from safetensors import safe_open

from . import merge

_EXCLUDE = re.compile(r"norm|conv|embed|lm_head|A_log|dt_bias|\.bias$")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)\.weight$")


# ----------------------------------------------------------------------------------------------- pure helpers

def target_params(names_shapes: list[tuple[str, tuple[int, ...]]]) -> list[str]:
    """Decoder-layer 2-D linear weights (attention, MLP, GDN projections); not norms, convs, embeddings, lm_head."""
    return [n for n, shp in names_shapes if len(shp) == 2 and _LAYER.search(n) and not _EXCLUDE.search(n)]


def fused_layout(name: str, packed: dict[str, list[str]], sizes: list[int] | None
                 ) -> tuple[int, list[tuple[str, int, int]]]:
    """For a vLLM param name -> (layer, [(hf module path, row offset, rows)]). An unfused param maps to itself
    with rows=-1 (all rows)."""
    m = _LAYER.search(name)
    layer, path = int(m.group(1)), m.group(2)                 # e.g. 3, "self_attn.qkv_proj"
    path = re.sub(r"\.base_layer$", "", path)                 # vLLM's LoRA wrappers: <module>.base_layer.weight
    head, _, leaf = path.rpartition(".")
    subs = packed.get(leaf)
    if not subs:
        return layer, [(path, 0, -1)]
    if sizes is not None and len(sizes) > len(subs):
        # e.g. GDN in_proj_qkvz: vLLM shards q, k, v, z (4 sizes) but the checkpoint packs in_proj_qkv + in_proj_z;
        # the leading shards belong to the first packed module
        k = len(sizes) - len(subs) + 1
        sizes = [sum(sizes[:k])] + list(sizes[k:])
    if sizes is None or len(sizes) != len(subs):
        raise ValueError(f"{name}: packed {subs} but output sizes {sizes}")
    out, off = [], 0
    for sub, n in zip(subs, sizes):
        out.append((f"{head}.{sub}" if head else sub, off, n))
        off += n
    return layer, out


def read_adapter(adapter: str | Path) -> tuple[float, dict[tuple[int, str], tuple[str, str]], Path]:
    ad = Path(adapter)
    cfg = json.loads((ad / "adapter_config.json").read_text())
    r = cfg["r"]
    scale = cfg["lora_alpha"] / (math.sqrt(r) if cfg.get("use_rslora") else r)
    keys = {}
    with safe_open(str(ad / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            m = re.search(r"\.layers\.(\d+)\.(.+)\.lora_A\.weight$", k)
            if m:
                keys[(int(m.group(1)), m.group(2))] = (k, k.replace(".lora_A.", ".lora_B."))
    return scale, keys, ad / "adapter_model.safetensors"


def update_for(name: str, shape: tuple[int, int], packed: dict[str, list[str]], sizes: list[int] | None,
               scale: float, keys: dict, fa, device) -> tuple[torch.Tensor | None, int]:
    """The LoRA update of a (possibly fused) vLLM weight, or None if the adapter does not touch it."""
    layer, parts = fused_layout(name, packed, sizes)
    dW, hit = None, 0
    for path, off, rows in parts:
        kk = keys.get((layer, path))
        if kk is None:
            continue
        piece = merge.lora_delta(fa.get_tensor(kk[0]).to(device), fa.get_tensor(kk[1]).to(device), scale)
        n = shape[0] if rows < 0 else rows
        if piece.shape != (n, shape[1]):
            raise ValueError(f"{name}/{path}: LoRA update {tuple(piece.shape)} != rows {n} x {shape[1]}")
        if dW is None:
            dW = torch.zeros(shape, dtype=torch.float32, device=device)
        dW[off:off + n] = piece
        hit += 1
    return dW, hit


def base_rows_for(name: str, shape: tuple[int, int], packed: dict[str, list[str]], sizes: list[int] | None,
                  base, device) -> torch.Tensor:
    """The pre-RL base weight of a (possibly fused) vLLM parameter, read from an HF checkpoint (census.Base)."""
    layer, parts = fused_layout(name, packed, sizes)
    out = torch.empty(shape, dtype=torch.float32, device=device)
    for path, off, rows in parts:
        w = base.get(f"model.layers.{layer}.{path}.weight").to(device).float()
        n = shape[0] if rows < 0 else rows
        if w.shape != (n, shape[1]):
            raise ValueError(f"{name}/{path}: base weight {tuple(w.shape)} != rows {n} x {shape[1]}")
        out[off:off + n] = w
    return out


def module_seed(seed: int, name: str) -> int:
    """An independent random stream per parameter (stochastic rounding): the spec's seed mixed with the name."""
    return (seed * 1_000_003 + zlib.crc32(name.encode())) % 2**31


def chunked_scheme(W: torch.Tensor, dW: torch.Tensor | None, scheme: str, role: str, seed: int,
                   rows_per_chunk: int = 4096, stats: dict | None = None) -> torch.Tensor:
    """merge.apply_scheme over row chunks (all grids are row-local); returns the float32 values to load.
    With `stats` and an update dW, accumulates the delivered update D = rl_q - base_q against dW: num = <D, dW>,
    den = <dW, dW> (gain = num / den, as census.measure), nnz = entries with D != 0 (a sparse sync's payload), n."""
    out = torch.empty(W.shape, dtype=torch.float32, device=W.device)
    for i in range(0, W.shape[0], rows_per_chunk):
        w = W[i:i + rows_per_chunk].float()
        d = dW[i:i + rows_per_chunk] if dW is not None else torch.zeros_like(w)
        base_q, rl_q = merge.apply_scheme(w, d, scheme, seed=seed + i)
        out[i:i + rows_per_chunk] = base_q if role == "base" else rl_q
        if stats is not None and dW is not None:
            D = rl_q - base_q
            stats["num"] = stats.get("num", 0.0) + float((D.double() * d.double()).sum())
            stats["den"] = stats.get("den", 0.0) + float((d.double() ** 2).sum())
            stats["nnz"] = stats.get("nnz", 0) + int((D != 0).sum())
            stats["n"] = stats.get("n", 0) + D.numel()
    return out


# ----------------------------------------------------------------------------------------------- vLLM mixin

class VariantWorker:
    """Mixed into vLLM's GPU worker by --worker-extension-cls; methods are called through collective_rpc."""

    def _model(self):
        mr = self.model_runner
        return mr.get_model() if hasattr(mr, "get_model") else mr.model

    def _setup(self):
        if getattr(self, "_mlsys_pristine", None) is not None:
            return
        model = self._model()
        params = dict(model.named_parameters())
        self._mlsys_targets = target_params([(n, tuple(p.shape)) for n, p in params.items()])
        self._mlsys_pristine = {n: params[n].detach().to("cpu").pin_memory() for n in self._mlsys_targets}
        self._mlsys_packed = getattr(model, "packed_modules_mapping", {}) or {}
        sizes = {}
        for n in self._mlsys_targets:
            mod = model.get_submodule(n.rsplit(".", 1)[0])
            sizes[n] = list(getattr(mod, "output_sizes", []) or []) or None
        self._mlsys_sizes = sizes
        self._mlsys_current = "none"

    def variant_info(self) -> dict:
        self._setup()
        return {"targets": len(self._mlsys_targets), "packed": self._mlsys_packed, "current": self._mlsys_current,
                "pristine_gb": round(sum(t.numel() * t.element_size() for t in self._mlsys_pristine.values()) / 2**30, 2)}

    @torch.no_grad()
    def apply_variant(self, spec_json: str) -> dict:
        self._setup()
        spec = json.loads(spec_json)
        scheme, role, seed = spec.get("scheme", "none"), spec.get("role", "base"), int(spec.get("seed", 0))
        if role == "rl" and scheme.startswith("unmerged_"):
            raise ValueError("unmerged variants: apply role=base here and hot-load the adapter through vLLM's LoRA path")
        t0 = time.time()
        params = dict(self._model().named_parameters())
        fa = keys = None
        scale = 1.0
        other_base = rl_disk = None
        if spec.get("base_other"):                      # full-parameter pair: served model = RL, base read from disk
            from .census import Base
            other_base = Base(spec["base_other"])
            if spec.get("rl_from"):                     # the RL weights too, in full precision: a served copy in fp16
                rl_disk = Base(spec["rl_from"])         # would round an fp32 master twice before the scheme sees it
        if role == "rl" and scheme != "none" and other_base is None:
            scale, keys, path = read_adapter(spec["adapter"])
            fa = safe_open(str(path), framework="pt")
        changed = merged = 0
        stats: dict = {}
        for n in self._mlsys_targets:
            p = params[n]
            W = self._mlsys_pristine[n].to(p.device, non_blocking=True)
            if scheme == "none":
                p.data.copy_(W)
                continue
            dW = None
            if other_base is not None:
                if role != "rl":
                    raise ValueError("base_other needs role=rl (serve the base checkpoint itself for base_q)")
                Wb = base_rows_for(n, tuple(p.shape), self._mlsys_packed, self._mlsys_sizes[n], other_base, p.device)
                Wr = W.float() if rl_disk is None else \
                    base_rows_for(n, tuple(p.shape), self._mlsys_packed, self._mlsys_sizes[n], rl_disk, p.device)
                dW = Wr - Wb                              # the full-parameter RL update of this weight
                new = chunked_scheme(Wb, dW, scheme, role, module_seed(seed, n), stats=stats).to(p.dtype)
                changed += int((new != W).sum())
                p.data.copy_(new)
                merged += 1
                del new, dW, Wb
                continue
            if fa is not None:
                dW, hit = update_for(n, tuple(p.shape), self._mlsys_packed, self._mlsys_sizes[n], scale, keys, fa,
                                     p.device)
                merged += hit
            if role == "rl" and dW is None and merge.is_on_grid(scheme) and scheme.endswith("merge"):
                p.data.copy_(W)                       # untouched module under a pure merge scheme: base as stored
                continue
            new = chunked_scheme(W, dW, scheme, role, module_seed(seed, n), stats=stats if role == "rl" else None).to(p.dtype)
            changed += int((new != W).sum())
            p.data.copy_(new)
            del new, dW
        if keys is not None and merged == 0:
            raise RuntimeError(f"adapter {spec['adapter']} matched no parameter (packed={self._mlsys_packed})")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._mlsys_current = f"{scheme}:{role}"
        out = {"scheme": scheme, "role": role, "merged_modules": merged, "entries_changed_vs_stored": changed,
               "seconds": round(time.time() - t0, 1)}
        if stats.get("den"):                          # the update as delivered (rl_q - base_q) vs the true update
            out |= {"gain": round(stats["num"] / stats["den"], 5), "delta_entries": stats["nnz"],
                    "delta_share": round(stats["nnz"] / max(stats["n"], 1), 6)}
        return out
