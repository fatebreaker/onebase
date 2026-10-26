"""One FP8 base on the GPU for a LoRA RL rollout engine and all its trainers (lowbit-lora P5, 2026-10-11).

The weight-matched trainer (mlsys/matched_fp8.py, parts="w") computes on exactly the FP8 codes and per-group scales that
vLLM's online FP8 serves, so trainer and engine hold bit-identical copies of the frozen base. Here the engine's worker
exports CUDA IPC handles of those tensors (and of the bf16 embedding and LM head), and each trainer on the same GPU maps
them instead of building its own: an extra RL job then costs its LoRA, optimizer and activations, not another base.

Engine:   VLLM_SERVER_DEV_MODE=1 vllm serve <bf16 dir> --quantization fp8 --enable-lora ...
              --worker-extension-cls mlsys.ipc_share.SharedBaseWorker
Trainer:  from mlsys.ipc_share import map_shared_base
          model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
          stats = map_shared_base(model, "http://127.0.0.1:<port>")       # before PEFT wraps the model

vLLM stores each online-FP8 weight as qweight.t(): a [K, N] view of a contiguous [N, K] tensor of codes, one scale per
(fused) tensor; a fused tensor's rows are the stacked HF modules (q/k/v, gate/up, GDN qkv+z, b+a). The trainer maps
the [N, K] storage and takes each module's rows as a contiguous view. Nothing is copied or written: the base is frozen.
"""
from __future__ import annotations

import base64
import re

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from .vllm_ext import fused_layout

_LAYER_MOD = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


# ------------------------------------------------------------------------------------------- IPC (de)serialization

def pack_cuda(t: torch.Tensor) -> dict:
    """JSON-friendly CUDA IPC description of `t` (the producer must keep `t` alive while consumers use it)."""
    _, a = reduce_tensor(t)
    (_, size, stride, off, _, dtype, dev, handle, nbytes, soff, _, rch, rco, evh, evs) = a
    b64 = lambda x: base64.b64encode(x).decode() if x is not None else None  # noqa: E731
    return {"size": list(size), "stride": list(stride), "offset": int(off), "dtype": str(dtype).replace("torch.", ""),
            "device": int(dev), "handle": b64(handle), "nbytes": int(nbytes), "storage_offset": int(soff),
            "ref_handle": b64(rch), "ref_offset": int(rco), "event": b64(evh), "event_sync": bool(evs)}


def unpack_cuda(d: dict) -> torch.Tensor:
    """Map a tensor packed by pack_cuda in another process on the same GPU (no copy)."""
    ub64 = lambda x: base64.b64decode(x) if x is not None else None  # noqa: E731
    return rebuild_cuda_tensor(torch.Tensor, torch.Size(d["size"]), tuple(d["stride"]), d["offset"],
                               torch.storage.TypedStorage, getattr(torch, d["dtype"]), d["device"], ub64(d["handle"]),
                               d["nbytes"], d["storage_offset"], False, ub64(d["ref_handle"]), d["ref_offset"],
                               ub64(d["event"]), d["event_sync"])


# ----------------------------------------------------------------------------------------------- engine side

class SharedBaseWorker:
    """Mixed into vLLM's GPU worker by --worker-extension-cls; called through POST /collective_rpc."""

    def _model(self):
        mr = self.model_runner
        return mr.get_model() if hasattr(mr, "get_model") else mr.model

    @torch.no_grad()
    def export_base_ipc(self) -> dict:
        """{'tensors': {id: packed [N, K] FP8 codes}, 'scales': {id: float}, 'modules': {'<layer>|<hf path>':
        {'tensor', 'row0', 'rows'}}, 'embed'/'lm_head': packed bf16 [V_pad, H] or None, 'n_fp8': count}."""
        model = self._model()
        packed = getattr(model, "packed_modules_mapping", {}) or {}
        keep = getattr(self, "_mlsys_ipc_keep", None)
        if keep is None:
            keep = self._mlsys_ipc_keep = []                    # the exported views stay alive with the engine
        out = {"tensors": {}, "scales": {}, "modules": {}, "embed": None, "lm_head": None, "n_fp8": 0}
        for n, p in model.named_parameters():
            if not n.endswith(".weight") or p.dim() != 2:
                continue
            if p.dtype == torch.float8_e4m3fn:
                mod = model.get_submodule(n.rsplit(".", 1)[0])
                sizes = list(getattr(mod, "output_sizes", []) or []) or None
                layer, parts = fused_layout(n, packed, sizes)
                codes = p.data.t()                              # [N, K]: the contiguous storage behind qweight.t()
                if not codes.is_contiguous():
                    raise ValueError(f"{n}: FP8 weight is not a transposed contiguous tensor (stride {p.stride()})")
                scale = getattr(mod, "weight_scale")
                if scale.numel() != 1:
                    raise ValueError(f"{n}: expected one per-tensor scale, got {tuple(scale.shape)}")
                keep.append(codes)
                out["tensors"][n] = pack_cuda(codes)
                out["scales"][n] = float(scale.detach().float().reshape(()).item())
                out["n_fp8"] += 1
                for path, off, rows in parts:
                    out["modules"][f"{layer}|{path}"] = {"tensor": n, "row0": off,
                                                         "rows": codes.shape[0] if rows < 0 else rows}
            elif p.dtype == torch.bfloat16 and (n.endswith("embed_tokens.weight") or n.startswith("lm_head.")):
                key = "embed" if "embed_tokens" in n else "lm_head"
                t = p.data.contiguous()
                keep.append(t)
                out[key] = pack_cuda(t)
        return out


# ---------------------------------------------------------------------------------------------- trainer side

def _rpc(url: str, method: str) -> dict:
    from .client import rpc
    res = rpc(url, method)
    if not res:
        raise RuntimeError(f"{url}: /collective_rpc {method} returned nothing (VLLM_SERVER_DEV_MODE=1 and "
                           "--worker-extension-cls mlsys.ipc_share.SharedBaseWorker are required)")
    return res[0]


@torch.no_grad()
def map_shared_base(model: torch.nn.Module, url: str, parts: str = "w", check_codes: bool = False) -> dict:
    """Point every eligible nn.Linear of an HF model (loaded on CPU, before PEFT) at the engine's FP8 codes and
    scales, and the embedding / LM head at the engine's bf16 tensors; move what remains (norms, GDN conv and gates) to
    the GPU. The forward is matched_fp8's (parts="w": FP8 weights, bf16 activations). check_codes=True also
    recomputes this process's own codes from the CPU weights and asserts they equal the engine's, bit for bit."""
    from .matched_fp8 import FP8_MAX, SKIP, _make_forward
    if parts != "w":
        raise ValueError("the shared base serves the weight-matched trainer (parts='w') only")
    info = _rpc(url, "export_base_ipc")
    tensors = {k: unpack_cuda(v) for k, v in info["tensors"].items()}
    dev = next(iter(tensors.values())).device
    fwd = _make_forward(False, parts)
    lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not SKIP.search(n)}
    mapped, mismatched = 0, []
    for n, m in lins.items():
        mm = _LAYER_MOD.search(n)
        key = f"{int(mm.group(1))}|{mm.group(2)}" if mm else None
        ent = info["modules"].get(key) if key else None
        if ent is None:
            raise KeyError(f"{n}: no engine tensor for {key}")
        codes = tensors[ent["tensor"]][ent["row0"]: ent["row0"] + ent["rows"]]
        if tuple(codes.shape) != tuple(m.weight.shape):
            raise ValueError(f"{n}: engine rows {tuple(codes.shape)} vs HF weight {tuple(m.weight.shape)}")
        scale = torch.tensor(info["scales"][ent["tensor"]], dtype=torch.float32, device=dev)
        if check_codes:     # this process's own codes (matched_fp8.patch_model's rule) against the engine's
            own = torch.clamp(m.weight.detach().float() / scale.cpu(), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            ne = int((own.view(torch.uint8) != codes.cpu().view(torch.uint8)).sum())
            if ne:
                mismatched.append((n, ne, own.numel()))
        m.register_buffer("w_codes", codes, persistent=False)
        m.register_buffer("w_scale", scale, persistent=False)
        bias = m.bias
        m.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=dev), requires_grad=False)
        if bias is not None:
            m.bias = torch.nn.Parameter(bias.detach().to(dev), requires_grad=False)
        m.forward = fwd.__get__(m, type(m))
        mapped += 1
    shared_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    for key, getter in (("embed", lambda: model.get_input_embeddings()), ("lm_head", lambda: model.get_output_embeddings())):
        if info.get(key) is None:
            continue
        mod = getter()
        t = unpack_cuda(info[key])
        v = mod.weight.shape[0]
        if t.shape[0] < v or t.shape[1] != mod.weight.shape[1]:
            raise ValueError(f"{key}: engine {tuple(t.shape)} vs HF {tuple(mod.weight.shape)}")
        if check_codes and not torch.equal(t[:v].cpu(), mod.weight.detach().to(torch.bfloat16)):
            mismatched.append((key, -1, t[:v].numel()))
        mod.weight = torch.nn.Parameter(t[:v], requires_grad=False)
        shared_bytes += t[:v].numel() * t.element_size()
    model.to(dev)                                               # the rest: norms, GDN conv/gates, biases (small)
    out = {"linear_layers": len(lins), "mapped": mapped, "engine_fp8_tensors": info["n_fp8"],
           "shared_gb": round(shared_bytes / 2**30, 2), "checked": check_codes, "parts": parts}
    if check_codes:   # the trainer now computes on the engine's codes; this reports how far its own would have been
        n_ent = sum(k for _, k, _ in mismatched if k > 0)
        n_all = sum(t for _, _, t in mismatched) or 1
        out["own_vs_engine"] = {"modules_differing": len(mismatched), "entries_differing": n_ent,
                                "examples": [(n, k) for n, k, _ in mismatched[:5]],
                                "embed_or_head_differ": [n for n, k, _ in mismatched if k < 0]}
    return out
