"""One bf16 base for the math RL engine and its trainer: the Unsloth-style baseline of RL-SHARED-BF16 (docs/REGISTRY.md;
mock review v18 R2, missing experiment 3: what FP8 adds beyond sharing one copy).

lowbit_lora/shared_base.py maps an online-FP8 engine's codes into the weight-matched FP8 trainer. Here the engine is vLLM's
bf16 engine and the trainer the plain bf16 trainer: the worker extension SharedBaseWorkerBF16 exports CUDA IPC handles of the
engine's bf16 linear weights ([N, K] as F.linear reads them; fused q/k/v, gate/up and GDN projections as vLLM stacks them),
and map_info_bf16 points the trainer's nn.Linear weights, embedding and LM head at those tensors (row views; nothing is
copied or written: the base is frozen). The trainer computes exactly as the bf16 trainer does, on the engine's tensors, and
holds no copy of its own; the engine must not sleep (vLLM's sleep mode would offload or discard the shared weights).
    llm = LLM(..., worker_extension_cls="lowbit_lora.shared_base_bf16.SharedBaseWorkerBF16")
    stats = map_info_bf16(model, llm.collective_rpc("export_base_ipc_bf16")[0], check=True)   # model on the CPU, before PEFT
"""
from __future__ import annotations

import torch
from mlsys.ipc_share import _LAYER_MOD, SharedBaseWorker, pack_cuda, unpack_cuda
from mlsys.matched_fp8 import SKIP
from mlsys.vllm_ext import fused_layout


class SharedBaseWorkerBF16(SharedBaseWorker):
    """SharedBaseWorker (mlsys/ipc_share.py, unchanged) plus the bf16 export."""

    @torch.no_grad()
    def export_base_ipc_bf16(self) -> dict:
        """{'tensors': {id: packed bf16 [N, K]}, 'modules': {'<layer>|<hf path>': {'tensor', 'row0', 'rows'}},
        'embed'/'lm_head': packed bf16 [V_pad, H] or None, 'n_bf16': count}."""
        model = self._model()
        packed = getattr(model, "packed_modules_mapping", {}) or {}
        keep = getattr(self, "_mlsys_ipc_keep", None)
        if keep is None:
            keep = self._mlsys_ipc_keep = []                    # the exported tensors stay alive with the engine
        out = {"tensors": {}, "modules": {}, "embed": None, "lm_head": None, "n_bf16": 0}
        for n, p in model.named_parameters():
            if not n.endswith(".weight") or p.dim() != 2 or p.dtype != torch.bfloat16:
                continue
            if n.endswith("embed_tokens.weight") or n.startswith("lm_head."):
                key = "embed" if "embed_tokens" in n else "lm_head"
                t = p.data.contiguous()
                keep.append(t)
                out[key] = pack_cuda(t)
                continue
            if not _LAYER_MOD.search(n):
                continue
            mod = model.get_submodule(n.rsplit(".", 1)[0])
            sizes = list(getattr(mod, "output_sizes", []) or []) or None
            layer, parts = fused_layout(n, packed, sizes)
            w = p.data                                          # [N, K], as the unquantized linear method reads it
            if not w.is_contiguous():
                raise ValueError(f"{n}: bf16 weight is not contiguous (stride {p.stride()})")
            keep.append(w)
            out["tensors"][n] = pack_cuda(w)
            out["n_bf16"] += 1
            for path, off, rows in parts:
                out["modules"][f"{layer}|{path}"] = {"tensor": n, "row0": off, "rows": w.shape[0] if rows < 0 else rows}
        return out


@torch.no_grad()
def map_info_bf16(model: torch.nn.Module, info: dict, check: bool = False) -> dict:
    """Point model's linear layers (matched_fp8's SKIP aside), embedding and LM head at the engine's bf16 tensors, then move
    the rest (norms, GDN conv and gates; a module the engine did not export keeps its own copy) to the engine's GPU."""
    tensors = {k: unpack_cuda(v) for k, v in info["tensors"].items()}
    dev = next(iter(tensors.values())).device
    lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not SKIP.search(n)}
    mapped, own, differ = 0, [], []
    for n, m in lins.items():
        mm = _LAYER_MOD.search(n)
        ent = info["modules"].get(f"{int(mm.group(1))}|{mm.group(2)}") if mm else None
        if ent is None:
            own.append(n)
            continue
        w = tensors[ent["tensor"]][ent["row0"]: ent["row0"] + ent["rows"]]
        if tuple(w.shape) != tuple(m.weight.shape):
            raise ValueError(f"{n}: engine rows {tuple(w.shape)} vs HF weight {tuple(m.weight.shape)}")
        if check and not torch.equal(w.cpu(), m.weight.detach().to(torch.bfloat16)):
            differ.append(n)
        bias = m.bias
        m.weight = torch.nn.Parameter(w, requires_grad=False)
        if bias is not None:
            m.bias = torch.nn.Parameter(bias.detach().to(dev), requires_grad=False)
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
        if check and not torch.equal(t[:v].cpu(), mod.weight.detach().to(torch.bfloat16)):
            differ.append(key)
        mod.weight = torch.nn.Parameter(t[:v], requires_grad=False)
        shared_bytes += t[:v].numel() * t.element_size()
    model.to(dev)
    return {"linear_layers": len(lins), "mapped": mapped, "own": len(own), "own_examples": own[:3],
            "engine_bf16_tensors": info["n_bf16"], "shared_gb": round(shared_bytes / 2**30, 2), "checked": check,
            "differ": len(differ), "differ_examples": differ[:3]}
