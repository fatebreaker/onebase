"""One FP8 base for an IN-PROCESS rollout engine and its trainer (lowbit-lora RL-INTEGRATED, 2026-10-22).

mlsys/ipc_share.py exports the engine's FP8 codes over CUDA IPC to a trainer in another process. The matched-precision engine
(lowbit_lora/w16_engine.py) runs in the trainer's own process (VLLM_ENABLE_V1_MULTIPROCESSING=0), where a process cannot open its
own IPC handles; there the worker hands over the tensors themselves (collective_rpc returns Python objects in process). Same
layout and selection as SharedBaseWorker.export_base_ipc, with tensors in place of packed handles.

    LLM(..., worker_extension_cls="mlsys.ipc_share_local.SharedBaseLocalWorker")    # engine in this process
    info = llm.collective_rpc("export_base_local")[0]                                # tensors, not handles
"""
from __future__ import annotations

import torch

from .ipc_share import SharedBaseWorker
from .vllm_ext import fused_layout


class SharedBaseLocalWorker(SharedBaseWorker):

    @torch.no_grad()
    def export_base_local(self) -> dict:
        """export_base_ipc's dict with the [N, K] FP8 code tensors and the bf16 embedding / LM head themselves."""
        import os
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
            raise RuntimeError("export_base_local needs the engine in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)")
        model = self._model()
        packed = getattr(model, "packed_modules_mapping", {}) or {}
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
                out["tensors"][n] = codes
                out["scales"][n] = float(scale.detach().float().reshape(()).item())
                out["n_fp8"] += 1
                for path, off, rows in parts:
                    out["modules"][f"{layer}|{path}"] = {"tensor": n, "row0": off,
                                                         "rows": codes.shape[0] if rows < 0 else rows}
            elif p.dtype == torch.bfloat16 and (n.endswith("embed_tokens.weight") or n.startswith("lm_head.")):
                out["embed" if "embed_tokens" in n else "lm_head"] = p.data
        return out
