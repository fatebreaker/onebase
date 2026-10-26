"""Low-precision copies of vLLM's stacked LoRA buffers, for the fused kernels' FP8- and NVFP4-stored adapter modes.

vLLM keeps each slice's adapters as lora_a_stacked[s] [L, 1, R, K] and lora_b_stacked[s] [L, 1, N_s, R] (bf16). An
AdapterStore holds, per slice:
  fp8:   E4M3 copies with one fp32 scale per row of A ([L, R]) and of B ([L, N_s]) (P0-NUMERICS's lora_fp8 format);
  nvfp4: E2M1 codes packed along K (A: [L, 1, R, K/2]) and along R (B: [L, 1, N_s, R/2]), E4M3 scales per 16 elements,
         and one inverse global scale per adapter and slice ([S, L]) (nvfp4_lora_vllm.nvfp4_quant_batched).
and the device pointer tables the kernels read (a_ptrs, b_ptrs: the codes; sa_ptrs, sb_ptrs: the scales; ag, bg: the
NVFP4 inverse globals). requantize(index) refreshes one slot (or all) after vLLM writes or clears it; after release(),
vLLM writes every slot through one staging slot and the copies are the only per-slot storage.
"""
from __future__ import annotations

import torch

from lowbit_lora.kernels.nvfp4_lora_vllm import nvfp4_quant_batched

FP8_MAX = 448.0


class AdapterStore:
    def __init__(self, a_stacked, b_stacked, fmt: str):
        assert fmt in ("bf16", "fp8", "nvfp4")
        self.fmt, self.a_stacked, self.b_stacked = fmt, a_stacked, b_stacked
        self.staged = False           # release(): a_stacked / b_stacked are one staging slot, not vLLM's L slots
        dev = a_stacked[0].device
        self.S = len(a_stacked)
        self.L, _, self.R, self.K = a_stacked[0].shape
        widths = [t.shape[2] for t in b_stacked]
        ptrs = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)  # noqa: E731
        self.ag = self.bg = torch.zeros(1, dtype=torch.float32, device=dev)
        if fmt == "bf16":
            self.a_ptrs, self.b_ptrs = ptrs(a_stacked), ptrs(b_stacked)
            self.sa_ptrs = self.sb_ptrs = self.a_ptrs                                   # unused
            return
        if fmt == "fp8":
            self.a_q = tuple(torch.zeros(t.shape, dtype=torch.float8_e4m3fn, device=dev) for t in a_stacked)
            self.b_q = tuple(torch.zeros(t.shape, dtype=torch.float8_e4m3fn, device=dev) for t in b_stacked)
            self.a_s = tuple(torch.zeros(self.L, self.R, dtype=torch.float32, device=dev) for _ in a_stacked)
            self.b_s = tuple(torch.zeros(self.L, w, dtype=torch.float32, device=dev) for w in widths)
        else:
            assert self.R % 16 == 0 and self.K % 16 == 0
            self.a_q = tuple(torch.zeros(self.L, 1, self.R, self.K // 2, dtype=torch.uint8, device=dev)
                             for _ in a_stacked)
            self.a_s = tuple(torch.zeros(self.L, 1, self.R, self.K // 16, dtype=torch.float8_e4m3fn, device=dev)
                             for _ in a_stacked)
            self.b_q = tuple(torch.zeros(self.L, 1, w, self.R // 2, dtype=torch.uint8, device=dev) for w in widths)
            self.b_s = tuple(torch.zeros(self.L, 1, w, self.R // 16, dtype=torch.float8_e4m3fn, device=dev)
                             for w in widths)
            self.ag = torch.zeros(self.S, self.L, dtype=torch.float32, device=dev)
            self.bg = torch.zeros(self.S, self.L, dtype=torch.float32, device=dev)
        self.a_ptrs, self.b_ptrs = ptrs(self.a_q), ptrs(self.b_q)
        self.sa_ptrs, self.sb_ptrs = ptrs(self.a_s), ptrs(self.b_s)
        self.requantize()

    def release(self, a_stage, b_stage) -> None:
        """Serve from the low-precision copies alone: vLLM's per-slot bf16 buffers give way to one staging slot
        (a_stage[s] [1, 1, R, K], b_stage[s] [1, 1, N_s, R]) that vLLM writes every adapter through, and
        requantize(index) then reads slot 0 of the staging buffers into slot index of the copies."""
        assert self.fmt != "bf16"
        self.a_stacked, self.b_stacked, self.staged = a_stage, b_stage, True

    @torch.no_grad()
    def requantize(self, index: int | None = None) -> None:
        if self.fmt == "bf16":
            return
        assert not (self.staged and index is None)
        idx = slice(None) if index is None else slice(index, index + 1)
        sidx = slice(0, 1) if self.staged else idx
        for src, q, sc, g in ((self.a_stacked, self.a_q, self.a_s, self.ag), (self.b_stacked, self.b_q, self.b_s, self.bg)):
            for s in range(self.S):
                t = src[s][sidx, 0]
                if self.fmt == "fp8":
                    tf = t.float()
                    scale = (tf.abs().amax(-1) / FP8_MAX).clamp_min(1e-12)
                    q[s][idx, 0] = (tf / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                    sc[s][idx] = scale
                else:
                    packed, sf, inv_g = nvfp4_quant_batched(t)
                    q[s][idx, 0], sc[s][idx, 0], g[s, idx] = packed, sf, inv_g

    def dequantized(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(A [L, S*R, K], B [L, N, R]) as the kernels see them (fp32), for references."""
        from lowbit_lora.kernels.nvfp4_lora_vllm import nvfp4_dequant_torch
        if self.fmt == "bf16":
            return (torch.cat([t[:, 0].float() for t in self.a_stacked], 1),
                    torch.cat([t[:, 0].float() for t in self.b_stacked], 1))
        if self.fmt == "fp8":
            return (torch.cat([q[:, 0].float() * s[..., None] for q, s in zip(self.a_q, self.a_s)], 1),
                    torch.cat([q[:, 0].float() * s[..., None] for q, s in zip(self.b_q, self.b_s)], 1))
        deq = lambda q, s, g: torch.stack([nvfp4_dequant_torch(q[i, 0], s[i, 0], 1.0) * g[i]   # noqa: E731
                                           for i in range(self.L)])
        return (torch.cat([deq(q, s, self.ag[k]) for k, (q, s) in enumerate(zip(self.a_q, self.a_s))], 1),
                torch.cat([deq(q, s, self.bg[k]) for k, (q, s) in enumerate(zip(self.b_q, self.b_s))], 1))
