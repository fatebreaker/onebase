"""GPU check of the two NVFP4 tile-dequantization formulations (GPU, seconds), to find where the vectorized one is
safe: MODE 0 = load packed bytes [ROWS, BK/2] and scales [ROWS, BK/16], then tl.join / tl.reshape (nvfp4_lora_vllm.
_dequant_tile; miscompiled somewhere at small tiles, job 65486657); MODE 1 = gather each element's byte, nibble and
scale (correct everywhere, but ~16x the load instructions). Both are checked stored directly, transposed (as the
expand's B tile), and fed into a tl.dot (as the shrink and the expand use them), against a torch reference.
    python tests/gpu_dequant_check.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from lowbit_lora.kernels.nvfp4_lora_vllm import _dequant_tile, _e2m1_value, nvfp4_dequant_torch, nvfp4_quant_torch


@triton.jit
def _deq(P, S, X, OUT, OUTT, DOT, ROWS: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    r = tl.arange(0, ROWS)
    if MODE == 0:
        p = tl.load(P + r[:, None] * (BK // 2) + tl.arange(0, BK // 2)[None, :])
        s = tl.load(S + r[:, None] * (BK // 16) + tl.arange(0, BK // 16)[None, :])
        d = _dequant_tile(p, s, ROWS, BK)
    else:
        k = tl.arange(0, BK)
        b = tl.load(P + r[:, None] * (BK // 2) + (k // 2)[None, :])
        nib = (b.to(tl.int32) >> ((k % 2) * 4)[None, :]) & 15
        s = tl.load(S + r[:, None] * (BK // 16) + (k // 16)[None, :])
        d = _e2m1_value(nib) * s.to(tl.float32)
    k = tl.arange(0, BK)
    tl.store(OUT + r[:, None] * BK + k[None, :], d)
    tl.store(OUTT + k[:, None] * ROWS + r[None, :], tl.trans(d))
    m = tl.arange(0, 16)
    x = tl.load(X + m[:, None] * BK + k[None, :])                           # [16, BK] bf16
    acc = tl.dot(x, tl.trans(d.to(tl.bfloat16)))                            # [16, ROWS]
    tl.store(DOT + m[:, None] * ROWS + r[None, :], acc)


def main() -> int:
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    bad = 0
    for rows, bk in [(16, 16), (64, 16), (128, 16), (16, 64), (64, 64), (128, 64), (16, 256), (64, 256), (128, 128)]:
        w = torch.randn(rows, bk, device="cuda")
        p, s = nvfp4_quant_torch(w, 448.0 * 6 / float(w.abs().max()))
        ref = nvfp4_dequant_torch(p, s, 1.0)                                   # [rows, bk], no global
        x = torch.randn(16, bk, device="cuda").to(torch.bfloat16)
        ref_dot = x.float() @ ref.to(torch.bfloat16).float().t()
        res = []
        for mode in (0, 1):
            out = torch.empty(rows, bk, device="cuda")
            outt = torch.empty(bk, rows, device="cuda")
            dot = torch.empty(16, rows, device="cuda")
            _deq[(1,)](p, s, x, out, outt, dot, ROWS=rows, BK=bk, MODE=mode)
            e1 = float((out - ref).abs().max())
            e2 = float((outt.t() - ref).abs().max())
            e3 = float((dot - ref_dot).norm() / ref_dot.norm())
            ok = e1 == 0 and e2 == 0 and e3 < 1e-2
            bad += not ok
            res.append(f"mode {mode}: direct {e1:.1e} transposed {e2:.1e} dot {e3:.1e} {'ok' if ok else 'WRONG'}")
        print(f"tile [{rows:3d} x {bk:3d}]  " + "  |  ".join(res), flush=True)
    print("PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
