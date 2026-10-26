"""One-layer check of lowbit_lora.train_lora (RL-KERNELS): the fused (K1 + K2) and hybrid (K1 + vendor FP8 GEMM) LoRA
projections against an fp32 reference with the same quantization (per-token e4m3 of x in the forward and of g in the
backward, the engine's codes and scale, h and u rounded to bf16 as the kernels do), for y, dX, dA and dB; B non-zero so
that every term of the backward is exercised. Prints one JSON row per (kind, shape); exits 1 if any relative error exceeds
--tol. GPU only.
    python bench/train_lora_test.py"""
from __future__ import annotations

import argparse
import json
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.03)
    a = ap.parse_args()
    from mlsys import matched_fp8 as mf
    from lowbit_lora import train_lora as tl
    torch.manual_seed(0)
    dev = "cuda"
    bad = 0
    for M, K, N, R in ((512, 4096, 1024, 64), (3000, 4096, 12288, 64), (777, 12288, 4096, 64)):
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        w = torch.randn(N, K, device=dev) * 0.02
        sw = (w.abs().max() / mf.FP8_MAX).float().reshape(())
        wq = (w / sw).clamp(-448, 448).to(torch.float8_e4m3fn)
        A = (torch.randn(R, K, device=dev) * 0.01)
        B = (torch.randn(N, R, device=dev) * 0.01)
        g = (torch.randn(M, N, device=dev) * 1e-3).to(torch.bfloat16)
        scale = 2.0
        # reference: the same quantizations, fp32 GEMMs
        xq, sx = mf._quant_rows(x.float())
        wf = wq.float() * sw
        h = (scale * (x.float() @ A.to(torch.bfloat16).float().t())).to(torch.bfloat16).float()
        y_ref = (xq.float() * sx) @ wf.t() + h @ B.to(torch.bfloat16).float().t()
        gq, sg = mf._quant_rows(g.float())
        u = (scale * (g.float() @ B.to(torch.bfloat16).float())).to(torch.bfloat16).float()
        dx_ref = (gq.float() * sg) @ wf + u @ A.to(torch.bfloat16).float()
        da_ref = u.t() @ x.float()
        db_ref = g.float().t() @ h
        # the projections without an adapter: lowbit_lora.fp8_train with our K2 (LOWBIT_TRAIN_GEMM=triton) and with torch
        from lowbit_lora import fp8_train as ft
        y0_ref = (xq.float() * sx) @ wf.t()
        dx0_ref = (gq.float() * sg) @ wf
        for gemm in ("triton", "torch"):
            ft.GEMM = gemm
            xx = x.clone().requires_grad_(True)
            y = ft._FP8LinearFB.apply(xx, wq, sw, "e4m3")
            y.backward(g)
            rel = lambda t, r: float((t.float() - r).norm() / r.norm())  # noqa: E731
            row = {"kind": f"plain_{gemm}", "M": M, "K": K, "N": N, "y": rel(y, y0_ref), "dx": rel(xx.grad, dx0_ref)}
            row["ok"] = row["y"] <= a.tol and row["dx"] <= a.tol
            bad += not row["ok"]
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        ft.GEMM = "auto"
        # W8A16 (the weight-matched arithmetic without activation quantization): plain and with the adapter
        hw = (scale * (x.float() @ A.to(torch.bfloat16).float().t())).to(torch.bfloat16).float()
        uw = (scale * (g.float() @ B.to(torch.bfloat16).float())).to(torch.bfloat16).float()
        yw_ref = x.float() @ wf.t()
        dxw_ref = g.float() @ wf
        xx = x.clone().requires_grad_(True)
        y = ft._W8A16LinearFn.apply(xx, wq, sw)
        y.backward(g)
        rel = lambda t, r: float((t.float() - r).norm() / r.norm())  # noqa: E731
        row = {"kind": "plain_w8a16", "M": M, "K": K, "N": N, "y": rel(y, yw_ref), "dx": rel(xx.grad, dxw_ref)}
        row["ok"] = row["y"] <= a.tol and row["dx"] <= a.tol
        bad += not row["ok"]
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        xx = x.clone().requires_grad_(True)
        Ap, Bp = A.clone().requires_grad_(True), B.clone().requires_grad_(True)
        y = tl._W16LoRAFn.apply(xx, wq, sw, Ap, Bp, scale)
        y.backward(g)
        row = {"kind": "w16", "M": M, "K": K, "N": N, "R": R, "y": rel(y, yw_ref + hw @ B.to(torch.bfloat16).float().t()),
               "dx": rel(xx.grad, dxw_ref + uw @ A.to(torch.bfloat16).float()), "da": rel(Ap.grad, uw.t() @ x.float()),
               "db": rel(Bp.grad, g.float().t() @ hw)}
        row["ok"] = all(row[k] <= a.tol for k in ("y", "dx", "da", "db"))
        bad += not row["ok"]
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        for kind, fn in (("fused", tl._FusedLoRAFn), ("hybrid", tl._HybridLoRAFn)):
            xx = x.clone().requires_grad_(True)
            Ap, Bp = A.clone().requires_grad_(True), B.clone().requires_grad_(True)
            y = fn.apply(xx, wq, sw, Ap, Bp, scale)
            y.backward(g)
            rel = lambda t, r: float((t.float() - r).norm() / r.norm())  # noqa: E731
            row = {"kind": kind, "M": M, "K": K, "N": N, "R": R, "y": rel(y, y_ref), "dx": rel(xx.grad, dx_ref),
                   "da": rel(Ap.grad, da_ref), "db": rel(Bp.grad, db_ref)}
            ok = all(row[k] <= a.tol for k in ("y", "dx", "da", "db"))
            row["ok"] = ok
            bad += not ok
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
