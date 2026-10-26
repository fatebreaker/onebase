"""GPU check of the FP8- and NVFP4-stored adapter paths of fp8_lora_vllm.VllmFusedFP8LoRA, isolating decode-mode
differences (split-K shrink vs 16-row GEMM tiles), after the H100 benchmark (job 65486657) found the NVFP4-stored
adapters off by ~9% at M <= 64 but exact at M >= 256.
    python tests/gpu_lowprec_adapter_check.py
For each storage format and M: rel. error against an fp32 reference with the stored (dequantized) adapters, in
  decode            (split-K shrink, the decode GEMM config)
  decode, split 1   (an op built with split_k=1: no split-K)
  decode, big tiles (split-K shrink, the GEMM forced to the M=256 config)
  prefill           (split 1, the GEMM config pick_cfg gives for this M)
"""
from __future__ import annotations

import torch

from lowbit_lora.kernels import fp8_lora_multi as FM
from lowbit_lora.kernels import fp8_lora_vllm as F


def main() -> int:
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    K, slices, L = 4096, (4096, 1024, 1024), 8
    N, S = sum(slices), len(slices)
    w = torch.randn(N, K, device="cuda") * 0.02
    s_w = (w.abs().max() / 448.0).reshape(1).float()
    w_q = (w / s_w).to(torch.float8_e4m3fn)
    meta = LoRAKernelMeta.make(L, 256, device="cuda")
    worst = {}
    for r in (16, 64):
        A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
        B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
        a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
        offs = [sum(slices[:s]) for s in range(S)]
        b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))
        for fmt in ("fp8", "fp4"):
            kw = {"lora_fp8": fmt == "fp8", "lora_fp4": fmt == "fp4"}
            op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, 256, **kw)
            op1 = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, 256, split_k=1, **kw)
            if fmt == "fp8":
                Ad = torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(op.a8, op.sa)], 1)
                Bd = torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(op.b8, op.sb)], 1)
            else:
                from lowbit_lora.kernels.nvfp4_lora_vllm import nvfp4_dequant_torch as dq4
                Ad = torch.cat([torch.stack([dq4(q[i, 0], qs[i, 0], 1.0) * op.ag[s, i] for i in range(L)])
                                for s, (q, qs) in enumerate(zip(op.a4, op.a4s))], 1)
                Bd = torch.cat([torch.stack([dq4(q[i, 0], qs[i, 0], 1.0) * op.bg[s, i] for i in range(L)])
                                for s, (q, qs) in enumerate(zip(op.b4, op.b4s))], 1)
            for M in (1, 16, 64, 256):
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                tok = (torch.arange(M, device="cuda") % min(4, L)).to(torch.int32)
                meta.prepare_tensors(tok)
                margs = meta.meta_args(M, False)
                ref = FM.reference_multi(x, tok, w_q, s_w, Ad, Bd, slices, 1.0)
                lora_part = ref - FM.reference_multi(x, torch.full_like(tok, -1), w_q, s_w, Ad, Bd, slices, 1.0)
                big = F.pick_cfg(256, N, K, r, slices)[1]
                runs = {"prefill": lambda: op(x, margs, mode="prefill")}
                if M <= 64:   # split-K (decode mode) runs only for M <= 64; its scratch is sized for that
                    runs.update({"decode": lambda: op(x, margs, mode="decode"),
                                 "decode, split 1": lambda: op1(x, margs, mode="decode"),
                                 "decode, big tiles": lambda: op(x, margs, mode="decode", cfg_g=big)})
                errs = {}
                for name, fn in runs.items():
                    y = fn().float()
                    errs[name] = float((y - ref).norm() / ref.norm())
                    worst[(fmt, name)] = max(worst.get((fmt, name), 0.0), errs[name])
                print(f"r={r} {fmt} M={M:4d} (LoRA share of |y| {float(lora_part.norm() / ref.norm()):.3f}): "
                      + "  ".join(f"{k} {v:.2e}" for k, v in errs.items()), flush=True)
    bad = {k: v for k, v in worst.items() if v > 2e-2}
    print("PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
