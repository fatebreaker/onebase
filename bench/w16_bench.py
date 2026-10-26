"""RL-MATCHED-KERNELS kernel bench (descriptive; docs/PLAN_MLSYS_V2.md): the fused weight-only FP8 (W8A16) multi-adapter
kernel against the paths a rollout engine has, per projection of Qwen3-8B's shapes, rank r, M tokens, n adapters (decode
batches M <= 64: one token per request, adapters round-robin; larger M: contiguous runs, as bench/multi_bench.py). CUDA-graph
timing, median of replays, L2 flushed (bench/kernel_bench.Timer).
  bf16        torch.mm on bf16 weights, no adapter
  w8a8        vLLM's per-token FP8 quantization + cutlass_scaled_mm, no adapter (the W8A8 engine's base)
  w8a8_ours   our fused W8A8 multi-adapter kernels (fp8_lora_multi.MultiFusedFP8LoRA; decode form for M <= 64)
  marlin      vLLM's FP8 Marlin GEMM (weight-only), no adapter (the weight-only engine's base)
  marlin_st   marlin + vLLM's lora_shrink + lora_expand (the weight-only engine's LoRA path in vLLM)
  w16         ours: w8a16_lora_multi.MultiFusedW8A16LoRA (W8A16 GEMM with the adapters fused)
  w16_base    ours with no adapter in the batch
Parity first: w16 against the fp32 reference (relative error <= 2e-2) and its relative difference to marlin_st's output
(which includes Marlin's bf16 rounding of the per-tensor scale). --tune: eager config search for w16 per (projection, M, n).
    python bench/w16_bench.py --out results/w16_kernel.jsonl [--ms 1,16,64,128,256] [--ns 1,4] [--ranks 64] [--tune]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_bench import SHAPES, Timer, gpu_info, rel_err  # noqa: E402


class _Lin(torch.nn.Module):
    """The attributes vLLM's prepare_fp8_layer_for_marlin reads (weight [K, N], per-tensor scale, sizes)."""

    def __init__(self, w_q, s_w, slices):
        super().__init__()
        self.weight = torch.nn.Parameter(w_q.t().contiguous(), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(s_w.clone(), requires_grad=False)
        self.output_size_per_partition, self.input_size_per_partition = w_q.shape
        self.orig_dtype = torch.bfloat16
        self.logical_widths = list(slices)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--ranks", default="64")
    ap.add_argument("--ms", default="1,16,64,128,256")
    ap.add_argument("--ns", default="1,4")
    ap.add_argument("--pool", type=int, default=8, help="adapter slots resident (max_loras)")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (apply_fp8_marlin_linear,
                                                                                prepare_fp8_layer_for_marlin)

    from lowbit_lora.kernels import fp8_lora_multi as F
    from lowbit_lora.kernels import w8a16_lora_multi as W
    torch.manual_seed(0)
    timer = Timer(warm=5 if a.quick else 20, reps=20 if a.quick else 200)
    import lowbit_lora
    kd = Path(lowbit_lora.__file__).resolve().parent / "kernels"
    sha = lambda f: hashlib.sha256((kd / f).read_bytes()).hexdigest()[:12]   # noqa: E731
    info = {**gpu_info(), "torch": torch.__version__, "job": os.environ.get("SLURM_JOB_ID", ""),
            "kernel_sha": sha("w8a16_lora_multi.py"), "pool": a.pool, "tuned": a.tune}
    ms = [int(m) for m in a.ms.split(",")]
    L = a.pool
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_v = LoRAKernelMeta.make(L, max(ms), device="cuda")
    for proj, (K, slices) in SHAPES[a.model].items():
        N, S = sum(slices), len(slices)
        w = torch.randn(N, K, device="cuda") * 0.02
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        w_t = w_q.t()
        w_bf = (w_q.float() * s_w).to(torch.bfloat16)
        lin = _Lin(w_q, s_w, slices)
        prepare_fp8_layer_for_marlin(lin, size_k_first=True)
        for r in [int(x) for x in a.ranks.split(",")]:
            A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
            B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
            a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))
            scale = 1.0                                    # vLLM folds alpha / r into B: compare like with like
            w16 = W.MultiFusedW8A16LoRA(w_q, s_w, A, B, slices, scale, max(ms))
            w8 = F.MultiFusedFP8LoRA(w_q, s_w, A, B, slices, scale, max(ms))
            buf = torch.empty(S * max(ms) * r, dtype=torch.float32, device="cuda")
            for M in ms:
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                mode = "decode" if M <= 64 else "prefill"
                for n in [int(v) for v in a.ns.split(",")] + [0]:
                    if n > L or n > M:
                        continue
                    ar = torch.arange(M, device="cuda")
                    tok = (torch.full((M,), -1, device="cuda") if n == 0 else
                           (ar % n if M <= 64 else (ar * n) // M)).to(torch.int32)
                    meta = F.segments(tok, L)
                    hbuf = buf[: S * M * r].view(S, M, r)
                    row = {**info, "proj": proj, "K": K, "N": N, "r": r, "M": M, "n": n}
                    cs = cg = None
                    if a.tune:
                        cs, cg = w16.tune(x, tok, meta, mode)
                        row["cfg"] = {"s": cs, "g": cg}

                    def w16_fn():
                        return w16(x, tok, meta, mode, cfg_s=cs, cfg_g=cg)

                    def marlin_fn():
                        return apply_fp8_marlin_linear(x, lin.weight, lin.weight_scale, lin.workspace, N, K, None)
                    if n == 0:
                        row["w16_base_us"] = timer.graph_time(w16_fn)
                        row["marlin_us"] = timer.graph_time(marlin_fn)
                        row["bf16_us"] = timer.graph_time(lambda: x @ w_bf.t())

                        def w8a8_fn():
                            xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                            return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)
                        row["w8a8_us"] = timer.graph_time(w8a8_fn)
                    else:
                        meta_v.prepare_tensors(tok)
                        margs = meta_v.meta_args(M, False)

                        def marlin_st_fn():
                            y = marlin_fn()
                            lora_shrink(x, a_st, hbuf, *margs, 1.0)
                            lora_expand(hbuf, b_st, y, *margs, offset_start=0, add_inputs=True)
                            return y
                        ref = W.reference_multi_w16(x, tok, w_q, s_w, A, B, slices, scale)
                        y16 = w16_fn().float().clone()
                        ym = marlin_st_fn().float().clone()
                        row["w16_err"] = rel_err(y16, ref)
                        row["w16_vs_marlin"] = rel_err(y16, ym)
                        row["marlin_err"] = rel_err(ym, ref)
                        row["w16_us"] = timer.graph_time(w16_fn)
                        row["marlin_st_us"] = timer.graph_time(marlin_st_fn)
                        row["w8a8_ours_us"] = timer.graph_time(lambda: w8(x, tok, meta, mode))
                    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                      if k not in info and k != "cfg"}), flush=True)
                    with open(out, "a") as f:
                        f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
