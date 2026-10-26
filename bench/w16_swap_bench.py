"""MATCHED-HOPPER-KERNEL bench (docs/REGISTRY.md; descriptive): the swap-AB W8A16 GEMM (kernels/w8a16_swap.py) against the
matched engine's current W8A16 GEMM (w8a16_lora_multi.w8a16_linear), vLLM's W8A8 base (per-token FP8 quantization +
cutlass_scaled_mm), vLLM's FP8 Marlin (weight-only) and bf16 torch.mm, per projection of Qwen3-8B's shapes (= Qwen3.5-9B's
dense dims) and batch size M, no adapter. CUDA-graph timing, median of replays, L2 flushed (bench/kernel_bench.Timer).
Parity: each swap configuration's output against the fp32 reference (relative error) and against the current kernel's output
(fraction of bit-identical bf16 elements; they differ only in the order of fp32 additions inside an MMA).
--tune: a config sweep per (projection, M) with a short timer, the best re-timed with the full timer.
    python bench/w16_swap_bench.py --out results/w16_swap.jsonl [--ms 1,8,16,32,64,128,256,512] [--tune]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_bench import SHAPES, Timer, gpu_info, rel_err  # noqa: E402
from w16_bench import _Lin  # noqa: E402


def candidates(M: int, N: int, K: int) -> list[dict]:
    bms = [16] if M <= 16 else [16, 32] if M <= 32 else [32, 64] if M <= 64 else [64, 128]
    sks = [1, 2, 4] if M <= 64 and N <= 8192 else [1]
    out = []
    for bm, bn, bk, nw, ns, sk in itertools.product(bms, (64, 128, 256), (64, 128, 256), (4, 8), (3, 4, 5), sks):
        if bn == 64 and nw == 8:              # one warpgroup's wgmma tile is 64 rows
            continue
        smem = ns * (bn * bk + bm * bk * 2)   # FP8 weight tile + bf16 activation tile per stage
        if smem > 220 * 1024 or (sk > 1 and K // sk < bk):
            continue
        out.append({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk, "num_warps": nw, "num_stages": ns})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--ms", default="1,8,16,32,64,128,256,512")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (apply_fp8_marlin_linear,
                                                                                prepare_fp8_layer_for_marlin)

    from lowbit_lora.kernels import w8a16_swap as SW
    from lowbit_lora.kernels.w8a16_lora_multi import w8a16_linear
    torch.manual_seed(0)
    full, quick = Timer(warm=20, reps=200), Timer(warm=5, reps=30)
    import lowbit_lora
    kd = Path(lowbit_lora.__file__).resolve().parent / "kernels"
    sha = lambda f: hashlib.sha256((kd / f).read_bytes()).hexdigest()[:12]   # noqa: E731
    info = {**gpu_info(), "torch": torch.__version__, "job": os.environ.get("SLURM_JOB_ID", ""),
            "swap_sha": sha("w8a16_swap.py"), "w16_sha": sha("w8a16_lora_multi.py"), "tuned": a.tune}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for proj, (K, slices) in SHAPES[a.model].items():
        N = sum(slices)
        w = torch.randn(N, K, device="cuda") * 0.02
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        w_t = w_q.t()
        w_bf = (w_q.float() * s_w).to(torch.bfloat16)
        lin = _Lin(w_q, s_w, slices)
        prepare_fp8_layer_for_marlin(lin, size_k_first=True)
        for M in [int(m) for m in a.ms.split(",")]:
            x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            ref = (x.float() @ w_q.float().t()) * s_w
            y_cur = w8a16_linear(x, w_q, s_w)
            row = {**info, "proj": proj, "K": K, "N": N, "M": M}
            row["cur_err"] = rel_err(y_cur.float(), ref)
            row["cur_us"] = full.graph_time(lambda: w8a16_linear(x, w_q, s_w))
            row["bf16_us"] = full.graph_time(lambda: x @ w_bf.t())
            row["marlin_us"] = full.graph_time(lambda: apply_fp8_marlin_linear(x, lin.weight, lin.weight_scale, lin.workspace,
                                                                                N, K, None))

            def w8a8_fn():
                xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)
            row["w8a8_us"] = full.graph_time(w8a8_fn)
            cfgs = [("default", SW.swap_cfg(M, N, K))]
            if a.tune:
                best, best_t, n_ok = None, float("inf"), 0
                for c in candidates(M, N, K):
                    try:
                        y = SW.w8a16_linear_swap(x, w_q, s_w, c)
                        if rel_err(y.float(), ref) > 2e-2:
                            print("PARITY FAIL", proj, M, c, flush=True)
                            continue
                        t = quick.graph_time(lambda c=c: SW.w8a16_linear_swap(x, w_q, s_w, c))
                    except Exception as e:  # noqa: BLE001  (out of resources and the like)
                        print("skip", c, type(e).__name__, str(e)[:120], flush=True)
                        continue
                    n_ok += 1
                    if t < best_t:
                        best, best_t = c, t
                row["n_cfgs"] = n_ok
                if best:
                    cfgs.append(("best", best))
            for tag, c in cfgs:
                y = SW.w8a16_linear_swap(x, w_q, s_w, c)
                row[f"swap_{tag}_err"] = rel_err(y.float(), ref)
                row[f"swap_{tag}_same_as_cur"] = (y == y_cur).float().mean().item()
                row[f"swap_{tag}_us"] = full.graph_time(lambda c=c: SW.w8a16_linear_swap(x, w_q, s_w, c))
                row[f"swap_{tag}_cfg"] = c
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k not in info and not k.endswith("_cfg")}), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
