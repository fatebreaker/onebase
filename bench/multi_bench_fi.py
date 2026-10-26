"""multi_bench.py (copied, unchanged otherwise) plus two variants (KERNEL-EXTERNAL, docs/REGISTRY.md; mock review R1 v22 M3):
  fi         ref + FlashInfer's segment GEMM (SegmentGEMMWrapper, the SGMV kernel of S-LoRA/Punica-style serving) for the
             shrink (all slices at once) and each slice's expand, on the batch sorted by adapter (given free: FlashInfer's best
             case), then the add; timed in a CUDA graph, or eagerly (marked fi_eager) if FlashInfer cannot be captured
  plugina16  the plugin's weight-only fused path (fp8_lora_vllm A16, LOWBIT_A16_MAX_M), bf16 adapters, against the W8A16 reference
multi_bench.py's header follows.
Multi-adapter microbenchmark (exploratory, Phase 1): the fused multi-adapter FP8 + LoRA kernels against vLLM's
stock path, on batches whose tokens use n distinct adapters out of a pool of L resident slots: round-robin (one
token per request) for M <= 64, contiguous runs (prefill chunks of n requests) for larger M.

    python bench/multi_bench.py --out results/multi_kernel.jsonl [--ranks 16,64] [--ms 16,64,256] [--ns 1,4,16,32]
Per (projection, r, M, n):
  ref     scaled_fp8_quant + cutlass_scaled_mm                         (no adapter: the merged cost)
  stock   ref + vLLM lora_shrink + lora_expand with n adapters (Punica metadata)
  fused   our multi-adapter shrink + GEMM-with-epilogue, bf16 adapters (decode form for M <= 64, else prefill form)
  fused8  the same with FP8-stored A and B (per-row scales)
  plugin  the kernels the vLLM plugin runs (fp8_lora_vllm.VllmFusedFP8LoRA: vLLM's stacked buffers and Punica
          metadata, adapter-sorted tiles, pick_cfg configs), bf16 adapters
  plugin8 the same with FP8-stored adapters (lora_fp8: E4M3, per-row scales)
  stock8  ref + vLLM's own FP8 LoRA kernels (lora_shrink_fp8 / lora_expand_fp8, weight-only, per-channel scales) on
          the same FP8 adapters as plugin8: vLLM ships them but does not wire them into dense LoRA layers
  plugin4 the same with NVFP4-stored adapters (lora_fp4: E2M1, E4M3 block scales of 16, a global per adapter/slice)
  pluginh the plugin's hybrid path: vLLM's quantize + CUTLASS for the base, then our shrink and expand-add kernels
  plugin8h / plugin4h  the hybrid path with FP8- / NVFP4-stored adapters
  pluginq / plugin8q / plugin4q  the "hybridq" path: our shrink kernel also writes the activation codes and scales
          (bit-identical to vLLM's), CUTLASS runs on them, then our expand-add: no separate quantize kernel
Parity against an fp32 reference first (relative error <= 2e-2); timing as bench/kernel_bench.py (CUDA graph,
median of 200 replays, L2 flushed). Default kernel configs (no tuning) unless --tune.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_bench import SHAPES, Timer, gpu_info, rel_err  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--ms", default="16,64,256")
    ap.add_argument("--ns", default="1,4,16,32")
    ap.add_argument("--pool", type=int, default=32, help="adapter slots resident (max_loras)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tune", action="store_true", help="eager config search per (projection, r, M, n) before timing")
    ap.add_argument("--variants", default="ref,stock,fused,fused8,plugin")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = set(a.variants.split(","))
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_expand_fp8, lora_shrink, lora_shrink_fp8

    from lowbit_lora.kernels import fp8_lora_multi as F
    from lowbit_lora.kernels.fp8_lora import quantize_rows_fp8
    from lowbit_lora.kernels.fp8_lora_vllm import VllmFusedFP8LoRA
    torch.manual_seed(0)
    timer = Timer(warm=5 if a.quick else 20, reps=20 if a.quick else 200)
    import lowbit_lora   # the imported copy (jobs run a snapshot taken at job start, jobs/lib.sh)
    kd = Path(lowbit_lora.__file__).resolve().parent / "kernels"
    sha = lambda f: hashlib.sha256((kd / f).read_bytes()).hexdigest()[:12]   # noqa: E731
    info = {**gpu_info(), "torch": torch.__version__, "job": os.environ.get("SLURM_JOB_ID", ""),
            "kernel_sha": sha("fp8_lora_multi.py"), "plugin_sha": sha("fp8_lora_vllm.py"), "pool": a.pool}
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
        for r in [int(x) for x in a.ranks.split(",")]:
            A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
            B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
            qa = [quantize_rows_fp8(A[i]) for i in range(L)]
            qb = [quantize_rows_fp8(B[i]) for i in range(L)]
            A8, SA = torch.stack([t for t, _ in qa]), torch.stack([s for _, s in qa])
            B8, SB = torch.stack([t for t, _ in qb]), torch.stack([s for _, s in qb])
            a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))
            scale = 2.0
            fused = F.MultiFusedFP8LoRA(w_q, s_w, A, B, slices, scale, max(ms))
            fused8 = F.MultiFusedFP8LoRA(w_q, s_w, A8, B8, slices, scale, max(ms), a_scale=SA, b_scale=SB)
            plug = VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max(ms))           # vLLM folds the LoRA scaling into B

            def vendor(xx):   # vLLM's per-token FP8 quantization + CUTLASS (the hybrid's base GEMM)
                xq_, sx_ = ops.scaled_fp8_quant(xx, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(xq_, w_t, sx_, s_w, torch.bfloat16)
            plug.base_fn = vendor

            def vendor_q(xq_, sx_):   # CUTLASS on our codes and per-token scales (the hybridq base GEMM)
                return ops.cutlass_scaled_mm(xq_, w_t, sx_, s_w, torch.bfloat16)
            plug.base_q_fn = vendor_q
            plug8 = (VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max(ms), lora_fp8=True)
                     if {"plugin8", "plugin8h", "plugin8q", "stock8"} & want else None)
            plug4 = (VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max(ms), lora_fp4=True)
                     if {"plugin4", "plugin4h", "plugin4q"} & want else None)
            if plug4 is not None:
                from lowbit_lora.kernels.nvfp4_lora_vllm import nvfp4_dequant_torch as dq4
                A4d = torch.cat([torch.stack([dq4(q[i, 0], qs[i, 0], 1.0) * plug4.ag[s, i] for i in range(L)])
                                 for s, (q, qs) in enumerate(zip(plug4.a4, plug4.a4s))], dim=1)
                B4d = torch.cat([torch.stack([dq4(q[i, 0], qs[i, 0], 1.0) * plug4.bg[s, i] for i in range(L)])
                                 for s, (q, qs) in enumerate(zip(plug4.b4, plug4.b4s))], dim=1)
            if plug8 is not None:   # its dequantized adapters, for the parity reference
                A8d = torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(plug8.a8, plug8.sa)], dim=1)
                B8d = torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(plug8.b8, plug8.sb)], dim=1)
            buf = torch.empty(S * max(ms) * r, dtype=torch.float32, device="cuda")
            for M in ms:
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                mode = "decode" if M <= 64 else "prefill"
                for n in [int(v) for v in a.ns.split(",")]:
                    if n > L or n > M:
                        continue
                    # decode batches (M <= 64): one token per request, adapters round-robin; larger batches are
                    # dominated by prefill chunks, i.e. contiguous runs of one request's tokens (n runs of M/n)
                    ar = torch.arange(M, device="cuda")
                    tok = (ar % n if M <= 64 else (ar * n) // M).to(torch.int32)
                    meta = F.segments(tok, L)
                    meta_v.prepare_tensors(tok)
                    margs = meta_v.meta_args(M, False)
                    hbuf = buf[: S * M * r].view(S, M, r)

                    def ref_fn():
                        xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                        return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)

                    def stock_fn():
                        y = ref_fn()
                        lora_shrink(x, a_st, hbuf, *margs, scale)
                        lora_expand(hbuf, b_st, y, *margs, offset_start=0, add_inputs=True)
                        return y

                    y_ref = F.reference_multi(x, tok, w_q, s_w, A, B, slices, scale)
                    y_ref8 = F.reference_multi(x, tok, w_q, s_w, A8, B8, slices, scale, a_scale=SA, b_scale=SB)
                    cfg = {"fused": (None, None), "fused8": (None, None)}
                    if a.tune:
                        cfg = {"fused": fused.tune(x, tok, meta, mode), "fused8": fused8.tune(x, tok, meta, mode)}
                    cf, c8 = cfg["fused"], cfg["fused8"]
                    errs, times = {}, {}
                    if "ref" in want:
                        times["ref"] = timer.graph_time(ref_fn)
                    if "stock" in want:
                        errs["stock"] = rel_err(stock_fn(), y_ref)
                        times["stock"] = timer.graph_time(stock_fn)
                    if "fused" in want:
                        errs["fused"] = rel_err(fused(x, tok, meta, mode, *cf).clone(), y_ref)
                        times["fused"] = timer.graph_time(lambda: fused(x, tok, meta, mode, *cf))
                    if "fused8" in want:
                        errs["fused8"] = rel_err(fused8(x, tok, meta, mode, *c8).clone(), y_ref8)
                        times["fused8"] = timer.graph_time(lambda: fused8(x, tok, meta, mode, *c8))
                    if "plugin" in want:
                        y_ref1 = F.reference_multi(x, tok, w_q, s_w, A, B, slices, 1.0)
                        errs["plugin"] = rel_err(plug(x, margs).clone(), y_ref1)
                        times["plugin"] = timer.graph_time(lambda: plug._run(x, margs, hybrid=False))
                    if "pluginh" in want:
                        errs["pluginh"] = rel_err(plug._run(x, margs, hybrid=True).clone(), y_ref1)
                        times["pluginh"] = timer.graph_time(lambda: plug._run(x, margs, hybrid=True))
                    if "pluginq" in want:
                        errs["pluginq"] = rel_err(plug._run(x, margs, hybrid="q").clone(), y_ref1)
                        times["pluginq"] = timer.graph_time(lambda: plug._run(x, margs, hybrid="q"))
                    if plug8 is not None:
                        plug8.base_fn, plug8.base_q_fn = vendor, vendor_q
                        y_ref8 = F.reference_multi(x, tok, w_q, s_w, A8d, B8d, slices, 1.0)
                    if "plugin8h" in want and plug8 is not None:
                        errs["plugin8h"] = rel_err(plug8._run(x, margs, hybrid=True).clone(), y_ref8)
                        times["plugin8h"] = timer.graph_time(lambda: plug8._run(x, margs, hybrid=True))
                    if "plugin8q" in want and plug8 is not None:
                        errs["plugin8q"] = rel_err(plug8._run(x, margs, hybrid="q").clone(), y_ref8)
                        times["plugin8q"] = timer.graph_time(lambda: plug8._run(x, margs, hybrid="q"))
                    if "plugin8" in want:
                        errs["plugin8"] = rel_err(plug8._run(x, margs, hybrid=False).clone(), y_ref8)
                        times["plugin8"] = timer.graph_time(lambda: plug8._run(x, margs, hybrid=False))
                    if "stock8" in want:
                        def stock8_fn():
                            y = ref_fn()
                            lora_shrink_fp8(x, list(plug8.a8), hbuf, *margs, 1.0, list(plug8.sa), per_channel_quant=True)
                            lora_expand_fp8(hbuf, list(plug8.b8), y, *margs, list(plug8.sb), offset_start=0,
                                            add_inputs=True, per_channel_quant=True)
                            return y
                        errs["stock8"] = rel_err(stock8_fn(), y_ref8)
                        times["stock8"] = timer.graph_time(stock8_fn)
                    if plug4 is not None:
                        plug4.base_fn, plug4.base_q_fn = vendor, vendor_q
                        y_ref4 = F.reference_multi(x, tok, w_q, s_w, A4d, B4d, slices, 1.0)
                        errs["plugin4"] = rel_err(plug4._run(x, margs, hybrid=False).clone(), y_ref4)
                        times["plugin4"] = timer.graph_time(lambda: plug4._run(x, margs, hybrid=False))
                        if "plugin4h" in want:
                            errs["plugin4h"] = rel_err(plug4._run(x, margs, hybrid=True).clone(), y_ref4)
                            times["plugin4h"] = timer.graph_time(lambda: plug4._run(x, margs, hybrid=True))
                        if "plugin4q" in want:
                            errs["plugin4q"] = rel_err(plug4._run(x, margs, hybrid="q").clone(), y_ref4)
                            times["plugin4q"] = timer.graph_time(lambda: plug4._run(x, margs, hybrid="q"))
                    if "fi" in want:
                        import flashinfer
                        if "fi_wrap" not in dir():
                            fi_wrap = flashinfer.SegmentGEMMWrapper(torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"))
                        order = torch.argsort(tok, stable=True)
                        xs, toks = x[order].contiguous(), tok[order]
                        ids, cnts = torch.unique_consecutive(toks, return_counts=True)
                        seg_lens, widx, nseg = cnts.to(torch.int64), ids.to(torch.int64), int(ids.numel())
                        Bs = [B[:, offs[s_]:offs[s_] + slices[s_]].contiguous() for s_ in range(S)]

                        def fi_fn():
                            xq, sx = ops.scaled_fp8_quant(xs, use_per_token_if_dynamic=True)
                            y = ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)
                            h = fi_wrap.run(xs, A, nseg, True, seg_lens=seg_lens, weight_indices=widx) * scale
                            for s_ in range(S):
                                ys = fi_wrap.run(h[:, s_ * r:(s_ + 1) * r].contiguous(), Bs[s_], nseg, True, seg_lens=seg_lens,
                                                 weight_indices=widx)
                                y[:, offs[s_]:offs[s_] + slices[s_]] += ys
                            return y
                        y_ref_s = F.reference_multi(xs, toks, w_q, s_w, A, B, slices, scale)
                        errs["fi"] = rel_err(fi_fn(), y_ref_s)
                        try:
                            times["fi"] = timer.graph_time(fi_fn)
                        except Exception as ex:   # noqa: BLE001  FlashInfer not capturable: eager median instead
                            print(f"fi not capturable ({type(ex).__name__}: {str(ex)[:80]}); eager timing", flush=True)
                            torch.cuda.synchronize()
                            ts = []
                            for _ in range(timer.reps):
                                timer.flush.zero_()
                                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                e0.record(); fi_fn(); e1.record(); e1.synchronize()
                                ts.append(1000 * e0.elapsed_time(e1))
                            import statistics as _st
                            times["fi_eager"] = _st.median(ts)
                    if "plugina16" in want and M <= 64:
                        y_ref16 = F.reference_multi(x, tok, w_q, s_w, A, B, slices, 1.0, xq=x.float(), sx=torch.ones(M, device="cuda"))
                        errs["plugina16"] = rel_err(plug._run(x, margs, a16=True).clone(), y_ref16)
                        times["plugina16"] = timer.graph_time(lambda: plug._run(x, margs, a16=True))
                    if "pluginb" in want:   # the plugin's kernels with no adapter token: our quantize + GEMM alone, the
                        # baseline that separates the LoRA overhead from our base GEMM's speed (review W4a). Last,
                        # because it rewrites the metadata buffers the other variants read.
                        meta_v.prepare_tensors(torch.full((M,), -1, dtype=torch.int32, device="cuda"))
                        margs0 = meta_v.meta_args(M, False)
                        errs["pluginb"] = rel_err(plug._run(x, margs0, hybrid=False).clone(), ref_fn())
                        times["pluginb"] = timer.graph_time(lambda: plug._run(x, margs0, hybrid=False))
                    rec = {**info, "model": a.model, "proj": proj, "N": N, "K": K, "r": r, "M": M, "n": n,
                           "mode": mode, "tuned": a.tune, "cfg": cfg, "t_us": {k: round(v, 2) for k, v in times.items()},
                           "rel_err": {k: float(f"{v:.3g}") for k, v in errs.items()},
                           "ok": {k: v <= 2e-2 for k, v in errs.items()},
                           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    print(json.dumps({k: rec[k] for k in ("proj", "r", "M", "n", "t_us", "ok")}), flush=True)
                    with out.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
