"""The PDL decode path (kernels/fp8_lora_pdl.py via lowbit_lora/hopper_pdl.run_pdl) on the H100 against vLLM's and the
plugin's other paths, per projection, rank r, batch size M, number of distinct adapters n and adapter storage format.
CUDA-graph timings (bench/kernel_bench.Timer: graph replays, L2 flushed before each, median), microseconds, of
  ref           vLLM's per-token quantize + CUTLASS's FP8 GEMM, no LoRA ("CUTLASS alone")
  stock         ref, then vLLM's lora_shrink / lora_expand with their default configurations
  stock_tuned   the same with vLLM's H100-tuned configurations (VLLM_TUNED_CONFIG_FOLDER, results/vllm_lora_tuned/
                Qwen3.5-9B-text_h100 as STOCK-TUNED-H100; vLLM's nearest keys for Qwen3-8B's qkv)
  stock_dual    vLLM's opt-in dual-stream LoRA (VLLM_LORA_ENABLE_DUAL_STREAM=1: the base GEMM on the main stream, shrink and
                expand on an aux stream with PDL, joined, then added), tuned configurations
  hybrid        ref, then our shrink and expand-add (what the plugin's H100 dispatch keeps today)
  swap          our K1, then the swap-AB K2 (HOPPER-SWAP-K2) at its swept tiles (results/swap_k2.jsonl)
  pdl           the new path: the shrink kernel, then the GEMM with programmatic dependent launch; at its default
                configuration (t_pdl_default) and at the best of a sweep (t_pdl, configuration pdl_best, all candidates
                in pdl_sweep)
  pdl_nopdl     pdl_best without the PDL launch attribute (the same kernels, serialized: what the overlap saves)
  pdl_noshrink  pdl_best's GEMM with LoRA but no adapter-shrink launch (h from the last shrink: the overlap's bound)
  shrink_only   pdl_best's shrink kernel alone
  k2_nolora     the new GEMM without LoRA (quantizing x itself, X_MODE 0, or after the quantization programs, X_MODE 1),
                best of a small tile sweep: our base GEMM against ref (CUTLASS alone)
  coop          the merged fallback kernel at pdl_best's tiles (cooperative launch where the grid fits)
  layer_*       one decoder layer's projections in one graph (L2 flushed once per replay) for ref, stock, stock_tuned,
                hybrid, swap and pdl at their best configurations, one rank-64 adapter
Checks: every output against stock's (bf16 adapters; relative error at most 2e-2), FP8-/NVFP4-stored adapters against the
hybrid path on the same stored adapters; pdl_best repeated bitwise (determinism); the GEMM's own E4M3 codes (X_MODE 0,
debug output) and the quantization programs' codes and scales (X_MODE 1) against vLLM's scaled_fp8_quant, byte for byte.
Stages in priority order (rows appended as each case finishes; cases not started once --budget-min has passed):
  headline  Qwen3-8B's four projections, r 64, M 1/4/8/16, one adapter, bf16: coarse grid, then a refinement
  qwen35    Qwen3.5-9B's two other projections (full-attention qkv, linear-attention in_proj_qkvz), as the headline
  largeM    Qwen3-8B, r 64, M 32/64/128, one adapter (coarse grid)
  adapters  8 and 32 distinct adapters (32 slots), M 8/16/32/64 (the headline's best and its neighbours)
  ranks     r 16 and 128, M 1/16
  formats   FP8- and NVFP4-stored adapters, r 64, M 1/16, n 1 and 8
Before each phase's timings, --workers processes compile its kernels (Triton's compile-only warmup for the new path; one
call of each baseline) into the job's Triton cache; nothing is timed while they run.
    python bench/pdl_k2_bench.py --out results/pdl_k2.jsonl [--budget-min 28] [--workers 8] [--stages ...] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_bench import SHAPES, Timer, gpu_info  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))   # after PYTHONPATH: a job's code snapshot (jobs/lib.sh job_env) comes first
# KERNEL-H200 (2026-10-25): another Hopper GPU reads the H100's configurations from a folder named for its own device
TUNED = Path(os.environ.get("LOWBIT_PDLB_TUNED") or REPO / "results/vllm_lora_tuned/Qwen3.5-9B-text_h100")
MODELS = {"qwen3_8b": SHAPES["qwen3_8b"],
          # Qwen3.5-9B: full-attention qkv (gated q), the linear-attention layers' in_proj_qkvz; o, gate_up, down as Qwen3-8B
          "qwen35_9b": {"qkv": (4096, (8192, 1024, 1024)), "qkvz": (4096, (2048, 2048, 4096, 4096)),
                        "o": (4096, (4096,)), "gate_up": (4096, (12288, 12288)), "down": (12288, (4096,))}}
# KERNEL-H200 amendment -TP (2026-10-25; mock review R3 v27, change 1): Qwen3-8B's per-GPU shapes under tensor parallelism of 2, 4 and
# 8 (vLLM's sharding: qkv and gate_up column-parallel, N split; o and down row-parallel, K split; with LoRA, a column-parallel layer
# keeps the full shrink and splits the expand, a row-parallel one splits the shrink's K and keeps the full expand), single GPU
for _t in (2, 4, 8):
    MODELS[f"qwen3_8b_tp{_t}"] = {"qkv": (4096, (4096 // _t, 1024 // _t, 1024 // _t)), "o": (4096 // _t, (4096,)),
                                  "gate_up": (4096, (12288 // _t, 12288 // _t)), "down": (12288 // _t, (4096,))}
# KERNEL-EXPOSURE (2026-10-25; mock reviews R1 v31 M3, R3 v30 change 3): shapes whose GEMMs are short enough that the shrink
# outlasts K2's main loop -- Qwen3-0.6B (hidden 1024) and Qwen3-1.7B (hidden 2048), 16 query and 8 key-value heads of 128
MODELS["qwen3_0p6b"] = {"qkv": (1024, (2048, 1024, 1024)), "o": (2048, (1024,)), "gate_up": (1024, (3072, 3072)),
                        "down": (3072, (1024,))}
MODELS["qwen3_1p7b"] = {"qkv": (2048, (2048, 1024, 1024)), "o": (2048, (2048,)), "gate_up": (2048, (6144, 6144)),
                        "down": (6144, (2048,))}
# E2E-LLAMA-H100 (2026-10-25; mock review R1 v31 M4a): Llama-3.1-8B's projections (gate_up and down wider than Qwen3-8B's), so the
# decode path runs Llama tiles of its own as vLLM runs Llama-tuned configurations
MODELS["llama31_8b"] = SHAPES["llama31_8b"]
# E2E-72B-H200 amendment -TILES and E2E-TP2-TUNED (2026-10-25): Qwen2.5-72B's projections (hidden 8192, 64 query and 8 key-value heads of
# 128, intermediate 29568) on one GPU and its per-GPU shards at TP2 (column-parallel qkv and gate_up split N, row-parallel o and down K)
MODELS["qwen25_72b"] = {"qkv": (8192, (8192, 1024, 1024)), "o": (8192, (8192,)), "gate_up": (8192, (29568, 29568)), "down": (29568, (8192,))}
MODELS["qwen25_72b_tp2"] = {"qkv": (8192, (4096, 512, 512)), "o": (4096, (8192,)), "gate_up": (8192, (14784, 14784)),
                            "down": (14784, (8192,))}
NSM = 132
FULL = ("headline", "qwen35", "smoke", "llama", "qwen72b", "qwen72b_tp2")   # stages with the full sweep and every extra variant
MODEL_VARS = ("tp", "ranks", "exposure", "floor")     # amendment -TP: also time the exposure model's inputs (no-shrink, shrink alone, K2 alone)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def kernel_sha() -> str:
    h = hashlib.sha256()
    for f in ("lowbit_lora/kernels/fp8_lora_pdl.py", "lowbit_lora/hopper_pdl.py", "lowbit_lora/kernels/fp8_lora_vllm.py",
              "lowbit_lora/kernels/fp8_lora_swap.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------------------------------------------- the sweep
def smem_ok(c: dict) -> bool:
    bm, bn, bk, st = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c.get("num_stages", 4)
    xb = 2 if c["X_MODE"] == 0 else 1                     # X_MODE 0 streams bf16 x and stores its fp8 codes once more
    return st * (bn * bk + bm * bk * xb) + (bm * bk if c["X_MODE"] == 0 else 0) <= 200 * 1024


def base_cfg(M: int, xm: int, bn: int, sk: int, bk: int | None = None) -> dict:
    bm = 16 if M <= 16 else 32 if M <= 32 else 64
    bk = bk or (256 if M <= 32 else 128)
    c = {"X_MODE": xm, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk, "num_stages": 4, "num_warps": 4,
         "W_PREFETCH": 2}
    if bm * bn >= 4096:            # two warpgroups (4 warps spilled at BLOCK_M 64)
        c["num_warps"] = 8
    return c


def coarse(M: int, K: int, N: int, widths) -> list[dict]:
    """X_MODE x BLOCK_N x SPLIT_K at BLOCK_K 256 (128 above 32 tokens), 4 stages: split-K only while the projection has
    fewer than 2 tiles per SM, at most 8 CTAs per SM, each split at least two K-blocks."""
    bm = 16 if M <= 16 else 32 if M <= 32 else 64
    xms = (0, 1) if M <= 32 else (1,)
    out = []
    for xm, bn, sk in itertools.product(xms, (64, 128), (1, 2, 4, 8)):
        if any(w % bn for w in widths):
            continue
        c = base_cfg(M, xm, bn, sk)
        tiles = cdiv(M, bm) * cdiv(N, bn)
        if K // sk < 2 * c["BLOCK_K"] or tiles * sk > 8 * NSM or (sk > 1 and tiles >= 2 * NSM):
            continue
        if smem_ok(c):
            out.append(c)
    return out


def refine(best: dict, K: int) -> list[dict]:
    """One change at a time around the coarse sweep's best: the K-block, the pipeline depth, the L2 prefetch of the
    weight stream (none, 8 K-steps, the CTA's whole split), TMA-fed weights, B in L2 only (not registers), the shrink's
    split, X_MODE 2."""
    out = []

    def add(**kv):
        c = {**best, **kv}
        if c != best and smem_ok(c) and K // c["SPLIT_K"] >= 2 * c["BLOCK_K"] and c not in out:
            out.append(c)
    add(BLOCK_K=128 if best["BLOCK_K"] == 256 else 256)
    for st in (3, 6):
        add(num_stages=st)
    k_split = cdiv(cdiv(K, best["SPLIT_K"]), best["BLOCK_K"]) * best["BLOCK_K"]
    whole = 1 << ((k_split // best["BLOCK_K"]).bit_length() - 1)      # the CTA's whole K-split (power of two)
    for wp in sorted({0, 8, whole} - {best.get("W_PREFETCH", 2)}):
        add(W_PREFETCH=wp)
    add(W_TMA=True)
    add(B_PREFETCH=1)
    split1 = best.get("SPLIT_K1") or (4 if K <= 4096 else 8)
    for s1 in (split1 // 2, split1 * 2):
        add(SPLIT_K1=s1)
    add(X_MODE=2)
    return out


def nolora_grid(M: int, K: int, widths) -> list[dict]:
    """Tiles for our base GEMM alone (k2_nolora), both activation modes."""
    return [base_cfg(M, xm, bn, sk) for xm in (0, 1) for bn, sk in ((64, 1), (128, 1), (64, 2), (64, 4))
            if not any(w % bn for w in widths) and K // sk >= 512]


def neighbors(best: dict, K: int, M: int, N: int, widths) -> list[dict]:
    """The later stages' small sweep: the headline's best for the projection and batch bucket, its split-K halved and
    doubled, and the other activation mode."""
    out = [dict(best)]
    tiles = cdiv(M, best["BLOCK_M"]) * cdiv(N, best["BLOCK_N"])
    for sk in (best["SPLIT_K"] // 2, best["SPLIT_K"] * 2):
        if 1 <= sk <= 16 and K // sk >= 2 * best["BLOCK_K"] and tiles * sk <= 8 * NSM:
            out.append({**best, "SPLIT_K": sk})
    if M <= 32:
        out.append({**best, "X_MODE": 1 - min(best["X_MODE"], 1)})
    return [c for c in out if smem_ok(c)]


# ---------------------------------------------------------------------------------------------------------- set-up
class Bank:
    """Weights per projection (shared across ranks and formats), adapters and ops per (projection, r, L, format)."""

    def __init__(self, max_m: int):
        self.max_m = max_m
        self.w = {}
        self.ops = {}

    def weight(self, model, proj):
        K, slices = MODELS[model][proj]
        key = (K, slices)
        if key not in self.w:
            g = torch.Generator(device="cuda").manual_seed(K + sum(slices) + len(slices))
            N = sum(slices)
            w = torch.randn(N, K, device="cuda", generator=g) * 0.02
            s_w = (w.abs().max() / 448.0).reshape(1).float()
            self.w[key] = ((w / s_w).to(torch.float8_e4m3fn), s_w)
            del w
        return self.w[key]

    def op(self, model, proj, r, L, fmt, empty: bool = False):
        from lowbit_lora.kernels import fp8_lora_vllm as F
        K, slices = MODELS[model][proj]
        key = (K, slices, r, L, fmt)
        if key not in self.ops:
            w_q, s_w = self.weight(model, proj)
            N, S = sum(slices), len(slices)
            g = torch.Generator(device="cuda").manual_seed(1000 + r)
            if empty:
                A = torch.zeros(L, S * r, K, device="cuda", dtype=torch.bfloat16)
                B = torch.zeros(L, N, r, device="cuda", dtype=torch.bfloat16)
            else:
                A = (torch.randn(L, S * r, K, device="cuda", generator=g) / K ** 0.5).to(torch.bfloat16)
                B = (torch.randn(L, N, r, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
            a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))
            del A, B
            op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, self.max_m, lora_fp8=fmt == "fp8", lora_fp4=fmt == "fp4")
            self.ops[key] = (op, a_st, b_st)
        return self.ops[key]

    def drop_ops(self):
        self.ops.clear()
        torch.cuda.empty_cache()


def make_meta(L, max_m, M, n):
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
    meta = LoRAKernelMeta.make(L, max_m, device="cuda")
    tok = (torch.arange(M, device="cuda") % n).to(torch.int32)        # n distinct adapters, interleaved (decode)
    meta.prepare_tensors(tok)
    return meta, meta.meta_args(M, False)


class Env:
    """vLLM's ops, the tuned-config switch and the dual-stream resources (the bench and its compile workers)."""

    def __init__(self):
        from vllm import _custom_ops as ops
        from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
        from vllm.lora.ops.triton_ops import utils as U
        self.ops, self.lora_shrink, self.lora_expand, self.U = ops, lora_shrink, lora_expand, U
        self.aux = torch.cuda.Stream()
        self.ev = (torch.cuda.Event(), torch.cuda.Event())
        gname = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
        self.tuned_ok = (TUNED / f"{gname}_SHRINK.json").exists() and (TUNED / f"{gname}_EXPAND_TRUE.json").exists()

    def set_tuned(self, on: bool, dual: bool = False) -> None:
        if on and self.tuned_ok:
            os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(TUNED)
        else:
            os.environ.pop("VLLM_TUNED_CONFIG_FOLDER", None)
        if dual:
            os.environ["VLLM_LORA_ENABLE_DUAL_STREAM"] = "1"
        else:
            os.environ.pop("VLLM_LORA_ENABLE_DUAL_STREAM", None)
        self.U.load_lora_op_config.cache_clear()
        self.U.get_lora_op_configs.cache_clear()

    def fns(self, op, a_st, b_st, x, margs):
        """vendor (vLLM's quantize + CUTLASS), stock (+ vLLM's LoRA kernels), dual (vLLM's dual-stream LoRA)."""
        ops, ls, le = self.ops, self.lora_shrink, self.lora_expand
        w_t = op.w_q.t()
        M, S, r = x.shape[0], op.S, op.R
        hb = torch.empty(S * M * r, dtype=torch.float32, device="cuda").view(S, M, r)
        lo_buf = torch.empty(M, op.N, dtype=torch.bfloat16, device="cuda")

        def vendor(xx=None):
            xq, sx = ops.scaled_fp8_quant(x if xx is None else xx, use_per_token_if_dynamic=True)
            return ops.cutlass_scaled_mm(xq, w_t, sx, op.s_w, torch.bfloat16)

        def stock():
            y = vendor()
            ls(x, a_st, hb, *margs, 1.0)
            le(hb, b_st, y, *margs, offset_start=0, add_inputs=True)
            return y

        def dual():   # vLLM's _apply_async_impl through maybe_execute_in_parallel
            ev0, ev1 = self.ev
            ev0.record()
            y = vendor()
            with torch.cuda.stream(self.aux):
                ev0.wait()
                lo_buf.zero_()
                ls(x, a_st, hb, *margs, 1.0)
                le(hb, b_st, lo_buf, *margs, offset_start=0, add_inputs=False)
                ev1.record()
            ev1.wait()
            y.add_(lo_buf)
            return y
        return vendor, stock, dual


# ------------------------------------------------------------------------------------------------------ precompile
PDL_KINDS = {"pdl": {}, "nopdl": dict(pdl=False), "nolora": dict(lora=False), "noshrink": dict(shrink=False),
             "shrinkonly": dict(gemm=False), "coop": dict(coop=True)}


def _precompile_worker(items) -> tuple[int, list[str]]:
    """Compile every listed variant: items (model, proj, r, L, fmt, M, n, cfg, kind). kind "base": one call of each
    baseline (vLLM's kernels with default and tuned configurations, dual-stream, hybrid, swap); "xqd": the new path with
    the debug output; else a run_pdl variant (compile only: no launch)."""
    import torch as T
    from lowbit_lora import hopper_swap
    from lowbit_lora.hopper_pdl import run_pdl
    hopper_swap.load_cfg()
    bank = Bank(max(i[5] for i in items))
    env = Env()
    errs, done = [], 0
    for model, proj, r, L, fmt, M, n, cfg, kind in items:
        try:
            op, a_st, b_st = bank.op(model, proj, r, L, fmt, empty=True)
            _, margs = make_meta(L, bank.max_m, M, n)
            x = T.zeros(M, op.K, dtype=T.bfloat16, device="cuda")
            if kind == "base":
                vendor, stock, dual = env.fns(op, a_st, b_st, x, margs)
                op.base_fn = vendor
                for tuned in (False, True):
                    env.set_tuned(tuned)
                    stock()
                env.set_tuned(True, dual=True)
                dual()
                env.set_tuned(False)
                op._run(x, margs, hybrid=True)
                hopper_swap.run_swap(op, x, margs)
                op.base_fn = None
                T.cuda.synchronize()
            elif kind == "xqd":
                run_pdl(op, x, margs, cfg=cfg, compile_only=True,
                        xqd=T.zeros(M, op.K, dtype=T.float8_e4m3fn, device="cuda"))
            else:
                run_pdl(op, x, margs, cfg=cfg, compile_only=True, **PDL_KINDS[kind])
            done += 1
        except Exception as e:   # noqa: BLE001
            errs.append(f"{proj} r{r} {fmt} M{M} {kind} {cfg}: {type(e).__name__}: {str(e)[:200]}")
    return done, errs


def precompile(items: list, workers: int, label: str) -> None:
    """Items grouped so that one worker compiles every item sharing a kernel family (the same configuration, kind,
    rank, slots and format; M 1 apart, as X_MODE 0's prologue specializes on it), groups balanced over the workers."""
    if not items or workers <= 0:
        return
    import multiprocessing as mp
    t0 = time.time()
    groups: dict = {}
    for it in items:
        model, proj, r, L, fmt, M, n, cfg, kind = it
        key = (model, proj, r, L, fmt, M > 16) if kind == "base" else (r, L, fmt, json.dumps(cfg, sort_keys=True),
                                                                        kind, M == 1)
        groups.setdefault(json.dumps(key, default=str), []).append(it)
    chunks = [[] for _ in range(workers)]
    for g in sorted(groups.values(), key=len, reverse=True):
        min(chunks, key=len).extend(g)
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_precompile_worker, [c for c in chunks if c])
    except Exception as e:   # noqa: BLE001 -- a crashed worker: the timings compile on demand instead
        print(f"[precompile {label}] workers failed ({type(e).__name__}: {str(e)[:200]}); compiling on demand",
              flush=True)
        return
    errs = [e for _, es in res for e in es]
    print(f"[precompile {label}] {sum(d for d, _ in res)}/{len(items)} calls, {len(groups)} kernel families, "
          f"{time.time() - t0:.0f} s{f'; {len(errs)} failed, first: {errs[0]}' if errs else ''}", flush=True)


# ----------------------------------------------------------------------------------------------------------- cases
class Runner:
    """hopper_pdl with the start-up probes' verdicts applied: if PDL launches cannot be captured in a CUDA graph, every
    variant runs the same kernels without the attribute; if cooperative launches cannot, the merged kernel launches
    plainly (both recorded in every row: pdl_capture, coop_capture)."""

    def __init__(self):
        from lowbit_lora import hopper_pdl as H
        self.H = H
        self.pdl_ok = True
        self.coop_ok = True

    def run_pdl(self, *a, **kw):
        if not self.pdl_ok:
            kw["pdl"] = False
        if kw.get("coop") and not self.coop_ok:
            kw["coop_launch"] = False
        return self.H.run_pdl(*a, **kw)

    def resolve_cfg(self, *a, **kw):
        return self.H.resolve_cfg(*a, **kw)


def probe_capture(ctx) -> dict:
    """Capture the new path (PDL launch) and the merged kernel (cooperative launch) once in a CUDA graph."""
    from lowbit_lora import hopper_pdl as H
    from lowbit_lora.kernels.fp8_lora_pdl import work
    op, _, _ = ctx.bank.op("qwen3_8b", "o", 16, 8, "bf16")
    _, margs = make_meta(8, ctx.bank.max_m, 1, 1)
    x = torch.randn(1, op.K, device="cuda").to(torch.bfloat16)
    res = {}
    for name, kw in (("pdl_capture", {}), ("coop_capture", dict(coop=True, coop_launch=True))):
        try:
            ctx.fast.graph_time(lambda: H.run_pdl(op, x, margs, cfg={"SPLIT_K": 2}, **kw))
            res[name] = True
        except Exception as e:   # noqa: BLE001
            res[name] = f"{type(e).__name__}: {str(e)[:200]}"
            try:
                torch.cuda.synchronize()
                work(x.device).reset()
            except Exception:   # noqa: BLE001
                pass
    ctx.runner.pdl_ok = res["pdl_capture"] is True
    ctx.runner.coop_ok = res["coop_capture"] is True
    return res


def rel(y, ref) -> float:
    return float((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))


class Case:
    """One (stage, model, projection, r, M, n, format) row: set-up and baselines, then sweeps, then the final variants."""

    def __init__(self, ctx, stage, model, proj, r, M, n, fmt, level):
        self.ctx, self.stage, self.model, self.proj, self.r, self.M, self.n, self.fmt, self.level = (
            ctx, stage, model, proj, r, M, n, fmt, level)
        self.K, self.slices = MODELS[model][proj]
        self.N, self.S = sum(self.slices), len(self.slices)
        self.L = 32 if n > 8 else 8
        self.full = stage in FULL
        self.key = (self.K, self.N, 16 if M <= 16 else M)
        self.rec = {**ctx.info, "stage": stage, "model": model, "proj": proj, "K": self.K, "N": self.N, "S": self.S,
                    "r": r, "M": M, "n": n, "L": self.L, "fmt": fmt}
        self.errs = {}
        self.tried = []
        self.d = None
        self.best = (float("inf"), None)
        self.H = ctx.runner

    def item(self, cfg, kind):
        return (self.model, self.proj, self.r, self.L, self.fmt, self.M, self.n, cfg, kind)

    def timed(self, name, fn, timer=None):
        try:
            self.rec["t_" + name] = round((timer or self.ctx.acc).graph_time(fn), 2)
        except Exception as e:   # noqa: BLE001
            self.rec["fail_" + name] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [{self.proj} M{self.M} {name}] {self.rec['fail_' + name]}", flush=True)

    def check(self, name, y, ref=None):
        ref = self.ref_fmt if ref is None else ref
        e = rel(y, ref)
        self.errs[name] = float(f"{e:.3g}")
        if self.fmt == "bf16" and ref is self.y_stock:   # the LoRA term alone against stock's (descriptive)
            self.errs[name + "_lora"] = float(f"{rel(y.float() - self.y_base, self.lora_part):.3g}")
        return e <= 2e-2

    def setup(self):
        """Inputs, references and the baselines' timings."""
        ctx, H = self.ctx, self.H
        from lowbit_lora.hopper_swap import run_swap
        self.op, a_st, b_st = ctx.bank.op(self.model, self.proj, self.r, self.L, self.fmt)
        op = self.op
        _, self.margs = make_meta(self.L, ctx.bank.max_m, self.M, self.n)
        g = torch.Generator(device="cuda").manual_seed(7 * self.M + self.n)
        self.x = torch.randn(self.M, self.K, device="cuda", generator=g).to(torch.bfloat16)
        x, margs = self.x, self.margs
        self.vendor, stock, dual = ctx.fns(op, a_st, b_st, x, margs)
        self.a_st, self.b_st = a_st, b_st
        op.base_fn = self.vendor
        ctx.set_tuned(False)
        self.y_stock = stock().float().clone()
        self.y_base = self.vendor().float()
        self.lora_part = self.y_stock - self.y_base
        self.timed("ref", self.vendor)
        self.timed("stock", stock)
        if ctx.tuned_ok:
            try:
                ctx.set_tuned(True)
                self.check("stock_tuned", stock(), self.y_stock)
                self.timed("stock_tuned", stock)
                ctx.set_tuned(True, dual=True)
                self.check("stock_dual", dual(), self.y_stock)
                self.timed("stock_dual", dual)
            except Exception as e:   # noqa: BLE001
                self.rec["fail_stock_tuned"] = f"{type(e).__name__}: {str(e)[:300]}"
            ctx.set_tuned(False)
        y_h = op._run(x, margs, hybrid=True).float().clone()
        self.ref_fmt = self.y_stock if self.fmt == "bf16" else y_h    # stored adapters: the hybrid path on them
        if self.fmt == "bf16":
            self.check("hybrid", y_h)
        else:
            self.errs["hybrid_vs_stock_bf16"] = float(f"{rel(y_h, self.y_stock):.3g}")
        self.timed("hybrid", lambda: op._run(x, margs, hybrid=True))
        try:
            self.check("swap", run_swap(op, x, margs))
            self.timed("swap", lambda: run_swap(op, x, margs))
        except Exception as e:   # noqa: BLE001
            self.rec["fail_swap"] = f"{type(e).__name__}: {str(e)[:300]}"
        self.d = H.resolve_cfg(op, self.M)
        self.rec["pdl_default"] = self.d
        try:
            t, e = self.pdl_time(self.d, ctx.acc)
            self.rec["t_pdl_default"] = round(t, 2) if t is not None else None
            self.errs["pdl_default"] = float(f"{e:.3g}")
            if t is not None:
                self.best = (t, self.d)
        except Exception as ex:   # noqa: BLE001
            self.rec["fail_pdl_default"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            traceback.print_exc()
        op.base_fn = None

    def pdl_time(self, c, timer=None, **kw):
        H, op, x, margs = self.H, self.op, self.x, self.margs
        e = rel(H.run_pdl(op, x, margs, cfg=c, **kw), self.ref_fmt)
        if e > 2e-2:
            return None, e
        return (timer or self.ctx.fast).graph_time(lambda: H.run_pdl(op, x, margs, cfg=c, **kw)), e

    def candidates(self, phase: str) -> list[dict]:
        if phase == "coarse":
            if self.level in ("full", "large"):
                return coarse(self.M, self.K, self.N, self.slices)
            seed = self.ctx.best.get(self.key) or self.ctx.best.get((self.K, self.N, 16)) or self.d
            return neighbors(seed, self.K, self.M, self.N, self.slices)
        if phase == "refine" and self.level == "full" and self.best[1] is not None:
            return refine(self.best[1], self.K)
        return []

    def sweep(self, cands):
        for c in cands:
            if self.ctx.over():
                break
            if any(t.get("cfg") == c for t in self.tried) or c == self.d:
                continue
            try:
                t, e = self.pdl_time(c)
            except Exception as ex:   # noqa: BLE001
                self.tried.append({"cfg": c, "fail": f"{type(ex).__name__}: {str(ex)[:160]}"})
                continue
            self.tried.append({"cfg": c, "t": round(t, 2) if t is not None else None, "err": float(f"{e:.3g}")})
            if t is not None and t < self.best[0]:
                self.best = (t, c)

    def extra_items(self) -> list:
        """The final variants at the current best (precompiled before the finish phase)."""
        bc = self.best[1] or self.d
        kinds = ["pdl", "nopdl"] + (["noshrink", "shrinkonly", "coop"] if self.full else
                                    ["coop"] if self.stage == "largeM" else ["noshrink", "shrinkonly"] if self.stage in MODEL_VARS else [])
        it = [self.item(bc, k) for k in kinds]
        if self.full or self.stage == "largeM" or self.stage in MODEL_VARS:
            it += [self.item({**bc, "X_MODE": xm}, "nolora") for xm in (0, 1)]
        return it

    def floor(self, bc: dict) -> None:
        """SHRINK-FLOOR: what any serialized shrink would cost. t_empty: an empty kernel in a graph (the launch floor);
        t_shrink_cublas: the adapter's down-projection as one cuBLAS bf16 GEMM, x [M, K] against every slice's A
        concatenated [S*r, K] (the fastest library path for these shapes); t_serial_cublas: that GEMM, then our GEMM with
        LoRA reading h (no shrink kernel, so nothing overlaps): a vendor-fast shrink serialized before the GEMM;
        k2_ctas: the GEMM's thread blocks (tiles x split-K) against the GPU's SMs."""
        H, op, x, margs, rec = self.H, self.op, self.x, self.margs, self.rec
        import triton
        import triton.language as tl

        @triton.jit
        def _empty(p):
            pass
        buf = torch.empty(1, device="cuda")
        self.timed("empty", lambda: _empty[(1,)](buf))
        a_cat = torch.cat([a[0, 0] for a in self.a_st], 0).contiguous()     # [S*r, K], slot 0 (the one adapter)
        hb = torch.empty(self.M, a_cat.shape[0], dtype=torch.bfloat16, device="cuda")
        self.timed("shrink_cublas", lambda: torch.mm(x, a_cat.t(), out=hb))
        H.run_pdl(op, x, margs, cfg=bc)                                       # h for the no-shrink GEMM

        def serial():
            torch.mm(x, a_cat.t(), out=hb)
            return H.run_pdl(op, x, margs, cfg=bc, shrink=False)
        self.timed("serial_cublas", serial)
        if os.environ.get("LOWBIT_FLOOR_EMPTY", "0") not in ("0", "", "false", "False"):
            # SHRINK-FLOOR amendment -FLOOR2 (mock review R1 v37, change 2; opt-in, so earlier runs are unchanged): the floor of ANY
            # serialized shrink, an empty kernel launched before the same GEMM reading h (a shrink that does no work)
            def serial_empty():
                _empty[(1,)](buf)
                return H.run_pdl(op, x, margs, cfg=bc, shrink=False)
            self.timed("serial_empty", serial_empty)
        if os.environ.get("LOWBIT_FLOOR_GEMV", "0") not in ("0", "", "false", "False") and self.M == 1:
            # SHRINK-FLOOR amendment -GEMV (2026-10-25, post hoc; mock review R1 v39 M3; opt-in, earlier runs unchanged): a minimal real
            # serialized shrink, cuBLAS's matrix-vector product (torch.mv) at one token, before the same GEMM reading h
            hv = hb[0]
            self.timed("shrink_gemv", lambda: torch.mv(a_cat, x[0], out=hv))

            def serial_gemv():
                torch.mv(a_cat, x[0], out=hv)
                return H.run_pdl(op, x, margs, cfg=bc, shrink=False)
            self.timed("serial_gemv", serial_gemv)
        g = H.H._geometry(op, x, H.resolve_cfg(op, self.M, bc), True)    # H: the Runner; H.H: lowbit_lora.hopper_pdl
        rec["k2_ctas"] = int(g["n_tiles"] * g["sk"])
        rec["shrink_ctas"] = int(g["gx"])
        rec["n_sm"] = int(torch.cuda.get_device_properties(0).multi_processor_count)

    def finish(self, out_path: Path):
        ctx, H, op, x, margs = self.ctx, self.H, self.op, self.x, self.margs
        rec = self.rec
        bc = self.best[1] or self.d
        rec["pdl_best"] = bc
        rec["pdl_sweep"] = self.tried
        late = ctx.over()               # past the budget: the best configuration's row only, no optional extras
        rec["late"] = late
        if self.level == "full" or self.key not in ctx.best:
            ctx.best[self.key] = bc
        try:
            y_p = H.run_pdl(op, x, margs, cfg=bc).float().clone()
            self.check("pdl", y_p)
            reps = [H.run_pdl(op, x, margs, cfg=bc).float() for _ in range(3)]
            rec["pdl_deterministic"] = all(torch.equal(y_p, yy) for yy in reps)
            self.timed("pdl", lambda: H.run_pdl(op, x, margs, cfg=bc))
            self.check("pdl_nopdl", H.run_pdl(op, x, margs, cfg=bc, pdl=False))
            self.timed("pdl_nopdl", lambda: H.run_pdl(op, x, margs, cfg=bc, pdl=False))
            if (self.full or self.stage in MODEL_VARS) and not late:
                H.run_pdl(op, x, margs, cfg=bc)                               # h for the no-shrink variant
                self.check("pdl_noshrink", H.run_pdl(op, x, margs, cfg=bc, shrink=False))
                self.timed("pdl_noshrink", lambda: H.run_pdl(op, x, margs, cfg=bc, shrink=False))
                self.timed("shrink_only", lambda: H.run_pdl(op, x, margs, cfg=bc, gemm=False))
            if self.stage == "floor" and not late:
                self.floor(bc)
        except Exception as ex:   # noqa: BLE001
            rec["fail_pdl"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            traceback.print_exc()
        # vLLM's codes, byte for byte: the GEMM's own quantization (X_MODE 0, its debug output) and the quantization
        # programs' (X_MODE 1), at the default tiles (precompiled)
        try:
            xq_v, sx_v = ctx.ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            xqd = torch.zeros(self.M, self.K, dtype=torch.float8_e4m3fn, device="cuda")
            H.run_pdl(op, x, margs, cfg={**self.d, "X_MODE": 0}, xqd=xqd)
            rec["xq_mode0_codes_differing"] = int((xqd.view(torch.uint8) != xq_v.view(torch.uint8)).sum())
            H.run_pdl(op, x, margs, cfg={**self.d, "X_MODE": 1})
            xq1 = op.scratch.xq[: self.M * self.K].view(self.M, self.K)
            rec["xq_mode1_codes_differing"] = int((xq1.view(torch.uint8) != xq_v.view(torch.uint8)).sum())
            rec["xq_mode1_scales_equal"] = bool(torch.equal(op.scratch.sx[: self.M], sx_v.view(-1)))
        except Exception as ex:   # noqa: BLE001
            rec["fail_xq_check"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        # our base GEMM without LoRA against CUTLASS alone
        if (self.full or self.stage == "largeM" or self.stage in MODEL_VARS) and not late:
            nl = []
            grid = [{**bc, "X_MODE": 0}, {**bc, "X_MODE": 1}] + (nolora_grid(self.M, self.K, self.slices)
                                                                  if self.full else [])
            for c in grid:
                try:
                    e0 = rel(H.run_pdl(op, x, margs, cfg=c, lora=False), self.y_base)
                    if e0 > 2e-2:
                        nl.append({"cfg": c, "err": e0})
                        continue
                    t0 = ctx.fast.graph_time(lambda: H.run_pdl(op, x, margs, cfg=c, lora=False))
                    nl.append({"cfg": c, "t": round(t0, 2), "err": float(f"{e0:.3g}")})
                except Exception as ex:   # noqa: BLE001
                    nl.append({"cfg": c, "fail": f"{type(ex).__name__}: {str(ex)[:160]}"})
            okn = [v for v in nl if v.get("t") is not None]
            if okn:
                b0 = min(okn, key=lambda v: v["t"])
                self.timed("k2_nolora", lambda: H.run_pdl(op, x, margs, cfg=b0["cfg"], lora=False))
                rec["k2_nolora_cfg"] = b0["cfg"]
                self.errs["k2_nolora_vs_ref"] = b0["err"]
            rec["k2_nolora_sweep"] = nl
        # the merged fallback: cooperative launch where the grid fits, else a plain launch (shrink programs first)
        if (self.full or self.stage == "largeM") and not late:
            from lowbit_lora.kernels.fp8_lora_pdl import work
            for launch in (True, False):
                try:
                    self.check("coop", H.run_pdl(op, x, margs, cfg=bc, coop=True, coop_launch=launch))
                    self.timed("coop", lambda: H.run_pdl(op, x, margs, cfg=bc, coop=True, coop_launch=launch))
                    rec["coop_cooperative_launch"] = launch
                    rec.pop("fail_coop", None)
                    break
                except Exception as ex:   # noqa: BLE001
                    rec["fail_coop"] = f"{type(ex).__name__}: {str(ex)[:300]}"
                    torch.cuda.synchronize()
                    work(x.device).reset()
        rec["err"] = self.errs
        rec["checks_ok"] = all(v <= 2e-2 for k, v in self.errs.items()
                               if not k.endswith("_lora") and k != "hybrid_vs_stock_bf16")
        # the new path's own checks: its outputs (every variant) within 2e-2, its codes byte-identical to vLLM's,
        # repeated launches bitwise identical
        rec["pdl_checks_ok"] = (all(v <= 2e-2 for k, v in self.errs.items() if not k.endswith("_lora") and
                                    k.split("_")[0] in ("pdl", "coop", "k2"))
                                and rec.get("xq_mode0_codes_differing") == 0 and rec.get("xq_mode1_codes_differing") == 0
                                and bool(rec.get("xq_mode1_scales_equal")) and bool(rec.get("pdl_deterministic")))
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["elapsed_s"] = round(ctx.elapsed())
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        show = ("ref", "stock", "stock_tuned", "stock_dual", "hybrid", "swap", "pdl", "pdl_nopdl", "pdl_noshrink",
                "k2_nolora", "coop")
        print(f"{self.stage:8s} {self.model} {self.proj:7s} r{self.r:<3d} M{self.M:<3d} n{self.n:<2d} {self.fmt:4s} | "
              + " ".join(f"{k} {rec.get('t_' + k, '-')}" for k in show)
              + f" | best {json.dumps({k: bc[k] for k in ('X_MODE', 'BLOCK_N', 'BLOCK_K', 'SPLIT_K') if k in bc})}"
              f" ok={rec['checks_ok']} pdl_ok={rec['pdl_checks_ok']} [{ctx.elapsed():.0f}s]", flush=True)
        if self.stage in ("headline", "qwen35", "qwen72b", "qwen72b_tp2") and self.n == 1 and self.r == 64 and self.fmt == "bf16":
            ctx.layer.setdefault((self.model, self.M), {})[self.proj] = self


class Ctx(Env):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.fast = Timer(warm=3, reps=15)
        self.acc = Timer(warm=5, reps=40)
        self.t0 = time.time()
        self.budget = a.budget_min * 60
        self.bank = Bank(128)
        self.info = {**gpu_info(), "job": os.environ.get("SLURM_JOB_ID", "") or os.environ.get("LB_JOB", ""),
                     "kernel_sha": os.environ.get("LOWBIT_SHA", "") or kernel_sha(), "torch": torch.__version__,
                     "tuned_configs_found": self.tuned_ok}
        self.runner = Runner()
        self.best = {}     # (K, N, M bucket) -> the headline's best pdl config (seeds the later stages' sweeps)
        self.layer = {}    # (model, M) -> {proj: Case} for the layer graphs

    def elapsed(self) -> float:
        return time.time() - self.t0

    def over(self) -> bool:
        return self.elapsed() > self.budget


def layer_graphs(ctx: Ctx, model: str, M: int, projs: list[str], out_path: Path) -> None:
    """One layer's projections back to back in one graph (L2 flushed once per replay): ref, stock, stock_tuned, hybrid,
    swap and pdl at their best configurations."""
    from lowbit_lora.hopper_swap import run_swap
    have = ctx.layer.get((model, M), {})
    if not all(p in have for p in projs):
        return
    fns = {p: ctx.fns(have[p].op, have[p].a_st, have[p].b_st, have[p].x, have[p].margs) for p in projs}

    def seq(kind):
        def f():
            ys = []
            for p in projs:
                c = have[p]
                vendor, stock, _ = fns[p]
                if kind == "ref":
                    ys.append(vendor())
                elif kind in ("stock", "stock_tuned"):
                    ys.append(stock())
                elif kind == "hybrid":
                    c.op.base_fn = vendor
                    ys.append(c.op._run(c.x, c.margs, hybrid=True))
                elif kind == "swap":
                    ys.append(run_swap(c.op, c.x, c.margs))
                else:
                    ys.append(c.H.run_pdl(c.op, c.x, c.margs, cfg=c.best[1] or c.d))
            return ys
        return f
    rec = {**ctx.info, "stage": "layer", "model": model, "M": M, "n": 1, "r": 64, "projs": projs,
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for kind in ("ref", "stock", "stock_tuned", "hybrid", "swap", "pdl"):
        ctx.set_tuned(kind == "stock_tuned")
        try:
            rec["t_layer_" + kind] = round(ctx.acc.graph_time(seq(kind)), 2)
        except Exception as e:   # noqa: BLE001
            rec["fail_layer_" + kind] = f"{type(e).__name__}: {str(e)[:300]}"
    ctx.set_tuned(False)
    for p in projs:
        have[p].op.base_fn = None
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"layer    {model} M{M:<3d} {'+'.join(projs)}: " + " ".join(
        f"{k[8:]} {rec[k]}" for k in rec if k.startswith("t_layer_")), flush=True)


def plan(stages: set, smoke: bool) -> list[tuple]:
    """(stage, model, proj, r, M, n, fmt, sweep level) in priority order."""
    q8 = list(MODELS["qwen3_8b"])
    if smoke:
        return [("smoke", "qwen3_8b", "o", 64, 1, 1, "bf16", "small"),
                ("smoke", "qwen3_8b", "down", 64, 16, 8, "bf16", "small")]
    out = []
    if "headline" in stages:
        out += [("headline", "qwen3_8b", p, 64, M, 1, "bf16", "full") for M in (1, 4, 8, 16) for p in q8]
    if "qwen35" in stages:
        out += [("qwen35", "qwen35_9b", p, 64, M, 1, "bf16", "full") for M in (1, 4, 8, 16) for p in ("qkv", "qkvz")]
    if "largeM" in stages:
        out += [("largeM", "qwen3_8b", p, 64, M, 1, "bf16", "large") for M in (32, 64, 128) for p in q8]
    if "adapters" in stages:
        out += [("adapters", "qwen3_8b", p, 64, M, n, "bf16", "small") for n in (8, 32) for M in (8, 16, 32, 64)
                if n <= M for p in q8]
    if "ranks" in stages:
        out += [("ranks", "qwen3_8b", p, r, M, 1, "bf16", "small") for r in (16, 128) for M in (1, 16) for p in q8]
    if "tp" in stages:                 # amendment -TP: one rank-64 adapter at one token on each tensor-parallel shard
        out += [("tp", f"qwen3_8b_tp{tp}", p, 64, 1, 1, "bf16", "small") for tp in (2, 4, 8) for p in q8]
    if "formats" in stages:
        out += [("formats", "qwen3_8b", p, 64, M, n, fmt, "small") for fmt in ("fp8", "fp4") for M in (1, 16)
                for n in (1, 8) if n <= M for p in q8]
    if "llama" in stages:              # E2E-LLAMA-H100: the headline's sweep on Llama-3.1-8B, and M 32 (largeM's grid)
        ql = list(MODELS["llama31_8b"])
        out += [("llama", "llama31_8b", p, 64, M, 1, "bf16", "full") for M in (1, 4, 8, 16) for p in ql]
        out += [("llama", "llama31_8b", p, 64, 32, 1, "bf16", "large") for p in ql]
    for st72, mdl in (("qwen72b", "qwen25_72b"), ("qwen72b_tp2", "qwen25_72b_tp2")):   # E2E-72B-H200 -TILES, E2E-TP2-TUNED
        if st72 in stages:
            q72 = list(MODELS[mdl])
            out += [(st72, mdl, p, 64, M, 1, "bf16", "full") for M in (1, 4, 8, 16) for p in q72]
            out += [(st72, mdl, p, 64, 32, 1, "bf16", "large") for p in q72]
    if "attr32" in stages:             # THRU-ATTR-H100: 32 tenants at the throughput workload's batch sizes (rank 64); the rows'
        # ref / stock / stock_tuned / stock_dual / hybrid times split each path's LoRA cost from the base GEMM's
        out += [("attr32", "qwen35_9b", p, 64, M, 32, "bf16", "small") for M in (64, 128, 256)
                for p in ("qkv", "qkvz", "o", "gate_up", "down")]
    if "floor" in stages:              # SHRINK-FLOOR: the decode path against a fast serialized shrink, ranks 16/64/128, one token
        lvl = os.environ.get("LOWBIT_FLOOR_LEVEL", "small")   # -FULL (2026-10-25, post hoc): "full", the headline's sweep at M = 1
        out += [("floor", "qwen3_8b", p, r, 1, 1, "bf16", lvl) for r in (16, 64, 128) for p in q8]
    if "exposure" in stages:           # KERNEL-EXPOSURE: small models at ranks 64 and 128 (full sweep), Qwen3-8B's TP-8 shards at rank 128
        out += [("exposure", m, p, r, 1, 1, "bf16", "full") for m in ("qwen3_0p6b", "qwen3_1p7b") for r in (64, 128) for p in q8]
        out += [("exposure", "qwen3_8b_tp8", p, 128, 1, 1, "bf16", "small") for p in q8]
    return out


def run_group(ctx: Ctx, specs: list[tuple], out_path: Path, workers: int) -> None:
    """One group of cases through the phases: baselines (after compiling them and the coarse sweep), the coarse sweep,
    the refinement (compiled first), the final variants (compiled first), rows written."""
    from lowbit_lora.hopper_pdl import resolve_cfg
    if ctx.over():
        print(f"[budget] skipping {len(specs)} cases of {sorted({s[0] for s in specs})}", flush=True)
        return
    items = []
    for s in specs:
        stage, model, proj, r, M, n, fmt, level = s
        K, slices = MODELS[model][proj]
        L = 32 if n > 8 else 8

        class _O:   # what resolve_cfg reads
            pass
        o = _O()
        o.N, o.K, o.S, o.R, o.widths, o.lora_fp4, o.lora_fp8 = sum(slices), K, len(slices), r, slices, fmt == "fp4", \
            fmt == "fp8"
        d = resolve_cfg(o, M)
        it = lambda c, k: (model, proj, r, L, fmt, M, n, c, k)   # noqa: E731
        items += [it(None, "base"), it(d, "pdl"), it({**d, "X_MODE": 0}, "xqd"), it({**d, "X_MODE": 1}, "pdl")]
        if level in ("full", "large"):
            items += [it(c, "pdl") for c in coarse(M, K, sum(slices), slices)]
        else:
            seed = ctx.best.get((K, sum(slices), 16 if M <= 16 else M)) or ctx.best.get((K, sum(slices), 16)) or d
            items += [it(c, "pdl") for c in neighbors(seed, K, M, sum(slices), slices)]
        if stage in FULL:
            items += [it(c, "nolora") for c in nolora_grid(M, K, slices)]
    precompile(items, workers, f"{specs[0][0]}: baselines + coarse")
    cases = []
    for s in specs:
        if ctx.over():
            break
        c = Case(ctx, *s)
        try:
            c.setup()
            c.sweep(c.candidates("coarse"))
            cases.append(c)
        except Exception:   # noqa: BLE001
            traceback.print_exc()
    ref_items = [c.item(cfg, "pdl") for c in cases for cfg in c.candidates("refine")]
    if not ctx.over():
        precompile(ref_items, workers, f"{specs[0][0]}: refinement")
    for c in cases:
        c.sweep(c.candidates("refine"))
    if not ctx.over():
        precompile([i for c in cases for i in c.extra_items()], workers, f"{specs[0][0]}: final variants")
    for c in cases:
        try:
            c.finish(out_path)
        except Exception:   # noqa: BLE001
            traceback.print_exc()


def summarize(out_path: Path, job: str) -> None:
    rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("job") == job]
    keys = ("ref", "stock", "stock_tuned", "stock_dual", "hybrid", "swap", "pdl", "pdl_nopdl", "k2_nolora", "coop")
    print("\n== one layer's projections summed (us), r 64, one adapter")
    for model, projs in (("qwen3_8b", ("qkv", "o", "gate_up", "down")), ("qwen35_9b", ("qkv", "o", "gate_up", "down")),
                         ("qwen35_9b", ("qkvz", "o", "gate_up", "down"))):
        for M in (1, 4, 8, 16, 32, 64, 128):
            rs = {}
            for r in rows:
                if r.get("n") == 1 and r.get("r") == 64 and r.get("fmt") == "bf16" and r.get("M") == M and \
                        r.get("stage") in ("headline", "qwen35", "largeM"):
                    if r["model"] == model or (model == "qwen35_9b" and r["proj"] in ("o", "gate_up", "down")):
                        rs[r["proj"]] = r
            if not all(p in rs for p in projs):
                continue
            tot = {k: sum(rs[p].get("t_" + k) or float("nan") for p in projs) for k in keys}
            print(f"  {model} {'+'.join(projs)} M {M:3d}: " + " ".join(f"{k} {v:.1f}" for k, v in tot.items())
                  + f" | pdl/ref {tot['pdl'] / tot['ref']:.3f} pdl/stock_tuned {tot['pdl'] / tot['stock_tuned']:.3f}"
                  f" k2_nolora/ref {tot['k2_nolora'] / tot['ref']:.3f}")
    lay = [r for r in rows if r.get("stage") == "layer"]
    if lay:
        print("== one layer in one graph (us)")
        for r in lay:
            print(f"  {r['model']} M {r['M']:3d}: " + " ".join(f"{k[8:]} {r[k]}" for k in r if k.startswith("t_layer_")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=28.0, help="start no case after this many minutes")
    ap.add_argument("--workers", type=int, default=8, help="compile processes before each phase's timings (0: none)")
    ap.add_argument("--stages", default="headline,qwen35,largeM,adapters,ranks,formats")   # also: tp (amendment -TP), exposure
    ap.add_argument("--smoke", action="store_true", help="two cases with a small sweep (a few minutes)")
    a = ap.parse_args()
    torch.manual_seed(0)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from lowbit_lora import hopper_swap
    n_swap = hopper_swap.load_cfg()
    ctx = Ctx(a)
    print(f"[pdl_k2_bench] {ctx.info} swap tiles: {n_swap} entries; budget {a.budget_min} min", flush=True)
    if not ctx.tuned_ok:
        print(f"[pdl_k2_bench] WARNING: no tuned vLLM LoRA configs for {torch.cuda.get_device_name()} in {TUNED}: "
              "stock_tuned and stock_dual not timed", flush=True)
    probe = probe_capture(ctx)
    ctx.info.update(probe)
    print(f"[pdl_k2_bench] capture probes: {probe}", flush=True)
    if probe["pdl_capture"] is not True:
        print("[pdl_k2_bench] WARNING: PDL launches could not be captured: every variant runs without the attribute",
              flush=True)
    specs = plan(set(a.stages.split(",")), a.smoke)
    groups = []
    for s in specs:                                        # consecutive cases of one stage form a group
        if groups and groups[-1][0][0] == s[0]:
            groups[-1].append(s)
        else:
            groups.append([s])
    for gi, g in enumerate(groups):
        if g[0][0] not in ("headline", "qwen35", "smoke"):
            ctx.bank.drop_ops()
        run_group(ctx, g, out_path, a.workers)
        if g[0][0] == "headline":
            for M in (1, 4, 8, 16):
                layer_graphs(ctx, "qwen3_8b", M, ["qkv", "o", "gate_up", "down"], out_path)
        if g[0][0] == "llama":
            for M in (1, 4, 8, 16):
                layer_graphs(ctx, "llama31_8b", M, ["qkv", "o", "gate_up", "down"], out_path)
        if g[0][0] in ("qwen72b", "qwen72b_tp2"):         # E2E-72B-H200 -TILES, E2E-TP2-TUNED: one layer in one graph
            for M in (1, 4, 8, 16):
                layer_graphs(ctx, "qwen25_72b" + ("_tp2" if g[0][0] == "qwen72b_tp2" else ""), M, ["qkv", "o", "gate_up", "down"],
                             out_path)
            ctx.layer.clear()
        if g[0][0] == "qwen35":
            for M in (1, 4, 8, 16):
                for projs in (["qkv", "o", "gate_up", "down"], ["qkvz", "o", "gate_up", "down"]):
                    have = {**{p: c for p, c in ctx.layer.get(("qwen3_8b", M), {}).items() if p != "qkv"},
                            **ctx.layer.get(("qwen35_9b", M), {})}
                    ctx.layer[("qwen35_9b_full", M)] = have
                    layer_graphs(ctx, "qwen35_9b_full", M, projs, out_path)
            ctx.layer.clear()                              # the cases (and their ops) can go now
    summarize(out_path, ctx.info["job"])
    print(f"[pdl_k2_bench] done in {ctx.elapsed() / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
