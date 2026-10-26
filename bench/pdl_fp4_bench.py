"""The NVFP4 decode path with the adapters off the critical path (kernels/nvfp4_lora_pdl.py via
lowbit_lora/blackwell_pdl.run_pdl4) on a Blackwell sm_120 slice (RTX PRO 6000 MIG 2g.48gb) against vLLM's and the plugin's
NVFP4 paths, per Qwen3-8B projection, rank r, batch size M, adapter count n and adapter storage format. CUDA-graph timings
(bench/kernel_bench.Timer, L2 flushed before every replay, median), microseconds, of
  ref           vLLM's scaled_fp4_quant + CUTLASS's FP4 GEMM (cutlass_scaled_fp4_mm), no LoRA ("CUTLASS alone")
  stock         ref, then vLLM's lora_shrink / lora_expand with their default configurations
  stock_tuned   the same with vLLM's configurations tuned for the slice (results/vllm_lora_tuned/Qwen3-8B-NVFP4_bw, as
                STOCK-TUNED-BW, jobs/e2e_stocktuned_bw.sbatch)
  stock_dual    vLLM's opt-in dual-stream LoRA (VLLM_LORA_ENABLE_DUAL_STREAM=1: the LoRA on an aux stream beside the
                base GEMM, PDL in its kernels on sm_90+; nothing gates it by architecture), tuned (vLLM's defaults if the
                slice's files are missing: stock_dual_tuned)
  fused         the plugin's NVFP4 fused path today: K1 (x's codes + every shrink), then K2 (_gemm_nvfp4)
  hybrid        the plugin's hybrid path: ref's GEMM, then our shrink and expand-add
  plugin        the best of fused, hybrid and stock_tuned (what the plugin's timed dispatch picks among, tuned)
  pdl           the new path: the shrink kernel, then the block-scaled FP4 GEMM with programmatic dependent launch, at
                its default configuration (t_pdl_default) and the best of a sweep (t_pdl; pdl_best; all in pdl_sweep);
                t_pdl_sweep_x0 / _x1: the sweep's best with the GEMM quantizing x itself (X_MODE 0) / with the shrink
                kernel's quantization programs (X_MODE 1; HOPPER-PDL-K2's winner, the default)
  pdl_nopdl     pdl_best without the PDL attribute; pdl_noshrink: without the adapter shrink (the overlap's bound);
  shrink_only   the shrink kernel alone; k2_nolora: the new GEMM without LoRA, best of a small sweep (against ref)
  layer_*       one decoder layer's four projections in one graph (L2 flushed once per replay): ref, stock, stock_tuned,
                stock_dual, fused, hybrid, pdl
Checks: every output against stock's (bf16 adapters; relative error at most 2e-2), FP8-/NVFP4-stored adapters against the
plugin's fused path on the same stored adapters; pdl_best repeated bitwise (determinism); the GEMM's own codes and E4M3
block scales (X_MODE 0, debug output) and the quantization programs' (X_MODE 1) against vLLM's scaled_fp4_quant (linear
layout), byte for byte; our GEMM without LoRA against CUTLASS on vLLM's codes (bitwise share, descriptive).
Stages in priority order under --budget-min: headline (r 64, M 1/4/8/16, one adapter, bf16: coarse grid + refinement),
largeM (M 32/64/128), adapters (8 and 32 adapters, M 8/16/32/64), ranks (16, 128 at M 1/16), formats (FP8- and
NVFP4-stored adapters, M 1/16, n 1/8). Before each phase's timings --workers processes compile its kernels. --smoke: two
cases (o M 1, down M 16 with 8 adapters), exit 1 unless both rows pass the new path's checks (the job's gate).
    python bench/pdl_fp4_bench.py --out results/pdl_fp4.jsonl [--budget-min 24] [--workers 8] [--stages ...] [--smoke]
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
TUNED = REPO / "results/vllm_lora_tuned/Qwen3-8B-NVFP4_bw"
MODELS = {"qwen3_8b": SHAPES["qwen3_8b"]}
FULL = ("headline", "smoke")


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def kernel_sha() -> str:
    h = hashlib.sha256()
    for f in ("lowbit_lora/kernels/nvfp4_lora_pdl.py", "lowbit_lora/blackwell_pdl.py", "lowbit_lora/kernels/fp8_lora_pdl.py",
              "lowbit_lora/kernels/nvfp4_lora_vllm.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def n_sm() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


# ------------------------------------------------------------------------------------------------------- the sweep
def smem_ok4(c: dict) -> bool:
    """sm_120: at most 99 KB of shared memory per block. Per buffered stage (num_stages - 1 of them) the weight codes
    and block scales, and x's bf16 tile (X_MODE 0) or its codes and scales (X_MODE 1): 2-10% above the offline compiles'
    totals (tests/pdl_fp4_interp_check.py --compile: 25-51 KB at the bench's tiles); a launch that fails is recorded."""
    bm, bn, bk, st = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c.get("num_stages", 4)
    xb = bm * bk * 2 if c["X_MODE"] == 0 else bm * bk // 2 + bm * bk // 16
    return max(1, st - 1) * (bn * bk // 2 + bn * bk // 16 + xb) <= 96 * 1024


def base_cfg4(M: int, xm: int, swap: bool, bn: int, sk: int, bk: int = 256) -> dict:
    """(num_warps is left to blackwell_pdl.resolve_cfg4: from the final tile and the rank, nvfp4_lora_pdl.warps4.)"""
    bm = (8 if M <= 8 else 16) if swap else (16 if M <= 16 else 32 if M <= 32 else 64)
    return {"X_MODE": xm, "SWAP": swap, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk,
            "num_stages": 4 if bn <= 64 else 3, "W_PREFETCH": 2}


def coarse4(M: int, K: int, N: int, widths, nsm: int) -> list[dict]:
    """X_MODE x orientation x BLOCK_N x SPLIT_K at BLOCK_K 256: split-K while the projection has fewer than about two
    tiles per SM, at most 8 CTAs per SM, each split at least two K-blocks; swap-AB only up to 16 tokens."""
    out = []
    xms = (0, 1) if M <= 16 else (1,)
    swaps = (True, False) if M <= 16 else (False,)
    for xm, swap, bn, sk in itertools.product(xms, swaps, (64, 128), (1, 2, 4)):
        if any(w % bn for w in widths):
            continue
        c = base_cfg4(M, xm, swap, bn, sk)
        tiles = cdiv(M, c["BLOCK_M"]) * cdiv(N, bn)
        if K // sk < 2 * c["BLOCK_K"] or tiles * sk > 8 * nsm or (sk > 1 and tiles >= 2 * nsm):
            continue
        if smem_ok4(c):
            out.append(c)
    return out


def refine4(best: dict, K: int) -> list[dict]:
    """One change at a time around the coarse best: the K-block, the pipeline depth, the weight stream's L2 prefetch
    (none, 8 K-steps, the CTA's whole split), B in L2 only, the shrink's split, the quantization programs' tile."""
    out = []

    def add(**kv):
        c = {**best, **kv}
        if c != best and smem_ok4(c) and K // c["SPLIT_K"] >= 2 * c["BLOCK_K"] and c not in out:
            out.append(c)
    add(BLOCK_K=128 if best["BLOCK_K"] == 256 else 256)
    for st in (best.get("num_stages", 4) - 1, best.get("num_stages", 4) + 1):
        if 2 <= st <= 6:
            add(num_stages=st)
    k_split = cdiv(cdiv(K, best["SPLIT_K"]), best["BLOCK_K"]) * best["BLOCK_K"]
    whole = 1 << ((k_split // best["BLOCK_K"]).bit_length() - 1)
    for wp in sorted({0, 8, whole} - {best.get("W_PREFETCH", 2)}):
        add(W_PREFETCH=wp)
    add(B_PREFETCH=1)
    split1 = best.get("SPLIT_K1") or (4 if K <= 4096 else 8)
    for s1 in (split1 // 2, split1 * 2):
        add(SPLIT_K1=s1)
    if best["X_MODE"] == 1:
        add(QBK=128)
    return out


def neighbors4(best: dict, K: int, M: int, N: int, nsm: int) -> list[dict]:
    out = [dict(best)]
    tiles = cdiv(M, best["BLOCK_M"]) * cdiv(N, best["BLOCK_N"])
    for sk in (best["SPLIT_K"] // 2, best["SPLIT_K"] * 2):
        if 1 <= sk <= 16 and K // sk >= 2 * best["BLOCK_K"] and tiles * sk <= 8 * nsm:
            out.append({**best, "SPLIT_K": sk})
    if M <= 16:
        out.append({**best, "X_MODE": 1 - min(best["X_MODE"], 1)})
    return [c for c in out if smem_ok4(c)]


def nolora_grid4(M: int, K: int, widths) -> list[dict]:
    out = []
    for xm, swap in ((0, True), (1, True), (1, False)):
        if not swap or M <= 16:
            for bn, sk in ((64, 1), (128, 1), (64, 2)):
                if not any(w % bn for w in widths) and K // sk >= 512:
                    out.append(base_cfg4(M, xm, swap, bn, sk))
    return out


# ---------------------------------------------------------------------------------------------------------- set-up
class Bank4:
    """NVFP4 weights per projection (our linear-layout scales and CUTLASS's swizzled ones), adapters and NVFP4 ops."""

    def __init__(self, max_m: int):
        self.max_m = max_m
        self.w = {}
        self.ops = {}

    def weight(self, model, proj):
        from vllm import _custom_ops as ops

        from lowbit_lora.kernels import nvfp4_lora_vllm as F4
        K, slices = MODELS[model][proj]
        key = (K, slices)
        if key not in self.w:
            g = torch.Generator(device="cuda").manual_seed(K + sum(slices) + len(slices))
            w = (torch.randn(sum(slices), K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
            w_global = F4.global_scale_for(w.float())
            wg = torch.tensor([w_global], dtype=torch.float32, device="cuda")
            w_q_sw, w_sf_sw = ops.scaled_fp4_quant(w, wg)                                  # CUTLASS layout
            w_q, w_sf = ops.scaled_fp4_quant(w, wg, is_sf_swizzled_layout=False)           # ours: linear scales
            self.w[key] = (w_q, w_sf, w_global, w_q_sw, w_sf_sw)
            del w
        return self.w[key]

    def op(self, model, proj, r, L, fmt, empty: bool = False):
        from lowbit_lora.kernels import nvfp4_lora_vllm as F4
        K, slices = MODELS[model][proj]
        key = (K, slices, r, L, fmt)
        if key not in self.ops:
            w_q, w_sf, w_global, _, _ = self.weight(model, proj)
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
            op = F4.VllmFusedNVFP4LoRA(w_q, w_sf, w_global, 1.0, a_st, b_st, self.max_m, lora_fmt=fmt)
            self.ops[key] = (op, a_st, b_st)
        return self.ops[key]

    def drop_ops(self):
        self.ops.clear()
        torch.cuda.empty_cache()


def make_meta(L, max_m, M, n):
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
    meta = LoRAKernelMeta.make(L, max_m, device="cuda")
    tok = (torch.arange(M, device="cuda") % n).to(torch.int32)
    meta.prepare_tensors(tok)
    return meta, meta.meta_args(M, False)


class Env4:
    """vLLM's ops, the tuned-config switch, the dual-stream resources, and the NVFP4 baselines of one case."""

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

    def fns(self, bank, model, proj, op, a_st, b_st, x, margs, xg, alpha):
        """vendor (scaled_fp4_quant + CUTLASS FP4 GEMM), stock (+ vLLM's LoRA kernels), dual (vLLM's dual stream)."""
        ops, ls, le = self.ops, self.lora_shrink, self.lora_expand
        _, _, _, w_q_sw, w_sf_sw = bank.weight(model, proj)
        M, S, r = x.shape[0], op.S, op.R
        hb = torch.empty(S * M * r, dtype=torch.float32, device="cuda").view(S, M, r)
        lo_buf = torch.empty(M, op.N, dtype=torch.bfloat16, device="cuda")

        def vendor(xx=None):
            xq, xs = ops.scaled_fp4_quant(x if xx is None else xx, xg)
            return ops.cutlass_scaled_fp4_mm(xq, w_q_sw, xs, w_sf_sw, alpha, torch.bfloat16)

        def stock():
            y = vendor()
            ls(x, a_st, hb, *margs, 1.0)
            le(hb, b_st, y, *margs, offset_start=0, add_inputs=True)
            return y

        def dual():
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


def case_x(M: int, K: int, n: int, device="cuda"):
    """Activations with outlier channels (as bench/nvfp4_check.py) and their static global scale (a checkpoint's
    input_scale is calibrated; here from the batch)."""
    from lowbit_lora.kernels import nvfp4_lora_vllm as F4
    g = torch.Generator(device=device).manual_seed(7 * M + n)
    x = (torch.randn(M, K, device=device, generator=g) * (torch.rand(M, 1, device=device, generator=g) * 2 + 0.1))
    x[:, :8] *= 10
    x = x.to(torch.bfloat16)
    return x, F4.global_scale_for(x.float())


# ------------------------------------------------------------------------------------------------------ precompile
PDL_KINDS = {"pdl": {}, "nopdl": dict(pdl=False), "nolora": dict(lora=False), "noshrink": dict(shrink=False),
             "shrinkonly": dict(gemm=False)}


def _precompile_worker4(items) -> tuple[int, list[str]]:
    """Compile every listed variant: items (model, proj, r, L, fmt, M, n, cfg, kind). kind "base": one call of each
    baseline (vendor, vLLM's kernels default / tuned / dual-stream, the plugin's fused and hybrid paths); "xqd": the new
    path with the debug output; else a run_pdl4 variant (compile only)."""
    import torch as T
    from lowbit_lora.blackwell_pdl import run_pdl4
    bank = Bank4(max(i[5] for i in items))
    env = Env4()
    errs, done = [], 0
    for model, proj, r, L, fmt, M, n, cfg, kind in items:
        try:
            op, a_st, b_st = bank.op(model, proj, r, L, fmt, empty=True)
            _, margs = make_meta(L, bank.max_m, M, n)
            x, xglob = case_x(M, op.K, n)
            xg = T.tensor([xglob], dtype=T.float32, device="cuda")
            op.x_global.copy_(xg)
            alpha = op.alpha
            if kind == "base":
                vendor, stock, dual = env.fns(bank, model, proj, op, a_st, b_st, x, margs, xg, alpha)
                for tuned in (False, True):
                    env.set_tuned(tuned)
                    stock()
                env.set_tuned(True, dual=True)
                dual()
                env.set_tuned(False)
                op(x, margs, hybrid_m=1 << 30, use_dispatch=False)
                op(x, margs, base_fn=vendor, hybrid_m=0, use_dispatch=False)
                T.cuda.synchronize()
            elif kind == "xqd":
                run_pdl4(op, x, margs, cfg=cfg, compile_only=True,
                         xqd=(T.zeros(M, op.K // 2, dtype=T.uint8, device="cuda"),
                              T.zeros(M, op.K // 16, dtype=T.float8_e4m3fn, device="cuda")))
            else:
                run_pdl4(op, x, margs, cfg=cfg, compile_only=True, **PDL_KINDS[kind])
            done += 1
        except Exception as e:   # noqa: BLE001
            errs.append(f"{proj} r{r} {fmt} M{M} {kind} {cfg}: {type(e).__name__}: {str(e)[:200]}")
    return done, errs


def precompile4(items: list, workers: int, label: str) -> None:
    if not items or workers <= 0:
        return
    import multiprocessing as mp
    t0 = time.time()
    groups: dict = {}
    for it in items:
        model, proj, r, L, fmt, M, n, cfg, kind = it
        key = (model, proj, r, L, fmt, M > 16) if kind == "base" else (r, L, fmt, json.dumps(cfg, sort_keys=True),
                                                                        kind)
        groups.setdefault(json.dumps(key, default=str), []).append(it)
    chunks = [[] for _ in range(workers)]
    for g in sorted(groups.values(), key=len, reverse=True):
        min(chunks, key=len).extend(g)
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_precompile_worker4, [c for c in chunks if c])
    except Exception as e:   # noqa: BLE001
        print(f"[precompile {label}] workers failed ({type(e).__name__}: {str(e)[:200]}); compiling on demand",
              flush=True)
        return
    errs = [e for _, es in res for e in es]
    print(f"[precompile {label}] {sum(d for d, _ in res)}/{len(items)} calls, {len(groups)} kernel families, "
          f"{time.time() - t0:.0f} s{f'; {len(errs)} failed, first: {errs[0]}' if errs else ''}", flush=True)


# ----------------------------------------------------------------------------------------------------------- cases
def rel(y, ref) -> float:
    return float((y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))


class Runner4:
    """blackwell_pdl with the start-up probe's verdict: without PDL capture in CUDA graphs, no PDL attribute."""

    def __init__(self):
        from lowbit_lora import blackwell_pdl as B
        self.B = B
        self.pdl_ok = True

    def run_pdl4(self, *a, **kw):
        if not self.pdl_ok:
            kw["pdl"] = False
        return self.B.run_pdl4(*a, **kw)

    def resolve_cfg4(self, *a, **kw):
        return self.B.resolve_cfg4(*a, **kw)


class Case4:
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
        self.errs, self.tried = {}, []
        self.d = None
        self.best = (float("inf"), None)
        self.H = ctx.runner

    def item(self, cfg, kind):
        return (self.model, self.proj, self.r, self.L, self.fmt, self.M, self.n, cfg, kind)

    def bind(self):
        """The op's static scales (x_global; alpha = 1 / (x_global w_global), also CUTLASS's) are shared by the cases of
        one projection, rank, adapter count and format: this case's before any of its launches."""
        self.op.x_global.copy_(self.xg)
        self.op.alpha.copy_(self.alpha_t)

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
        return e <= 2e-2

    def setup(self):
        ctx = self.ctx
        self.op, a_st, b_st = ctx.bank.op(self.model, self.proj, self.r, self.L, self.fmt)
        op = self.op
        _, self.margs = make_meta(self.L, ctx.bank.max_m, self.M, self.n)
        self.x, xglob = case_x(self.M, self.K, self.n)
        x, margs = self.x, self.margs
        _, _, w_global, _, _ = ctx.bank.weight(self.model, self.proj)
        self.xg = torch.tensor([xglob], dtype=torch.float32, device="cuda")
        self.alpha_t = torch.tensor([1.0 / (xglob * w_global)], dtype=torch.float32, device="cuda")
        self.bind()
        self.vendor, stock, dual = ctx.fns(ctx.bank, self.model, self.proj, op, a_st, b_st, x, margs, self.xg, op.alpha)
        self.a_st, self.b_st = a_st, b_st
        ctx.set_tuned(False)
        self.y_stock = stock().float().clone()
        self.y_base = self.vendor().float()
        self.timed("ref", self.vendor)
        self.timed("stock", stock)
        if ctx.tuned_ok:
            try:
                ctx.set_tuned(True)
                self.check("stock_tuned", stock(), self.y_stock)
                self.timed("stock_tuned", stock)
            except Exception as e:   # noqa: BLE001
                self.rec["fail_stock_tuned"] = f"{type(e).__name__}: {str(e)[:300]}"
        self.rec["stock_dual_tuned"] = ctx.tuned_ok          # (without the slice's files: vLLM's default configs)
        try:
            ctx.set_tuned(True, dual=True)
            self.check("stock_dual", dual(), self.y_stock)
            self.timed("stock_dual", dual)
        except Exception as e:   # noqa: BLE001
            self.rec["fail_stock_dual"] = f"{type(e).__name__}: {str(e)[:300]}"
        ctx.set_tuned(False)
        y_f = op(x, margs, hybrid_m=1 << 30, use_dispatch=False).float().clone()
        self.ref_fmt = self.y_stock if self.fmt == "bf16" else y_f       # stored adapters: the fused path on them
        if self.fmt == "bf16":
            self.check("fused", y_f)
        self.timed("fused", lambda: op(x, margs, hybrid_m=1 << 30, use_dispatch=False))
        try:
            self.check("hybrid", op(x, margs, base_fn=self.vendor, hybrid_m=0, use_dispatch=False))
            self.timed("hybrid", lambda: op(x, margs, base_fn=self.vendor, hybrid_m=0, use_dispatch=False))
        except Exception as e:   # noqa: BLE001
            self.rec["fail_hybrid"] = f"{type(e).__name__}: {str(e)[:300]}"
        cands = [self.rec.get(k) for k in ("t_fused", "t_hybrid", "t_stock_tuned") if self.rec.get(k) is not None]
        self.rec["t_plugin"] = min(cands) if cands else None
        # (without num_warps: the resolver derives it from each variant's final tile and the rank, nvfp4_lora_pdl.warps4)
        self.d = {k: v for k, v in self.H.resolve_cfg4(op, self.M).items() if k != "num_warps"}
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

    def pdl_time(self, c, timer=None, **kw):
        H, op, x, margs = self.H, self.op, self.x, self.margs
        e = rel(H.run_pdl4(op, x, margs, cfg=c, **kw), self.ref_fmt)
        if e > 2e-2:
            return None, e
        return (timer or self.ctx.fast).graph_time(lambda: H.run_pdl4(op, x, margs, cfg=c, **kw)), e

    def candidates(self, phase: str) -> list[dict]:
        nsm = self.ctx.nsm
        if phase == "coarse":
            if self.level in ("full", "large"):
                return coarse4(self.M, self.K, self.N, self.slices, nsm)
            seed = self.ctx.best.get(self.key) or self.ctx.best.get((self.K, self.N, 16)) or self.d
            return neighbors4(seed, self.K, self.M, self.N, nsm)
        if phase == "refine" and self.level == "full" and self.best[1] is not None:
            return refine4(self.best[1], self.K)
        return []

    def sweep(self, cands):
        self.bind()
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
        bc = self.best[1] or self.d
        kinds = ["pdl", "nopdl"] + (["noshrink", "shrinkonly"] if self.full else [])
        it = [self.item(bc, k) for k in kinds] + [self.item({**bc, "SPLIT_K": 1}, "nolora")]
        if self.full or self.stage == "largeM":
            it += [self.item({**bc, "X_MODE": xm}, "nolora") for xm in (0, 1)]
        return it

    def finish(self, out_path: Path):
        ctx, H, op, x, margs = self.ctx, self.H, self.op, self.x, self.margs
        self.bind()
        rec = self.rec
        bc = self.best[1] or self.d
        rec["pdl_best"] = bc
        rec["pdl_best_resolved"] = H.resolve_cfg4(op, self.M, bc)
        rec["pdl_sweep"] = self.tried
        for xm in (0, 1):   # the registered form (X_MODE 0: the GEMM quantizes x itself) and the quantization programs'
            ts = [t["t"] for t in self.tried if t.get("t") is not None and t["cfg"].get("X_MODE") == xm]
            if rec.get("t_pdl_default") is not None and self.d.get("X_MODE") == xm:
                ts.append(rec["t_pdl_default"])
            rec[f"t_pdl_sweep_x{xm}"] = min(ts) if ts else None
        late = ctx.over()
        rec["late"] = late
        if self.level == "full" or self.key not in ctx.best:
            ctx.best[self.key] = bc
        try:
            y_p = H.run_pdl4(op, x, margs, cfg=bc).float().clone()
            self.check("pdl", y_p)
            reps = [H.run_pdl4(op, x, margs, cfg=bc).float() for _ in range(3)]
            rec["pdl_deterministic"] = all(torch.equal(y_p, yy) for yy in reps)
            self.timed("pdl", lambda: H.run_pdl4(op, x, margs, cfg=bc))
            self.check("pdl_nopdl", H.run_pdl4(op, x, margs, cfg=bc, pdl=False))
            self.timed("pdl_nopdl", lambda: H.run_pdl4(op, x, margs, cfg=bc, pdl=False))
            if self.full and not late:
                H.run_pdl4(op, x, margs, cfg=bc)
                self.check("pdl_noshrink", H.run_pdl4(op, x, margs, cfg=bc, shrink=False))
                self.timed("pdl_noshrink", lambda: H.run_pdl4(op, x, margs, cfg=bc, shrink=False))
                self.timed("shrink_only", lambda: H.run_pdl4(op, x, margs, cfg=bc, gemm=False))
        except Exception as ex:   # noqa: BLE001
            rec["fail_pdl"] = f"{type(ex).__name__}: {str(ex)[:300]}"
            traceback.print_exc()
        # vLLM's codes and block scales (linear layout), byte for byte: the GEMM's own (X_MODE 0) and the quantization
        # programs' (X_MODE 1), at the default tiles; our no-adapter GEMM against CUTLASS on vLLM's codes
        try:
            ops = ctx.ops
            xq_v, xs_v = ops.scaled_fp4_quant(x, self.xg, is_sf_swizzled_layout=False)
            xs_v = xs_v.view(torch.uint8).reshape(self.M, -1)[:, : self.K // 16]
            xqd = (torch.zeros(self.M, self.K // 2, dtype=torch.uint8, device="cuda"),
                   torch.zeros(self.M, self.K // 16, dtype=torch.float8_e4m3fn, device="cuda"))
            H.run_pdl4(op, x, margs, cfg={**self.d, "X_MODE": 0}, xqd=xqd)
            rec["xq_mode0_codes_differing"] = int((xqd[0] != xq_v.view(torch.uint8)[:, : self.K // 2]).sum())
            rec["xq_mode0_scales_differing"] = int((xqd[1].view(torch.uint8) != xs_v).sum())
            H.run_pdl4(op, x, margs, cfg={**self.d, "X_MODE": 1})
            xq1 = op.scratch.xq[: self.M * (self.K // 2)].view(self.M, self.K // 2)
            sx1 = op.scratch.sxq[: self.M * (self.K // 16)].view(self.M, self.K // 16)
            rec["xq_mode1_codes_differing"] = int((xq1 != xq_v.view(torch.uint8)[:, : self.K // 2]).sum())
            rec["xq_mode1_scales_differing"] = int((sx1.view(torch.uint8) != xs_v).sum())
            y0 = H.run_pdl4(op, x, margs, cfg={**bc, "SPLIT_K": 1}, lora=False)
            rec["gemm_bitwise_vs_cutlass"] = float((y0.view(torch.int16) == self.vendor().view(torch.int16))
                                                   .float().mean())
        except Exception as ex:   # noqa: BLE001
            rec["fail_xq_check"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        if (self.full or self.stage == "largeM") and not late:
            nl = []
            grid = [{**bc, "X_MODE": 0}, {**bc, "X_MODE": 1}] + (nolora_grid4(self.M, self.K, self.slices)
                                                                  if self.full else [])
            for c in grid:
                try:
                    e0 = rel(H.run_pdl4(op, x, margs, cfg=c, lora=False), self.y_base)
                    if e0 > 2e-2:
                        nl.append({"cfg": c, "err": e0})
                        continue
                    t0 = ctx.fast.graph_time(lambda: H.run_pdl4(op, x, margs, cfg=c, lora=False))
                    nl.append({"cfg": c, "t": round(t0, 2), "err": float(f"{e0:.3g}")})
                except Exception as ex:   # noqa: BLE001
                    nl.append({"cfg": c, "fail": f"{type(ex).__name__}: {str(ex)[:160]}"})
            okn = [v for v in nl if v.get("t") is not None]
            if okn:
                b0 = min(okn, key=lambda v: v["t"])
                self.timed("k2_nolora", lambda: H.run_pdl4(op, x, margs, cfg=b0["cfg"], lora=False))
                rec["k2_nolora_cfg"] = b0["cfg"]
                self.errs["k2_nolora_vs_ref"] = b0["err"]
            rec["k2_nolora_sweep"] = nl
        rec["err"] = self.errs
        rec["checks_ok"] = all(v <= 2e-2 for v in self.errs.values())
        rec["pdl_checks_ok"] = (all(v <= 2e-2 for k, v in self.errs.items() if k.split("_")[0] in ("pdl", "k2"))
                                and rec.get("xq_mode0_codes_differing") == 0 and rec.get("xq_mode1_codes_differing") == 0
                                and rec.get("xq_mode0_scales_differing") == 0
                                and rec.get("xq_mode1_scales_differing") == 0 and bool(rec.get("pdl_deterministic")))
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["elapsed_s"] = round(ctx.elapsed())
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        show = ("ref", "stock", "stock_tuned", "stock_dual", "fused", "hybrid", "plugin", "pdl", "pdl_nopdl",
                "pdl_noshrink", "k2_nolora")
        print(f"{self.stage:8s} {self.proj:7s} r{self.r:<3d} M{self.M:<3d} n{self.n:<2d} {self.fmt:5s} | "
              + " ".join(f"{k} {rec.get('t_' + k, '-')}" for k in show)
              + f" | best {json.dumps({k: bc[k] for k in ('X_MODE', 'SWAP', 'BLOCK_M', 'BLOCK_N', 'SPLIT_K') if k in bc})}"
              f" ok={rec['checks_ok']} pdl_ok={rec['pdl_checks_ok']} [{ctx.elapsed():.0f}s]", flush=True)
        if self.stage == "headline" and self.n == 1 and self.r == 64 and self.fmt == "bf16":
            ctx.layer.setdefault(self.M, {})[self.proj] = self


class Ctx4(Env4):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.fast = Timer(warm=3, reps=15)
        self.acc = Timer(warm=5, reps=40)
        self.t0 = time.time()
        self.budget = a.budget_min * 60
        self.bank = Bank4(128)
        self.nsm = n_sm()
        self.runner = Runner4()
        self.info = {**gpu_info(), "n_sm": self.nsm, "job": os.environ.get("SLURM_JOB_ID", ""),
                     "kernel_sha": os.environ.get("LOWBIT_SHA", "") or kernel_sha(), "torch": torch.__version__,
                     "tuned_configs_found": self.tuned_ok}
        self.best = {}
        self.layer = {}

    def elapsed(self) -> float:
        return time.time() - self.t0

    def over(self) -> bool:
        return self.elapsed() > self.budget


def probe_capture(ctx) -> object:
    from lowbit_lora.blackwell_pdl import run_pdl4
    from lowbit_lora.kernels.nvfp4_lora_pdl import work
    op, _, _ = ctx.bank.op("qwen3_8b", "o", 16, 8, "bf16")
    _, margs = make_meta(8, ctx.bank.max_m, 1, 1)
    x, xglob = case_x(1, op.K, 1)
    op.x_global.fill_(xglob)
    try:
        ctx.fast.graph_time(lambda: run_pdl4(op, x, margs, cfg={"SPLIT_K": 2}))
        return True
    except Exception as e:   # noqa: BLE001
        try:
            torch.cuda.synchronize()
            work(x.device).reset()
        except Exception:   # noqa: BLE001
            pass
        return f"{type(e).__name__}: {str(e)[:200]}"


def layer_graphs4(ctx: Ctx4, M: int, out_path: Path) -> None:
    projs = ["qkv", "o", "gate_up", "down"]
    have = ctx.layer.get(M, {})
    if not all(p in have for p in projs):
        return
    for p in projs:   # (four projections, four ops: each holds its case's static scales for the whole layer)
        have[p].bind()
    fns = {p: ctx.fns(ctx.bank, have[p].model, p, have[p].op, have[p].a_st, have[p].b_st, have[p].x, have[p].margs,
                      have[p].xg, have[p].op.alpha) for p in projs}

    def seq(kind):
        def f():
            ys = []
            for p in projs:
                c = have[p]
                vendor, stock, dual = fns[p]
                if kind == "ref":
                    ys.append(vendor())
                elif kind in ("stock", "stock_tuned"):
                    ys.append(stock())
                elif kind == "stock_dual":
                    ys.append(dual())
                elif kind == "fused":
                    ys.append(c.op(c.x, c.margs, hybrid_m=1 << 30, use_dispatch=False))
                elif kind == "hybrid":
                    ys.append(c.op(c.x, c.margs, base_fn=vendor, hybrid_m=0, use_dispatch=False))
                else:
                    ys.append(c.H.run_pdl4(c.op, c.x, c.margs, cfg=c.best[1] or c.d))
            return ys
        return f
    rec = {**ctx.info, "stage": "layer", "model": "qwen3_8b", "M": M, "n": 1, "r": 64, "projs": projs,
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for kind in ("ref", "stock", "stock_tuned", "stock_dual", "fused", "hybrid", "pdl"):
        if kind == "stock_tuned" and not ctx.tuned_ok:
            continue
        ctx.set_tuned(kind in ("stock_tuned", "stock_dual"), dual=kind == "stock_dual")
        try:
            rec["t_layer_" + kind] = round(ctx.acc.graph_time(seq(kind)), 2)
        except Exception as e:   # noqa: BLE001
            rec["fail_layer_" + kind] = f"{type(e).__name__}: {str(e)[:300]}"
    ctx.set_tuned(False)
    with out_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"layer    M{M:<3d}: " + " ".join(f"{k[8:]} {rec[k]}" for k in rec if k.startswith("t_layer_")), flush=True)


def plan(stages: set, smoke: bool) -> list[tuple]:
    q8 = list(MODELS["qwen3_8b"])
    if smoke:
        return [("smoke", "qwen3_8b", "o", 64, 1, 1, "bf16", "small"),
                ("smoke", "qwen3_8b", "down", 64, 16, 8, "bf16", "small")]
    out = []
    if "headline" in stages:
        out += [("headline", "qwen3_8b", p, 64, M, 1, "bf16", "full") for M in (1, 4, 8, 16) for p in q8]
    if "largeM" in stages:
        out += [("largeM", "qwen3_8b", p, 64, M, 1, "bf16", "large") for M in (32, 64, 128) for p in q8]
    if "adapters" in stages:
        out += [("adapters", "qwen3_8b", p, 64, M, n, "bf16", "small") for n in (8, 32) for M in (8, 16, 32, 64)
                if n <= M for p in q8]
    if "ranks" in stages:
        out += [("ranks", "qwen3_8b", p, r, M, 1, "bf16", "small") for r in (16, 128) for M in (1, 16) for p in q8]
    if "formats" in stages:
        out += [("formats", "qwen3_8b", p, 64, M, n, fmt, "small") for fmt in ("fp8", "nvfp4") for M in (1, 16)
                for n in (1, 8) if n <= M for p in q8]
    return out


def run_group4(ctx: Ctx4, specs: list[tuple], out_path: Path, workers: int) -> None:
    from lowbit_lora.blackwell_pdl import resolve_cfg4
    if ctx.over():
        print(f"[budget] skipping {len(specs)} cases of {sorted({s[0] for s in specs})}", flush=True)
        return
    items = []
    for s in specs:
        stage, model, proj, r, M, n, fmt, level = s
        K, slices = MODELS[model][proj]
        L = 32 if n > 8 else 8
        op, _, _ = ctx.bank.op(model, proj, r, L, fmt, empty=True)
        d = {k: v for k, v in resolve_cfg4(op, M).items() if k != "num_warps"}
        it = lambda c, k: (model, proj, r, L, fmt, M, n, c, k)   # noqa: E731
        items += [it(None, "base"), it(d, "pdl"), it({**d, "X_MODE": 0}, "xqd"), it({**d, "X_MODE": 1}, "pdl"),
                  it({**d, "SPLIT_K": 1}, "nolora")]
        if level in ("full", "large"):
            items += [it(c, "pdl") for c in coarse4(M, K, sum(slices), slices, ctx.nsm)]
        else:
            seed = ctx.best.get((K, sum(slices), 16 if M <= 16 else M)) or ctx.best.get((K, sum(slices), 16)) or d
            items += [it(c, "pdl") for c in neighbors4(seed, K, M, sum(slices), ctx.nsm)]
        if stage in FULL:
            items += [it(c, "nolora") for c in nolora_grid4(M, K, slices)]
    ctx.bank.drop_ops()                                     # (the empty ops above only resolved the defaults)
    precompile4(items, workers, f"{specs[0][0]}: baselines + coarse")
    cases = []
    for s in specs:
        if ctx.over():
            break
        c = Case4(ctx, *s)
        try:
            c.setup()
            c.sweep(c.candidates("coarse"))
            cases.append(c)
        except Exception:   # noqa: BLE001
            traceback.print_exc()
    ref_items = [c.item(cfg, "pdl") for c in cases for cfg in c.candidates("refine")]
    if not ctx.over():
        precompile4(ref_items, workers, f"{specs[0][0]}: refinement")
    for c in cases:
        c.sweep(c.candidates("refine"))
    if not ctx.over():
        precompile4([i for c in cases for i in c.extra_items()], workers, f"{specs[0][0]}: final variants")
    for c in cases:
        try:
            c.finish(out_path)
        except Exception:   # noqa: BLE001
            traceback.print_exc()


def summarize4(out_path: Path, job: str) -> None:
    rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("job") == job]
    keys = ("ref", "stock", "stock_tuned", "stock_dual", "fused", "hybrid", "plugin", "pdl", "pdl_sweep_x0",
            "pdl_sweep_x1", "pdl_nopdl", "k2_nolora")
    print("\n== one Qwen3-8B layer's four projections summed (us), r 64, one adapter")
    for M in (1, 4, 8, 16, 32, 64, 128):
        rs = {r["proj"]: r for r in rows if r.get("n") == 1 and r.get("r") == 64 and r.get("fmt") == "bf16"
              and r.get("M") == M and r.get("stage") in ("headline", "largeM")}
        if not all(p in rs for p in ("qkv", "o", "gate_up", "down")):
            continue
        tot = {k: sum(rs[p].get("t_" + k) or float("nan") for p in ("qkv", "o", "gate_up", "down")) for k in keys}
        print(f"  M {M:3d}: " + " ".join(f"{k} {v:.1f}" for k, v in tot.items())
              + f" | pdl/ref {tot['pdl'] / tot['ref']:.3f} pdl/stock_tuned {tot['pdl'] / tot['stock_tuned']:.3f}"
              f" pdl/plugin {tot['pdl'] / tot['plugin']:.3f} k2_nolora/ref {tot['k2_nolora'] / tot['ref']:.3f}")
    lay = [r for r in rows if r.get("stage") == "layer"]
    if lay:
        print("== one layer in one graph (us)")
        for r in lay:
            print(f"  M {r['M']:3d}: " + " ".join(f"{k[8:]} {r[k]}" for k in r if k.startswith("t_layer_")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=24.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stages", default="headline,largeM,adapters,ranks,formats")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cc = torch.cuda.get_device_capability()
    assert cc[0] == 12, f"needs Blackwell sm_12x (the RTX PRO 6000 slice), got sm_{cc[0]}{cc[1]}"
    torch.manual_seed(0)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = Ctx4(a)
    print(f"[pdl_fp4_bench] {ctx.info}; cutlass fp4 supported: "
          f"{ctx.ops.cutlass_scaled_mm_supports_fp4(cc[0] * 10 + cc[1])}; budget {a.budget_min} min", flush=True)
    if not ctx.tuned_ok:
        print(f"[pdl_fp4_bench] WARNING: no tuned vLLM LoRA configs for {torch.cuda.get_device_name()} in {TUNED}: "
              "stock_tuned not timed, stock_dual with vLLM's default configurations", flush=True)
    probe = probe_capture(ctx)
    ctx.info["pdl_capture"] = probe
    ctx.runner.pdl_ok = probe is True
    print(f"[pdl_fp4_bench] PDL capture probe: {probe}", flush=True)
    specs = plan(set(a.stages.split(",")), a.smoke)
    groups = []
    for s in specs:
        if groups and groups[-1][0][0] == s[0]:
            groups[-1].append(s)
        else:
            groups.append([s])
    for g in groups:
        if g[0][0] not in ("headline", "smoke"):
            ctx.bank.drop_ops()
        run_group4(ctx, g, out_path, a.workers)
        if g[0][0] == "headline":
            for M in (1, 4, 8, 16):
                layer_graphs4(ctx, M, out_path)
            ctx.layer.clear()
    summarize4(out_path, ctx.info["job"])
    print(f"[pdl_fp4_bench] done in {ctx.elapsed() / 60:.1f} min", flush=True)
    if a.smoke:   # the job's gate: both cases written, the new path's checks passed (outputs, codes, determinism)
        rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
        rows = [r for r in rows if r.get("job") == ctx.info["job"] and r.get("stage") == "smoke"]
        bad = [r for r in rows if not r.get("pdl_checks_ok") or any(k.startswith("fail_pdl") for k in r)]
        if len(rows) < len(specs) or bad:
            print(f"[pdl_fp4_bench] SMOKE: {len(rows)}/{len(specs)} rows, {len(bad)} failing the new path's checks: "
                  + "; ".join(f"{r['proj']} M{r['M']} err {r.get('err')} "
                              + str({k: v for k, v in r.items() if k.startswith(('fail_', 'xq_'))}) for r in bad),
                  flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
