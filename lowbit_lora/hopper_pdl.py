"""The PDL decode path (kernels/fp8_lora_pdl.py) as a path of the plugin's fused op on Hopper, without touching the
registered plugin code: run_pdl(op, x, meta_args) launches the shrink kernel (K1's adapter segments; with X_MODE 1 or 2 also
per-token quantization programs) and then the swap-AB GEMM with programmatic dependent launch, so the GEMM streams its
weights while the shrink runs and waits for it only before its epilogue. op is a VllmFusedFP8LoRA (or anything with its
fields: w_q, s_w, N, K, R, S, L, widths, end0-2, a_ptrs, sa_ptrs, b_ptrs, sb_ptrs, ag, bg, stride_al, stride_am, scratch,
lora_fp8, lora_fp4, dot_f32). Kernel benchmark first (bench/pdl_k2_bench.py); a plugin path only through an opt-in patch
registered before its data (install(), LOWBIT_PDL=1, as hopper_swap).
    from lowbit_lora.hopper_pdl import run_pdl; y = run_pdl(op, x, meta.meta_args(M, False))
"""
from __future__ import annotations

import os

import torch
import triton

from lowbit_lora.kernels import fp8_lora_vllm as F
from lowbit_lora.kernels.fp8_lora_pdl import _coop, _gemm_pdl, _shrink_pdl, default_cfg, work

CFG: dict = {}   # (bucket of M, N, K, n_slices, R, adapter format) -> swept config (bench/pdl_k2_bench.py), else defaults
AMAX_ELEMS = 16384   # X_MODE 0: elements of x per amax load step (128 per thread at 4 warps)
QBLOCK = 4096        # X_MODE 1/2: elements of a token row per quantization step


def _fmt(op) -> str:
    return "fp4" if op.lora_fp4 else "fp8" if op.lora_fp8 else "bf16"


def resolve_cfg(op, M: int, cfg: dict | None = None) -> dict:
    """The defaults of fp8_lora_pdl.default_cfg, then the swept table, then cfg (later keys win)."""
    c = default_cfg(M, op.N, op.K, op.widths, op.R, op.lora_fp4)
    c.update(CFG.get((F._bucket(M), op.N, op.K, op.S, op.R, _fmt(op)), {}))
    if cfg:
        c.update(cfg)
    return c


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def _wdesc(op, bn: int, bk: int):
    from triton.tools.tensor_descriptor import TensorDescriptor
    d = getattr(op, "_pdl_wdesc", None)
    if d is None:
        d = op._pdl_wdesc = {}
    if (bn, bk) not in d:                                    # W never changes: one descriptor per tile shape
        d[(bn, bk)] = TensorDescriptor.from_tensor(op.w_q, [bn, bk])
    return d[(bn, bk)]


def _geometry(op, x, c: dict, lora: bool):
    """Launch geometry shared by the PDL and cooperative launches."""
    M, K = x.shape
    S, R = op.S, op.R
    sc = op.scratch
    split1 = int(c["SPLIT_K1"])
    while split1 > 1 and split1 * S * M * R > sc.hp.numel():
        split1 //= 2                                                  # what the shared scratch holds
    bsr = min(64, R)
    n_sr = S * R // bsr
    s_bm = 16
    nx = triton.cdiv(M, s_bm) * n_sr * split1 if lora else 0
    xm = int(c["X_MODE"])
    gx = max(nx, M) if xm != 0 else max(nx, 1)                        # quantization programs: one token each
    bm, bn, bk, sk = int(c["BLOCK_M"]), int(c["BLOCK_N"]), int(c["BLOCK_K"]), int(c["SPLIT_K"])
    assert all(w % bn == 0 for w in op.widths), (bn, op.widths)
    k_split = triton.cdiv(triton.cdiv(K, sk), bk) * bk
    n_tiles = triton.cdiv(M, bm) * triton.cdiv(op.N, bn)
    br = max(16, triton.next_power_of_2(R))
    # the epilogue's rank chunk: 64 for NVFP4-stored B (vectorized dequantization) and for tiles above 8192 elements
    # (BLOCK_N 128 at rank 128 spilled: 255 registers + 184 bytes of stack), the whole rank otherwise
    rck = int(c.get("R_CHUNK") or (64 if (op.lora_fp4 or bn * br > 8192) else br))
    rck = min(rck, br)
    bp = int(c.get("B_PREFETCH", 2))
    if rck < br:
        bp = min(bp, 1)                                               # rank chunks: the L2 prefetch only
    bpipe = bool(c.get("B_PIPE", bn * rck <= 4096))                  # two B tiles live in the adapter loop
    # X_MODE 0's amax prologue: one token row per step at M = 1, four rows otherwise (AMAX_ELEMS elements per step);
    # fixed values in the other modes, so that their compiled kernels do not depend on M
    am = (1 if M == 1 else min(4, bm)) if xm == 0 else 1
    abk = max(64, min(triton.next_power_of_2(K), AMAX_ELEMS // am)) if xm == 0 else 64
    return dict(M=M, K=K, split1=split1, bsr=bsr, n_sr=n_sr, s_bm=s_bm, nx=nx, gx=gx, xm=xm, bm=bm, bn=bn, bk=bk,
                sk=sk, k_split=k_split, n_tiles=n_tiles, br=br, rck=rck, bp=bp, bpipe=bpipe, am=am, abk=abk,
                kq=triton.cdiv(K, QBLOCK) * QBLOCK)


def _k2_consts(op, c: dict, g: dict, interp: bool, precise: bool) -> dict:
    return dict(BLOCK_M=g["bm"], BLOCK_N=g["bn"], BLOCK_K=g["bk"], GROUP_N=int(c.get("GROUP_N", 8)), SPLIT_K=g["sk"],
                EVEN_K=g["K"] % g["bk"] == 0, X_MODE=g["xm"], QRN=2 if interp else 1, AMAX_M=g["am"],
                AMAX_BK=g["abk"], SPLIT_K1=g["split1"], BLOCK_R=g["br"], R_CHUNK=g["rck"],
                LP=triton.next_power_of_2(op.L + 1), B_PREFETCH=g["bp"], B_PIPE=g["bpipe"],
                W_PREFETCH=int(c.get("W_PREFETCH", 0)),
                W_TMA=bool(c.get("W_TMA", False)) and not interp, PRECISE=precise, B_FP8=op.lora_fp8,
                B_FP4=op.lora_fp4, DOT_F32=bool(getattr(op, "dot_f32", False)), GPU=not interp)


def _launch(fn, grid, args, kw, compile_only: bool):
    if compile_only:   # compile (into Triton's cache) without launching: the bench precompiles its sweep in parallel
        return None if _interpret() else fn.warmup(*args, grid=grid, **kw)
    return fn[grid](*args, **kw)


def run_pdl(op, x, meta_args, cfg: dict | None = None, out=None, precise: bool | None = None, pdl: bool = True,
            lora: bool = True, shrink: bool = True, gemm: bool = True, xqd=None, coop: bool = False,
            coop_launch: bool = True, compile_only: bool = False):
    """y = x W^T (FP8, x quantized per token as vLLM does) + every adapter's expansion, as the fused path computes it.
    pdl=False launches the GEMM without the PDL attribute (the same kernels, serialized: the overlap's control).
    lora=False: the GEMM without LoRA (no adapter shrink; with X_MODE 1/2 the quantization programs still run first).
    shrink=False: the GEMM with LoRA without the adapter shrink (h from the last shrink: the overlap's bound; with X_MODE
    1/2 the quantization programs still run first).
    gemm=False: the shrink kernel alone (the breakdown). compile_only: compile every kernel the call would launch, no launch.
    xqd: a [M, K] float8_e4m3fn buffer that receives the codes the GEMM quantized (X_MODE 0; bit-identity checks).
    coop=True: the merged fallback kernel (cooperative launch if coop_launch)."""
    tok, srt, seg_n, seg_start, seg_ids = meta_args[:5]
    M, K = x.shape
    interp = _interpret()
    c = resolve_cfg(op, M, cfg)
    if precise is None:
        precise = os.environ.get("LOWBIT_PRECISE", "1") != "0"
    g = _geometry(op, x, c, lora)
    sc = op.scratch
    hp, sxb = sc.hp, sc.sx
    xq = sc.xq[: M * K].view(M, K)
    wk = work(x.device)
    wk.ensure(g["sk"] * M * op.N if g["sk"] > 1 else 1, g["n_tiles"])
    y = out if out is not None else torch.empty(M, op.N, dtype=torch.bfloat16, device=x.device)
    W = _wdesc(op, g["bn"], g["bk"]) if (c.get("W_TMA") and not interp) else op.w_q
    xd = xqd if xqd is not None else xq
    kc = _k2_consts(op, c, g, interp, precise)
    kc["DEBUG_XQ"] = xqd is not None
    L1 = op.L + 1
    if coop:
        assert lora, "the merged kernel is the LoRA path"
        nsx = g["gx"]
        nsh = nsx * (L1 + (1 if g["xm"] != 0 else 0))
        kw = dict(num_warps=int(c.get("num_warps", 4)), num_stages=int(c.get("num_stages", 4)))
        if coop_launch and not interp:
            kw["launch_cooperative_grid"] = True
        _launch(_coop, (nsh + g["n_tiles"] * g["sk"],), (
            x, xq, sxb, W, op.w_q, op.s_w, hp, op.a_ptrs, op.sa_ptrs, op.ag, op.b_ptrs, op.sb_ptrs, op.bg, y, wk.p,
            wk.cnt, wk.sync, tok, srt, seg_ids, seg_n, seg_start, xd,
            M, op.N, K, op.R, op.S, L1, op.end0, op.end1, op.end2, g["k_split"], nsx, nsh, nsh, nsx, g["n_tiles"],
            x.stride(0), xq.stride(0), op.w_q.stride(0), y.stride(0), op.stride_al, op.stride_am),
            dict(S_SPLIT_K=g["split1"], N_SR=g["n_sr"], S_BLOCK_M=g["s_bm"], BLOCK_SR=g["bsr"],
                 S_BLOCK_K=int(c.get("S_BLOCK_K", 128)), QBLOCK=QBLOCK, KQ=g["kq"],
                 A_FP8=op.lora_fp8, A_FP4=op.lora_fp4, **kc, **kw), compile_only)
        return y
    qseg = g["xm"] != 0                       # X_MODE 1/2 read x_q from the quantization programs: always launched
    adapters = lora and shrink
    launched_shrink = False
    if adapters or qseg:
        nz = (L1 if adapters else 0) + (1 if qseg else 0)
        kw = dict(num_warps=4, num_stages=int(c.get("S_STAGES", 2)))
        _launch(_shrink_pdl, (g["gx"], 1, nz), (
            x, op.a_ptrs, op.sa_ptrs, op.ag, hp, xq, sxb, wk.sync, srt, seg_ids, seg_n, seg_start,
            M, K, op.R, op.S, L1, g["nx"], x.stride(0), op.stride_al, op.stride_am, xq.stride(0)),
            dict(SPLIT_K=g["split1"], N_SR=g["n_sr"], BLOCK_M=g["s_bm"], BLOCK_SR=g["bsr"],
                 BLOCK_K=int(c.get("S_BLOCK_K", 128)), QSEG=qseg, QSIGNAL=g["xm"] == 1 and gemm, QBLOCK=QBLOCK,
                 KQ=g["kq"], QRN=kc["QRN"], USE_GDC=pdl and gemm and not interp, DOT_F32=kc["DOT_F32"],
                 A_FP8=op.lora_fp8, A_FP4=op.lora_fp4, **kw), compile_only)
        launched_shrink = True
    if not gemm:
        return None
    # the GEMM waits (griddepcontrol.wait) only for a shrink kernel launched right before it; without one: SYNC_MODE 1
    sync_mode = 0 if launched_shrink else 1
    kw = dict(num_warps=int(c.get("num_warps", 4)), num_stages=int(c.get("num_stages", 4)))
    if pdl and launched_shrink and not interp:
        kw["launch_pdl"] = True
    _launch(_gemm_pdl, (g["n_tiles"], g["sk"]), (
        x, xq, sxb, W, op.w_q, op.s_w, hp, op.b_ptrs, op.sb_ptrs, op.bg, y, wk.p, wk.cnt, wk.sync, tok, seg_ids, seg_n,
        xd, M, op.N, K, op.R, op.S, L1, op.end0, op.end1, op.end2, g["k_split"], g["gx"], 0,
        x.stride(0), xq.stride(0), op.w_q.stride(0), y.stride(0)),
        dict(HAS_LORA=lora, SYNC_MODE=sync_mode, **kc, **kw), compile_only)
    return y


def load_cfg(path: str | None = None, job: str | None = None) -> int:
    """Swept configurations from bench/pdl_k2_bench.py (results/pdl_k2.jsonl, the newest job unless given): for each
    (bucket of M, N, K, slices, rank, adapter format), the one-adapter row's best configuration. Returns the number of
    entries."""
    import json
    from pathlib import Path
    f = Path(path or Path(__file__).resolve().parents[1] / "results/pdl_k2.jsonl")
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    rows = [r for r in rows if r.get("pdl_best") and r.get("n") == 1]
    if not rows:
        return 0
    job = job or max(r["job"] for r in rows)
    for r in rows:
        if r["job"] == job:
            CFG[(F._bucket(r["M"]), r["N"], r["K"], r["S"], r["r"], r.get("fmt", "bf16"))] = r["pdl_best"]
    return len(CFG)


# ------------------------------------------------------------------------------------------------- the plugin's PDL path
PDL_MAX_M = int(os.environ.get("LOWBIT_PDL_MAX_M", "32"))     # the dispatch times the PDL path for buckets M <= this
# SERVE-H100-PDL-STOCK (opt-in, off by default): with bf16 adapters, every bucket above this M runs vLLM's own LoRA kernels after
# the vendor GEMM ("hybrids") instead of the timed choice; the decode buckets keep the dispatch's choice (the PDL path)
STOCK_ABOVE = int(os.environ.get("LOWBIT_STOCK_ABOVE", "0"))
# opt-in, off by default (2026-10-26; mock review R3 v42, H100 default capture): buckets above this M never pick "hybrids" (vLLM's
# LoRA kernels inside our op, whose eager host time is ours plus vLLM's: about 175 against 61 us per call on the H100 with direct
# launches, results/host_overhead_h100.jsonl); the dispatch picks among our own paths there
NOHYB_ABOVE = int(os.environ.get("LOWBIT_NOHYB_ABOVE", "0"))
# opt-in, off by default (2026-10-26; mock review R1 v44): every bucket up to PDL_MAX_M takes the decode path whenever it is a candidate,
# instead of the timed choice (on the 72B the per-projection timing sent a third of the projections at one token to vLLM's kernels)
PDL_FORCE = os.environ.get("LOWBIT_PDL_FORCE", "0") not in ("0", "", "false", "False")
# EAGER-FAST (2026-10-26; opt-in LOWBIT_EAGER_ABOVE, 0 = off, every earlier run unchanged): buckets above vLLM's CUDA-graph capture bound
# run eagerly, where a path's host time adds to its GPU time; above this many tokens the dispatch times its candidates in eager calls
# (wall time per call with the GPU fed back to back) instead of in graph replays. The profile of a four-tenant 2,048-token step on
# the H100 found the graph-timed dispatch serving most projections with vLLM's own LoRA kernels (hybrids) there, whose launches add
# their host time to the operator's.
EAGER_ABOVE = int(os.environ.get("LOWBIT_EAGER_ABOVE", "0"))


def _eager_us(fn, reps: int = 20) -> float:
    """Microseconds per eager call of fn, calls issued back to back (host and GPU time overlap across calls as in an eager step)."""
    import time
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1e6 * (time.perf_counter() - t) / reps
_TABLE: dict = {}      # (bucket, N, K, S, R, fmt) -> the swept best configuration (results/pdl_k2.jsonl, one adapter)
_TABLE_SRC: dict = {}  # where the table came from (path, job, entries), for the install line and the report
DECODE_BUCKETS = tuple(b for b in F._BUCKETS if b <= 32)


def load_table(path: str | None = None, job: str | None = None) -> int:
    """The plugin path's configurations: every one-adapter row of bench/pdl_k2_bench.py's output with a swept best
    (pdl_best), of the newest job unless given, keyed by (bucket of M, N, K, slices, rank, adapter format). The path:
    the argument, else LOWBIT_PDL_CFG, else results/pdl_k2.jsonl beside the package (a job's code snapshot has none:
    the pod script sets LOWBIT_PDL_CFG). A missing or empty file leaves the table empty (defaults; plugin_cfg)."""
    import json
    from pathlib import Path
    f = Path(path or os.environ.get("LOWBIT_PDL_CFG") or Path(__file__).resolve().parents[1] / "results/pdl_k2.jsonl")
    _TABLE.clear()
    rows = []
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("pdl_best") and r.get("n") == 1 and "N" in r and "K" in r and "S" in r and "r" in r:
                    rows.append(r)
    if rows:
        job = job or max(r.get("job", "") for r in rows)
        for r in rows:
            if r.get("job", "") == job:
                _TABLE[(F._bucket(r["M"]), r["N"], r["K"], r["S"], r["r"], r.get("fmt", "bf16"))] = dict(r["pdl_best"])
    _TABLE_SRC.update(path=str(f), job=job if rows else None, entries=len(_TABLE))
    return len(_TABLE)


def _retile(c: dict, b: int) -> dict:
    """A configuration found for another batch bucket, re-tiled for bucket b: 16-token tiles up to 16 tokens, 32 above;
    8 warps from 4096-element output tiles (as fp8_lora_pdl.default_cfg)."""
    c = dict(c)
    bm = 16 if b <= 16 else 32 if b <= 32 else 64
    if c.get("BLOCK_M", bm) != bm:
        c["BLOCK_M"] = bm
        c["num_warps"] = 8 if bm * c.get("BLOCK_N", 64) >= 4096 else 4
    return c


def plugin_cfg(op, M: int) -> dict:
    """The PDL configuration for this op's shape at M tokens, robust to what the sweep did not measure: the exact
    (bucket, shape, rank, format) entry; else the same bucket and shape at the nearest rank (bf16 first); else the same
    shape at the nearest measured decode bucket, re-tiled; else fp8_lora_pdl's defaults. Cached per (op, bucket)."""
    import math
    b = F._bucket(M)
    cache = op.__dict__.setdefault("_pdl_cfgs", {})
    if b in cache:
        return cache[b]
    shape = (op.N, op.K, op.S)
    fmt = _fmt(op)
    c, how = None, "default"
    if (b, *shape, op.R, fmt) in _TABLE:
        c, how = _TABLE[(b, *shape, op.R, fmt)], "exact"
    if c is None:   # the same bucket and shape, the nearest rank: this format, then bf16, then any
        for want in (fmt, "bf16", None):
            ks = [k for k in _TABLE if k[0] == b and k[1:4] == shape and (want is None or k[5] == want)]
            if ks:
                k = min(ks, key=lambda k: (abs(math.log2(k[4]) - math.log2(max(op.R, 1))), -k[4]))
                c, how = _TABLE[k], f"rank {k[4]} {k[5]}"
                break
    if c is None:   # the same shape at the nearest measured decode bucket (log distance; the larger on ties)
        ks = [k for k in _TABLE if k[1:4] == shape and k[0] <= max(DECODE_BUCKETS)]
        if ks:
            pref = [k for k in ks if k[4] == op.R and k[5] == fmt] or [k for k in ks if k[5] == fmt] or ks
            k = min(pref, key=lambda k: (abs(math.log2(k[0]) - math.log2(b)), -k[0], abs(k[4] - op.R)))
            c, how = _retile(_TABLE[k], b), f"bucket {k[0]} rank {k[4]} {k[5]}"
    cfg = dict(c) if c is not None else {}
    cache[b] = cfg
    op.__dict__.setdefault("_pdl_cfg_src", {})[b] = how
    return cfg


def _report(line: str) -> None:
    rep = os.environ.get("LOWBIT_REPORT")
    if rep:
        try:
            with open(rep, "a") as fh:
                fh.write(f"{os.getpid()} {line}\n")
        except OSError:
            pass


def install() -> None:
    """vLLM general plugin hook, opt-in with LOWBIT_PDL=1 (entry point lowbit_pdl in pylib_pdl/, on PYTHONPATH only for
    engines that opt in; as pylib_swap and hopper_swap for LOWBIT_SWAP): on compute capability 9 the fused op's timed
    dispatch also times the PDL path for every decode bucket (M <= LOWBIT_PDL_MAX_M, 32) against its other candidates
    (fused, hybrid, hybridq, and vLLM's own LoRA kernels, "hybrids", tuned when VLLM_TUNED_CONFIG_FOLDER is set) and
    serves it where it is fastest, through the plugin's registered custom op (torch.ops.vllm.lowbit_fused_fp8_lora:
    opaque to torch.compile, captured in vLLM's CUDA graphs with the PDL edge). Configurations: plugin_cfg over
    results/pdl_k2.jsonl (LOWBIT_PDL_CFG, LOWBIT_PDL_JOB). The PDL kernels are compiled, and their split-K workspace
    sized, for every decode bucket during the plugin's warm-up, before vLLM captures its graphs. The registered plugin
    module is patched in memory only; with LOWBIT_PDL unset this does nothing. Chains with hopper_swap.install.
    LOWBIT_REPORT gets, per layer, "pdl_dispatch=<bucket>:<path>,..." (the plugin's own line maps the PDL path to "f")."""
    if os.environ.get("LOWBIT_PDL", "0") != "1" or getattr(F.VllmFusedFP8LoRA, "_pdl_installed", False):
        return
    V = F.VllmFusedFP8LoRA
    n_cfg = load_table(None, os.environ.get("LOWBIT_PDL_JOB") or None)
    orig_run, orig_call, orig_warm = V._run, V.__call__, V.warmup
    n_ops = [0]

    def _run(self, x, meta_args, out=None, mode=None, cfg_s=None, cfg_g=None, hybrid=False, split=None):
        if hybrid == "p":
            return run_pdl(self, x, meta_args, cfg=plugin_cfg(self, x.shape[0]), out=out)
        return orig_run(self, x, meta_args, out=out, mode=mode, cfg_s=cfg_s, cfg_g=cfg_g, hybrid=hybrid, split=split)

    def __call__(self, x, meta_args, out=None, mode=None, cfg_s=None, cfg_g=None):
        if self.base_fn is not None and self.dispatch.get(F._bucket(x.shape[0])) == "pdl":
            return run_pdl(self, x, meta_args, cfg=plugin_cfg(self, x.shape[0]), out=out)
        return orig_call(self, x, meta_args, out=out, mode=mode, cfg_s=cfg_s, cfg_g=cfg_g)

    def tune_dispatch(self) -> dict:
        """VllmFusedFP8LoRA.tune_dispatch with the PDL path as one more candidate for the decode buckets on compute
        capability 9 (and the swap path when hopper_swap is installed); the dispatch key is marked "pdl", so tables never
        mix with the unpatched op's. Same timing as the plugin's (F._graph_us, CUDA-graph replays, LOWBIT_DISPATCH_K)."""
        if self.base_fn is None:
            return self.dispatch
        kd = int(os.environ.get("LOWBIT_DISPATCH_K", "1"))
        swap = getattr(V, "_swap_installed", False)
        key = (torch.cuda.get_device_name(self.w_q.device), self.N, self.K, self.R, tuple(self.widths), _fmt(self),
               self.base_q_fn is not None, kd, "pdl", PDL_MAX_M, swap, STOCK_ABOVE, EAGER_ABOVE, NOHYB_ABOVE, PDL_FORCE)
        tune_max = int(os.environ.get("LOWBIT_TUNE_MAX_M", "2048"))
        if key not in F._DISPATCH:
            import json
            table, prev, times = {}, None, {}
            for b in F._BUCKETS:
                M = min(b, self.max_m)
                if M < 1 or (table and M == prev):
                    break
                if table and M > tune_max:
                    table[b] = table[max(table)]
                    prev = M
                    continue
                prev = M
                x = torch.randn(M, self.K, device=self.w_q.device).to(torch.bfloat16)
                metas = [self._dummy_meta(M, 0)] + ([self._dummy_meta_multi(M, kd)] if min(kd, self.L, M) > 1 else [])
                timer = _eager_us if (EAGER_ABOVE and b > EAGER_ABOVE) else F._graph_us
                tm = lambda h: sum(timer(lambda m=m: self._run(x, m, hybrid=h)) for m in metas)   # noqa: E731
                t = {"fused": tm(False), "hybrid": tm(True)}
                if self.base_q_fn is not None:
                    t["hybridq"] = tm("q")
                if not (self.lora_fp8 or self.lora_fp4):
                    t["hybrids"] = tm("s")
                if self.cc_major == 9:
                    cands = ((("pdl", "p"),) if M <= PDL_MAX_M else ()) + ((("swap", "w"),) if swap else ())
                    for name, h in cands:
                        try:
                            t[name] = tm(h)
                        except Exception as e:   # noqa: BLE001 -- not a candidate for this shape and bucket
                            torch.cuda.synchronize()
                            work(self.w_q.device).reset()
                            _report(f"pdl-candidate-failed N={self.N} K={self.K} M={M} {name}: {type(e).__name__}: "
                                    f"{str(e)[:200]}")
                pick = {k: v for k, v in t.items() if not (NOHYB_ABOVE and b > NOHYB_ABOVE and k == "hybrids")}
                table[b] = min(pick, key=pick.get)
                if PDL_FORCE and M <= PDL_MAX_M and "pdl" in t:
                    table[b] = "pdl"
                if STOCK_ABOVE and b > STOCK_ABOVE and "hybrids" in t:   # opt-in: vLLM's own LoRA kernels above the decode buckets
                    table[b] = "hybrids"
                times[b] = {k: round(v, 2) for k, v in t.items()}
                del x, metas
            F._DISPATCH[key] = table
            torch.cuda.empty_cache()
            _report(f"pdl-dispatch-table N={self.N} K={self.K} R={self.R} widths={list(self.widths)} fmt={_fmt(self)} "
                    f"cfg_src={json.dumps({b: self.__dict__.get('_pdl_cfg_src', {}).get(b) for b in DECODE_BUCKETS})} "
                    f"us={json.dumps(times)}")
        self.dispatch = F._DISPATCH[key]
        return self.dispatch

    def warmup(self) -> None:
        orig_warm(self)                                   # times the dispatch (patched) and compiles the other paths
        n_ops[0] += 1
        if self.cc_major == 9 and self.base_fn is not None:
            # the PDL kernels of every bucket that chose them compiled (the dispatch timed them already: a safety net), and
            # the shared split-K workspace grown to each such bucket's largest M, here: vLLM captures its CUDA graphs
            # after the warm-up, and nothing may grow inside them
            for b in DECODE_BUCKETS:
                M = min(b, self.max_m)
                if M < 1 or b > PDL_MAX_M or self.dispatch.get(b) != "pdl":
                    continue
                x = torch.zeros(M, self.K, dtype=torch.bfloat16, device=self.w_q.device)
                for lora in (0, -1):
                    try:
                        self._run(x, self._dummy_meta(M, lora), hybrid="p")
                    except Exception as e:   # noqa: BLE001
                        _report(f"pdl-warmup-failed N={self.N} K={self.K} M={M}: {type(e).__name__}: {str(e)[:200]}")
                        break
            torch.cuda.synchronize()
        d = self.dispatch or {}
        _report(f"pdl-layer {n_ops[0]} N={self.N} K={self.K} R={self.R} "
                f"pdl_dispatch={','.join(f'{b}:{v}' for b, v in sorted(d.items()))}")

    V._run, V.__call__, V.tune_dispatch, V.warmup = _run, __call__, tune_dispatch, warmup
    V._pdl_installed = True
    print(f"[lowbit] PDL decode path installed for Hopper dispatch (decode buckets M <= {PDL_MAX_M}; {n_cfg} swept "
          f"configurations from {_TABLE_SRC.get('path')}, job {_TABLE_SRC.get('job')})", flush=True)
