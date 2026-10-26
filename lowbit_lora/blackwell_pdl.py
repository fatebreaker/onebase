"""The NVFP4 decode path with the adapters off the critical path (kernels/nvfp4_lora_pdl.py) as a path of the plugin's
NVFP4 op on Blackwell sm_120, without touching the registered plugin code: run_pdl4(op, x, meta_args) launches the shrink
kernel (K1's adapter segments; with X_MODE 1 also the NVFP4 quantization programs) and then the block-scaled FP4 GEMM with
programmatic dependent launch, which streams its weights while the shrink runs and waits for it only before its epilogue.
op is a VllmFusedNVFP4LoRA (nvfp4_lora_vllm.py: w_q [N, K/2] codes, w_sf [N, K/16] linear-layout E4M3 block scales,
x_global, alpha, store (AdapterStore: bf16 / FP8 / NVFP4 adapters), scratch with xq / sxq / hp). The kernel benchmark first
(bench/pdl_fp4_bench.py); a plugin path only through an opt-in patch registered before its data.
    from lowbit_lora.blackwell_pdl import run_pdl4; y = run_pdl4(op, x, meta.meta_args(M, False))
"""
from __future__ import annotations

import os

import torch
import triton

from lowbit_lora.kernels.nvfp4_lora_pdl import _gemm_pdl4, _shrink_pdl4, default_cfg4, warps4, work

CFG4: dict = {}   # (bucket of M, N, K, n_slices, R, adapter format) -> swept config (bench/pdl_fp4_bench.py)
_NSM: dict = {}


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def _n_sm(dev) -> int:
    if dev.type != "cuda":
        return 94
    k = str(dev)
    if k not in _NSM:
        _NSM[k] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NSM[k]


def _bucket(M: int) -> int:
    from lowbit_lora.kernels.fp8_lora_vllm import _bucket as b
    return b(M)


def resolve_cfg4(op, M: int, cfg: dict | None = None) -> dict:
    """fp8-style precedence: nvfp4_lora_pdl.default_cfg4, then the swept table, then cfg (later keys win)."""
    fmt = op.lora_fmt
    c = default_cfg4(M, op.N, op.K, op.widths, op.R, _n_sm(op.w_q.device), fmt == "nvfp4")
    c.update(CFG4.get((_bucket(M), op.N, op.K, op.S, op.R, fmt), {}))
    if cfg:
        c.update(cfg)
    if not (cfg and "num_warps" in cfg):   # warps follow the final tile (spills otherwise: nvfp4_lora_pdl.warps4)
        c["num_warps"] = warps4(int(c["BLOCK_M"]), int(c["BLOCK_N"]), max(16, triton.next_power_of_2(op.R)))
    return c


def _launch(fn, grid, args, kw, compile_only: bool):
    if compile_only:
        return None if _interpret() else fn.warmup(*args, grid=grid, **kw)
    return fn[grid](*args, **kw)


def run_pdl4(op, x, meta_args, cfg: dict | None = None, out=None, pdl: bool = True, lora: bool = True,
             shrink: bool = True, gemm: bool = True, xqd=None, compile_only: bool = False):
    """y = alpha * (x's NVFP4 codes, vLLM's recipe) x W (block-scaled FP4 MMA) + every adapter's expansion.
    pdl=False: the same kernels without the PDL attribute. lora=False: the GEMM without LoRA (with X_MODE 1 the
    quantization programs still run first). shrink=False: no adapter shrink (h from the last one; the quantization
    programs still run with X_MODE 1). gemm=False: the shrink kernel alone. xqd: (codes [M, K/2] uint8, scales [M, K/16]
    float8_e4m3fn) buffers that receive the codes and scales the GEMM computed itself (X_MODE 0; bit-identity checks)."""
    tok, srt, seg_n, seg_start, seg_ids = meta_args[:5]
    M, K = x.shape
    N, S, R, L1 = op.N, op.S, op.R, op.L + 1
    interp = _interpret()
    c = resolve_cfg4(op, M, cfg)
    sc = op.scratch
    hp = sc.hp
    xq = sc.xq[: M * (K // 2)].view(M, K // 2)
    sxq = sc.sxq[: M * (K // 16)].view(M, K // 16)
    st = op.store
    fmt = op.lora_fmt
    split1 = int(c["SPLIT_K1"])
    while split1 > 1 and split1 * S * M * R > hp.numel():
        split1 //= 2
    bsr = min(64, R)
    n_sr = S * R // bsr
    s_bm = 16
    nx = triton.cdiv(M, s_bm) * n_sr * split1 if lora else 0
    xm = int(c["X_MODE"])
    qbm, qbk = int(c.get("QBM", 16)), int(c.get("QBK", 256))
    qsplit = max(1, min(16, K // (2 * qbk)))
    n_q = triton.cdiv(M, qbm) * qsplit
    gx = max(nx, n_q) if xm == 1 else max(nx, 1)
    bm, bn, bk, sk = int(c["BLOCK_M"]), int(c["BLOCK_N"]), int(c["BLOCK_K"]), int(c["SPLIT_K"])
    assert all(w % bn == 0 for w in op.widths), (bn, op.widths)
    k_split = triton.cdiv(triton.cdiv(K, sk), bk) * bk
    n_tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    br = max(16, triton.next_power_of_2(R))
    rck = int(c.get("R_CHUNK") or (64 if (fmt == "nvfp4" or bn * br > 8192) else br))
    rck = min(rck, br)
    bp = int(c.get("B_PREFETCH", 2))
    if rck < br:
        bp = min(bp, 1)
    bpipe = bool(c.get("B_PIPE", bn * rck <= 4096 and fmt != "nvfp4"))   # (NVFP4-stored B: one tile, else spills)
    wk = work(x.device)
    wk.ensure(sk * M * N if sk > 1 else 1, n_tiles)
    y = out if out is not None else torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    dot_f32 = bool(getattr(op, "dot_f32", False))
    qseg = xm == 1
    adapters = lora and shrink
    launched = False
    if adapters or qseg:
        nz = (L1 if adapters else 0) + (1 if qseg else 0)
        _launch(_shrink_pdl4, (gx, 1, nz), (
            x, st.a_ptrs, st.sa_ptrs, st.ag, hp, xq, sxq, op.x_global, wk.sync, srt, seg_ids, seg_n, seg_start,
            M, K, R, S, L1, nx, x.stride(0), op.stride_al, op.stride_am, xq.stride(0), sxq.stride(0)),
            dict(SPLIT_K=split1, N_SR=n_sr, BLOCK_M=s_bm, BLOCK_SR=bsr, BLOCK_K=int(c.get("S_BLOCK_K", 128)),
                 QSEG=qseg, QSIGNAL=qseg and gemm, QBM=qbm, QBK=qbk, QSPLIT=qsplit, PTX=not interp,
                 USE_GDC=pdl and gemm and not interp, DOT_F32=dot_f32, A_FP8=fmt == "fp8", A_FP4=fmt == "nvfp4",
                 num_warps=4, num_stages=int(c.get("S_STAGES", 2))), compile_only)
        launched = True
    if not gemm:
        return None
    kw = dict(num_warps=int(c.get("num_warps", 4)), num_stages=int(c.get("num_stages", 4)))
    if pdl and launched and not interp:
        kw["launch_pdl"] = True
    xd, sd = xqd if xqd is not None else (xq, sxq)
    _launch(_gemm_pdl4, (n_tiles, sk), (
        x, xq, sxq, op.x_global, op.w_q, op.w_sf, op.alpha, hp, st.b_ptrs, st.sb_ptrs, st.bg, y, wk.p, wk.cnt, wk.sync,
        tok, seg_ids, seg_n, xd, sd, M, N, K, R, S, L1, op.end0, op.end1, op.end2, k_split, gx,
        x.stride(0), xq.stride(0), sxq.stride(0), y.stride(0)),
        dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_N=int(c.get("GROUP_N", 8)), SPLIT_K=sk,
             EVEN_K=K % bk == 0, SWAP=bool(c.get("SWAP", True)), X_MODE=xm, HAS_LORA=lora, SPLIT_K1=split1,
             BLOCK_R=br, R_CHUNK=rck, LP=triton.next_power_of_2(L1), B_PREFETCH=bp, B_PIPE=bpipe,
             W_PREFETCH=int(c.get("W_PREFETCH", 0)), SYNC_MODE=0 if launched else 1, PTX=not interp,
             B_FP8=fmt == "fp8", B_FP4=fmt == "nvfp4", DOT_F32=dot_f32, DEBUG_XQ=xqd is not None, GPU=not interp,
             **kw), compile_only)
    return y
