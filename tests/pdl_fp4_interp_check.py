"""CPU checks of the NVFP4 decode path (kernels/nvfp4_lora_pdl.py via lowbit_lora/blackwell_pdl.run_pdl4), no GPU needed.

Interpreter (default; driven by vLLM's own LoRAKernelMeta on the CPU, the NVFP4 op of nvfp4_lora_vllm.py with bf16, FP8-
and NVFP4-stored adapters): both MMA orientations (SWAP: weight rows on the 16-row side, 8- or 16-token tiles; the
plugin's: tokens on it), both activation modes (X_MODE 0: the GEMM quantizes x itself; 1: quantization programs and a
release/acquire counter), split-K 1 and 3 (the last-arriving CTA's fixed-order reduction), ranks 16 to 128, one to four
slices; cases: one token, single-token segments, empty segments, tokens without an adapter, a no-adapter batch after an
adapter batch (stale token map), CUDA-graph padding. Each output against
  * the fp32 reference (nvfp4_lora_vllm.reference: vLLM's recipe with exact reciprocals) - relative error at most 1e-2;
  * the existing NVFP4 K2 (the plugin's fused path, K1 + _gemm_nvfp4, on the same op) - relative error at most 1e-2, and
    the share of bitwise-identical bf16 outputs reported;
and: the codes and block scales the GEMM computes itself (X_MODE 0, debug output) and the quantization programs write
(X_MODE 1) byte-identical to K1's recipe (nvfp4_quant_torch); repeated launches bitwise identical; every counter back at
zero; the GEMM without the shrink launch (h from the previous shrink) identical to the PDL path; without LoRA within 1e-2
of the base reference and bitwise equal to the exact (fp64) block-scaled product of K1's codes stored as the interpreter
stores bf16 (truncating; the GPU rounds to nearest; the interpreter's errors against the fp32 references are that). (The interpreter has no dot_scaled and no inline PTX: DOT_F32 and PTX=False, as for the existing
NVFP4 kernels; tests on sm_120 check those: bench/pdl_fp4_bench.py.)
    TRITON_INTERPRET=1 python tests/pdl_fp4_interp_check.py [--quick]

--compile (no interpreter): offline compile for sm_120 of the shrink and the GEMM at the bench's decode tiles, with the
JIT's divisibility specialization, checking the PTX for griddepcontrol.launch_dependents (shrink), griddepcontrol.wait
(GEMM), the native block-scaled FP4 MMA (kind::mxf4nvf4.block_scale.scale_vec::4X), cp.async pipelining, the
release / acquire atomics, the cubin's registers and spills, and shared memory within sm_120's 99 KB per block.
    python tests/pdl_fp4_interp_check.py --compile
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))


def interp_main(quick: bool) -> int:
    assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1 (or --compile)"
    import torch
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    from lowbit_lora.blackwell_pdl import run_pdl4
    from lowbit_lora.kernels import nvfp4_lora_vllm as F
    from lowbit_lora.kernels.nvfp4_lora_pdl import work

    torch.manual_seed(0)
    worst, worst_k2, bad = 0.0, 0.0, []
    same_bits, exact = [], []
    wk = work(torch.device("cpu"))
    CS = {"BLOCK_M": 16, "BLOCK_SR": 16, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}

    def check(name, ok, msg=""):
        if not ok:
            bad.append(f"{name}: {msg}")
            print("  FAIL", name, msg, flush=True)

    shapes = [(256, (64, 32, 32), 16, 4, 2), (192, (64,), 64, 3, 3), (256, (64, 64), 32, 8, 2), (128, (64,), 128, 2, 1),
              (128, (32, 32, 32, 32), 16, 3, 2)]
    if quick:
        shapes = shapes[:2]
    for K, widths, R, L, split1 in shapes:
        N, S = sum(widths), len(widths)
        w = torch.randn(N, K) * 0.05
        w_global = F.global_scale_for(w)
        w_q, w_sf = F.nvfp4_quant_torch(w, w_global)
        a_st = tuple((torch.randn(L, 1, R, K) / K ** 0.5).to(torch.bfloat16) for _ in range(S))
        b_st = tuple((torch.randn(L, 1, n, R) * 0.05).to(torch.bfloat16) for n in widths)
        A = torch.cat([t[:, 0] for t in a_st], dim=1)
        B = torch.cat([t[:, 0] for t in b_st], dim=1)
        ops = {}
        for fmt in ("bf16", "fp8", "nvfp4"):
            o = F.VllmFusedNVFP4LoRA(w_q, w_sf, w_global, 1.0, a_st, b_st, max_m=96, split_k=split1, lora_fmt=fmt)
            o.dot_f32 = True
            ops[fmt] = (o, A, B) if fmt == "bf16" else (o, *o.store.dequantized())
        meta = LoRAKernelMeta.make(L, 96, device="cpu")
        bn = min(64, min(widths))
        cases = [(1, "one"), (min(4, L), "single-token segs"), (21, "mixed"), (6, "two of L"), (13, "none"),
                 (24, "padded")]
        if quick:
            cases = cases[:3] + cases[4:5]
        for M, kind in cases:
            x = torch.randn(M, K).to(torch.bfloat16)
            x[:, :3] *= 8                                                    # outlier channels
            x_global = F.global_scale_for(x)
            for o, _, _ in ops.values():
                o.x_global.fill_(x_global)
                o.alpha.fill_(1.0 / (x_global * w_global))
            if kind == "one":
                tok = torch.full((M,), L - 1, dtype=torch.int32)
            elif kind == "single-token segs":
                tok = torch.randperm(L)[:M].to(torch.int32)
            elif kind == "mixed":
                tok = torch.randint(-1, L, (M,), dtype=torch.int32)
            elif kind == "two of L":
                tok = torch.tensor([1, 1, 0, 1, 0, 0][:M], dtype=torch.int32) % L
            else:
                tok = torch.full((M,), -1, dtype=torch.int32)
            if kind == "padded":
                tok = torch.randint(-1, L, (M,), dtype=torch.int32)
                tok[0] = 0
                meta.token_indices_sorted_by_lora_ids[M - 5:M] = 10 ** 6
                meta.token_lora_mapping[M - 5:M] = L - 1
                meta.prepare_tensors(tok[: M - 5])
                tok[M - 5:] = -1
            else:
                if kind == "none":
                    meta.prepare_tensors(torch.zeros(M, dtype=torch.int32))
                meta.prepare_tensors(tok)
            margs = meta.meta_args(M, False)
            q_ref, sf_ref = F.nvfp4_quant_torch(x, x_global)
            for fmt, (op, Ar, Br) in ops.items():
                ref = F.reference(x, tok, w_q, w_sf, w_global, x_global, Ar, Br, widths)
                ref0 = F.reference(x, torch.full((M,), -1), w_q, w_sf, w_global, x_global, Ar, Br, widths)
                CG = {"BLOCK_M": 16, "BLOCK_N": bn, "BLOCK_K": 64, "GROUP_M": 2, "num_warps": 4, "num_stages": 1}
                y_k2 = op(x, margs, cfg_s=CS, cfg_g=CG, hybrid_m=1 << 30, use_dispatch=False).float().clone()
                grid = [(xm, swap, bm, sk) for xm in (0, 1) for swap, bm in ((True, 8), (True, 16), (False, 16))
                        for sk in (1, 3)] if fmt == "bf16" else [(1, True, 16, 2), (0, False, 16, 1)]
                if quick and fmt == "bf16":
                    grid = [g for g in grid if g[3] == 3 or g[2] == 8]
                for xm, swap, bm, sk in grid:
                    cfg = {"X_MODE": xm, "SWAP": swap, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 64, "SPLIT_K": sk,
                           "num_stages": 1, "SPLIT_K1": split1, "S_BLOCK_K": 32, "S_STAGES": 1, "QBM": 16, "QBK": 64}
                    tag = f"K={K} w={widths} R={R} L={L} M={M} {kind} {fmt} X_MODE={xm} swap={swap} BM={bm} split-K={sk}"
                    xqd = (torch.zeros(M, K // 2, dtype=torch.uint8), torch.zeros(M, K // 16, dtype=torch.float8_e4m3fn)) \
                        if xm == 0 else None
                    y = run_pdl4(op, x, margs, cfg=cfg, xqd=xqd).float()
                    err = float((y - ref).norm() / ref.norm())
                    ek2 = float((y - y_k2).norm() / y_k2.norm())
                    worst, worst_k2 = max(worst, err), max(worst_k2, ek2)
                    same_bits.append(float((y.to(torch.bfloat16).view(torch.int16)
                                            == y_k2.to(torch.bfloat16).view(torch.int16)).float().mean()))
                    check(tag + " vs reference", err < 1e-2, f"{err:.2e}")
                    check(tag + " vs the existing K2", ek2 < 1e-2, f"{ek2:.2e}")
                    if xm == 0:
                        check(tag + " codes", torch.equal(xqd[0], q_ref), f"{int((xqd[0] != q_ref).sum())} differ")
                        check(tag + " scales", torch.equal(xqd[1].view(torch.uint8), sf_ref.view(torch.uint8)),
                              f"{int((xqd[1].view(torch.uint8) != sf_ref.view(torch.uint8)).sum())} differ")
                    else:
                        xq_k = op.scratch.xq[: M * (K // 2)].view(M, K // 2)
                        sx_k = op.scratch.sxq[: M * (K // 16)].view(M, K // 16)
                        check(tag + " codes", torch.equal(xq_k, q_ref), f"{int((xq_k != q_ref).sum())} differ")
                        check(tag + " scales", torch.equal(sx_k.view(torch.uint8), sf_ref.view(torch.uint8)),
                              "scales differ")
                    check(tag + " counters", int(wk.cnt.abs().sum()) == 0 and int(wk.sync.abs().sum()) == 0,
                          f"cnt {wk.cnt.tolist()} sync {wk.sync.tolist()}")
                    y2 = run_pdl4(op, x, margs, cfg=cfg).float()
                    check(tag + " repeat", torch.equal(y, y2), "a second launch differs")
                    if fmt == "bf16" and (sk == 3 or bm == 8):
                        y3 = run_pdl4(op, x, margs, cfg=cfg, shrink=False).float()
                        check(tag + " no-shrink", torch.equal(y, y3), "differs from the PDL path")
                    if fmt == "bf16" and kind in ("one", "mixed") and sk == 1:
                        y0 = run_pdl4(op, x, margs, cfg=cfg, lora=False).float()
                        e0 = float((y0 - ref0).norm() / ref0.norm())
                        worst = max(worst, e0)
                        check(tag + " no-LoRA", e0 < 1e-2, f"{e0:.2e}")
                        # the block-scaled product itself: exact (fp64) on K1's codes, then the interpreter's bf16
                        # store, which truncates (the GPU's rounds to nearest)
                        ex = (F.nvfp4_dequant_torch(q_ref, sf_ref, 1.0).double()
                              @ F.nvfp4_dequant_torch(w_q, w_sf, 1.0).double().t()) * float(op.alpha)
                        rtz = (ex.float().view(torch.int32) & ~0xFFFF).view(torch.float32)
                        same = float((y0 == rtz).float().mean())
                        exact.append(same)
                        check(tag + " no-LoRA exact", same >= 0.999, f"{100 * same:.2f}% bitwise")
            print(f"K={K} widths={widths} R={R} L={L} M={M:2d} {kind:17s}: worst vs reference {worst:.2e}, vs the "
                  f"existing K2 {worst_k2:.2e}", flush=True)
    share = sum(same_bits) / max(1, len(same_bits))
    print(("PASS" if not bad else f"FAIL ({len(bad)})") + f"  worst rel err {worst:.2e} (reference), {worst_k2:.2e} "
          f"(existing K2); outputs bitwise equal to the existing K2's {100 * share:.1f}%; without LoRA, bitwise equal to "
          f"the exact block-scaled product (bf16, truncated as the interpreter stores) in at least "
          f"{100 * min(exact or [0]):.2f}% of outputs per case")
    for b in bad[:20]:
        print("  ", b)
    return 1 if bad else 0


PTR4 = {"X": "*bf16", "XQ": "*u8", "SXQ": "*fp8e4nv", "GS": "*fp32", "W": "*u8", "SW": "*fp8e4nv", "ALPHA": "*fp32",
        "HP": "*fp32", "A_PTRS": "*i64", "SA_PTRS": "*i64", "AG": "*fp32", "B_PTRS": "*i64", "SB_PTRS": "*i64",
        "BG": "*fp32", "Y": "*bf16", "P": "*fp32", "CNT": "*i32", "SYNC": "*i32", "TOK": "*i32", "SORTED": "*i32",
        "SEG_IDS": "*i32", "SEG_N": "*i32", "SEG_START": "*i32", "XQD": "*u8", "SXD": "*fp8e4nv"}


def compile_main() -> int:
    assert os.environ.get("TRITON_INTERPRET", "0") != "1", "--compile runs without the interpreter"
    import re

    import triton

    import pdl_k2_interp_check as T
    from lowbit_lora.kernels import nvfp4_lora_pdl as P4
    T.DIV16 |= {"stride_sxm"}
    SMEM = 101376   # sm_120's shared memory per block (99 KB): a kernel above it fails to launch
    bad = 0

    def ok(name, cond, extra=""):
        nonlocal bad
        bad += not cond
        print(("  ok   " if cond else "  FAIL ") + name + (f"  {extra}" if extra else ""), flush=True)

    print("== shrink (adapter segments; + NVFP4 quantization programs), sm_120")
    for qseg, fmt in ((False, "bf16"), (True, "bf16"), (True, "fp8"), (True, "nvfp4")):
        k = T.build(P4._shrink_pdl4, dict(SPLIT_K=4, N_SR=3, BLOCK_M=16, BLOCK_SR=64, BLOCK_K=128, QSEG=qseg,
                                          QSIGNAL=qseg, QBM=16, QBK=256, QSPLIT=8, PTX=True, USE_GDC=True,
                                          A_FP8=fmt == "fp8", A_FP4=fmt == "nvfp4"),
                    {"num_warps": 4, "num_stages": 2}, cc=120, ptr=PTR4)
        ptx = k.asm["ptx"]
        ok(f"shrink qseg={qseg} {fmt}", ptx.count("griddepcontrol.launch_dependents") == 1
           and (not qseg or re.search(T.RELEASE, ptx) is not None) and k.metadata.shared <= SMEM,
           f"launch_dependents {ptx.count('griddepcontrol.launch_dependents')}, cp.async "
           f"{ptx.count('cp.async.cg') + ptx.count('cp.async.ca')}, smem {k.metadata.shared}, {T.usage(k)}")
    print("== GEMM (K2), sm_120, launch_pdl")
    grid = []
    for xm in (0, 1):
        for swap, bm in ((True, 8), (True, 16), (False, 16)):
            for bn, sk, st in ((64, 1, 4), (64, 2, 4), (128, 1, 3)):
                grid.append(dict(xm=xm, swap=swap, bm=bm, bn=bn, sk=sk, st=st, nw=8 if bn >= 128 else 4))
    grid += [dict(xm=1, swap=True, bm=16, bn=64, sk=2, st=4, br=128, rck=64, bp=1, nw=8),
             dict(xm=1, swap=True, bm=16, bn=64, sk=2, st=4, br=16), dict(xm=1, swap=True, bm=8, bn=64, sk=2, st=4, fmt="fp8"),
             dict(xm=1, swap=True, bm=8, bn=64, sk=2, st=4, fmt="nvfp4"), dict(xm=0, swap=True, bm=8, bn=64, sk=1, st=4,
                                                                               lora=False, sync=1, bp=0, wpf=0),
             dict(xm=1, swap=False, bm=32, bn=64, sk=1, st=4), dict(xm=1, swap=False, bm=64, bn=64, sk=1, st=3, nw=8),
             dict(xm=1, swap=False, bm=64, bn=128, sk=1, st=3, nw=8), dict(xm=0, swap=True, bm=8, bn=64, sk=2, st=5, bk=128)]
    for g in grid:
        st = g.pop("st")
        nw = g.pop("nw", 4)
        fmt = g.pop("fmt", "bf16")
        br = g.pop("br", 64)
        rck = g.pop("rck", None) or br
        bk = g.pop("bk", 256)
        kc = dict(BLOCK_M=g["bm"], BLOCK_N=g["bn"], BLOCK_K=bk, GROUP_N=8, SPLIT_K=g["sk"], EVEN_K=True,
                  SWAP=g["swap"], X_MODE=g["xm"], HAS_LORA=g.get("lora", True), SPLIT_K1=4, BLOCK_R=br, R_CHUNK=rck,
                  LP=triton.next_power_of_2(9), B_PREFETCH=g.get("bp", 2) if rck == br else min(g.get("bp", 2), 1),
                  B_PIPE=g["bn"] * rck <= 4096 and fmt != "nvfp4", W_PREFETCH=g.get("wpf", 2),
                  SYNC_MODE=g.get("sync", 0), PTX=True,
                  B_FP8=fmt == "fp8", B_FP4=fmt == "nvfp4")
        name = f"GEMM {fmt} X_MODE {g['xm']} swap={g['swap']} BM {g['bm']} BN {g['bn']} BK {bk} split-K {g['sk']} R {br}" \
               f"{' no-LoRA' if not kc['HAS_LORA'] else ''} stages {st} warps {nw}"
        try:
            k = T.build(P4._gemm_pdl4, kc, {"num_warps": nw, "num_stages": st, "launch_pdl": True}, cc=120, ptr=PTR4)
        except Exception as e:   # noqa: BLE001
            ok(name, False, f"{type(e).__name__}: {str(e)[:300]}")
            continue
        ptx, ttg = k.asm["ptx"], k.asm["ttgir"]
        u = T.usage(k)
        nat = ptx.count("kind::mxf4nvf4.block_scale.scale_vec::4X")
        wait_ok = (ptx.count("griddepcontrol.wait") >= 1) == (kc["SYNC_MODE"] == 0)
        pipe = ttg.count("async_copy_global_to_local")
        atom = (kc["SPLIT_K"] == 1 or "atom.global.gpu.acq_rel.add" in ptx) and \
               (kc["X_MODE"] != 1 or "ld.global.gpu.acquire" in ptx)
        ok(name, wait_ok and nat > 0 and pipe > 0 and atom and u["stack"] == 0 and k.metadata.launch_pdl
           and k.metadata.shared <= SMEM,
           f"wait {ptx.count('griddepcontrol.wait')}, FP4 MMA {nat}, async copies {pipe}, L2 prefetch "
           f"{ptx.count('prefetch.global.L2')}, smem {k.metadata.shared}, {u}")
    print("PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--compile" in sys.argv:
        raise SystemExit(compile_main())
    raise SystemExit(interp_main("--quick" in sys.argv))
