"""CPU checks of the PDL decode path (kernels/fp8_lora_pdl.py via lowbit_lora/hopper_pdl.run_pdl), no GPU needed.

Interpreter (default; Triton's CPU interpreter, driven by vLLM's own LoRAKernelMeta on the CPU and vLLM's per-slice stacked
adapter buffers): every activation mode (X_MODE 0: the GEMM quantizes x itself; 1: per-token quantization programs and a
release/acquire flag; 2: the GEMM waits for the whole shrink kernel), split-K 1 and 3 (the last-arriving CTA's fixed-order
reduction), bf16, FP8- and NVFP4-stored adapters, ranks 16 to 128 (64-wide epilogue chunks), one to four slices, against
an fp32 reference with x quantized by torch (relative error at most 1e-2; bf16 rounding is about 4e-3). Cases: one token;
single-token segments; empty segments (8 slots, 2 used); tokens without an adapter; a batch without adapters after an
adapter batch (vLLM leaves the token map stale); CUDA-graph padding (stale metadata past the real tokens). Also:
  * the codes and scales the GEMM computes (X_MODE 0, through its debug output) and the quantization programs write
    (X_MODE 1, 2) are byte-identical to torch's per-token quantization (IEEE division, round to nearest even), i.e. to
    K1's rule (the interpreter emulates the PTX conversion exactly, QRN=2);
  * repeated launches are bitwise identical (split-K included); every counter is back at zero after each launch;
  * the merged fallback kernel (coop) and the GEMM without the shrink launch (h from the previous shrink) give the PDL
    path's output bit for bit; the GEMM without LoRA gives the base GEMM.
    TRITON_INTERPRET=1 python tests/pdl_k2_interp_check.py [--quick]

--compile (no interpreter): offline compile for sm_90 of the shrink, the GEMM and the merged kernel at the bench's
decode tiles, with the JIT's specialization (16-byte divisibility of pointers and of the multiple-of-16 integers, as at
run time; without it Triton does not pipeline), checking the PTX for griddepcontrol.launch_dependents (shrink),
griddepcontrol.wait (GEMM), FP8 wgmma (e4m3 x e4m3), cp.async pipelining of the weight stream, the release / acquire
atomics of split-K and of the flag, and the cubin's registers and spills (cuobjdump).
    python tests/pdl_k2_interp_check.py --compile
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------------------------------------- interpreter
def interp_main(quick: bool) -> int:
    assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1 (or --compile)"
    import torch
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    from lowbit_lora import hopper_pdl as H
    from lowbit_lora.kernels import fp8_lora_multi as FM
    from lowbit_lora.kernels import fp8_lora_vllm as F
    from lowbit_lora.kernels.fp8_lora_pdl import work
    from lowbit_lora.kernels.nvfp4_lora_vllm import nvfp4_dequant_torch as dq4

    torch.manual_seed(0)
    worst, bad = 0.0, []
    wk = work(torch.device("cpu"))

    def check(name, ok, msg=""):
        if not ok:
            bad.append(f"{name}: {msg}")
            print("  FAIL", name, msg)

    def tq(x):   # torch's per-token quantization: K1's rule (IEEE division, RN-even E4M3), the reference's
        xf = x.float()
        sx = (xf.abs().amax(dim=1) / 448.0).clamp_min(1.0 / (448.0 * 512.0))
        return (xf / sx[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), sx

    # (K, slice widths, rank, adapter slots, the shrink's split): one to four slices; rank 128 in 64-wide epilogue chunks
    shapes = [(256, (64, 32, 32), 16, 4, 2), (192, (128,), 64, 3, 3), (256, (64, 64), 32, 8, 2), (128, (64,), 128, 2, 1),
              (128, (32, 32, 32, 32), 16, 3, 2)]
    if quick:
        shapes = shapes[:2]
    for K, widths, R, L, split1 in shapes:
        N, S = sum(widths), len(widths)
        w = torch.randn(N, K) * 0.05
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        a_st = tuple((torch.randn(L, 1, R, K) / K ** 0.5).to(torch.bfloat16) for _ in range(S))
        b_st = tuple((torch.randn(L, 1, n, R) * 0.05).to(torch.bfloat16) for n in widths)
        A = torch.cat([t[:, 0] for t in a_st], dim=1)
        B = torch.cat([t[:, 0] for t in b_st], dim=1)
        ops = {"bf16": (F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64), A, B)}
        op8 = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64, lora_fp8=True)
        ops["fp8"] = (op8, torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(op8.a8, op8.sa)], dim=1),
                      torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(op8.b8, op8.sb)], dim=1))
        if R % 16 == 0:
            op4 = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64, lora_fp4=True)

            def deq4(q, qs, g, i):
                return dq4(q[i, 0], qs[i, 0], 1.0) * g[i]
            ops["fp4"] = (op4, torch.cat([torch.stack([deq4(q, qs, op4.ag[s], i) for i in range(L)]) for s, (q, qs) in
                                          enumerate(zip(op4.a4, op4.a4s))], dim=1),
                          torch.cat([torch.stack([deq4(q, qs, op4.bg[s], i) for i in range(L)]) for s, (q, qs) in
                                     enumerate(zip(op4.b4, op4.b4s))], dim=1))
        for o, _, _ in ops.values():
            o.dot_f32 = True
        meta = LoRAKernelMeta.make(L, 64, device="cpu")
        bn = min(64, min(widths))
        cases = [(1, "one"), (min(4, L), "single-token segs"), (21, "mixed"), (6, "two of L"), (13, "none"),
                 (24, "padded")]
        if quick:
            cases = cases[:3] + cases[4:5]
        for M, kind in cases:
            x = torch.randn(M, K).to(torch.bfloat16)
            if kind == "one":
                tok = torch.full((M,), L - 1, dtype=torch.int32)
            elif kind == "single-token segs":
                tok = torch.randperm(L)[:M].to(torch.int32)              # M distinct adapters, one token each
            elif kind == "mixed":
                tok = torch.randint(-1, L, (M,), dtype=torch.int32)      # several adapters and tokens without one
            elif kind == "two of L":
                tok = torch.tensor([1, 1, 0, 1, 0, 0][:M], dtype=torch.int32) % L   # the other slots: empty segments
            else:
                tok = torch.full((M,), -1, dtype=torch.int32)
            if kind == "padded":   # graph padding: metadata for the first M - 5 tokens, stale entries beyond
                tok = torch.randint(-1, L, (M,), dtype=torch.int32)
                tok[0] = 0
                meta.token_indices_sorted_by_lora_ids[M - 5:M] = 10 ** 6
                meta.token_lora_mapping[M - 5:M] = L - 1                # a stale adapter id the padding must ignore
                meta.prepare_tensors(tok[: M - 5])
                tok[M - 5:] = -1
            else:
                if kind == "none":   # after an adapter batch: vLLM leaves the token map stale
                    meta.prepare_tensors(torch.zeros(M, dtype=torch.int32))
                meta.prepare_tensors(tok)
            margs = meta.meta_args(M, False)
            xq_ref, sx_ref = tq(x)
            for fmt, (op, Ar, Br) in ops.items():
                ref = FM.reference_multi(x, tok, w_q, s_w, Ar, Br, widths, 1.0)
                ref0 = FM.reference_multi(x, torch.full((M,), -1, dtype=torch.int32), w_q, s_w, Ar, Br, widths, 1.0)
                modes = (0, 1, 2) if fmt == "bf16" else (0, 1)
                for xm in modes:
                    for sk in ((1, 3) if fmt == "bf16" else (2,)):
                        cfg = {"X_MODE": xm, "BLOCK_M": 16, "BLOCK_N": bn, "BLOCK_K": 64, "SPLIT_K": sk,
                               "num_stages": 1, "SPLIT_K1": split1, "S_BLOCK_K": 32, "S_STAGES": 1}
                        xqd = torch.zeros(M, K, dtype=torch.float8_e4m3fn) if xm == 0 else None
                        y = H.run_pdl(op, x, margs, cfg=cfg, xqd=xqd).float()
                        err = float((y - ref).norm() / ref.norm())
                        worst = max(worst, err)
                        tag = f"K={K} w={widths} R={R} L={L} M={M} {kind} {fmt} X_MODE={xm} split-K={sk}"
                        check(tag + " error", err < 1e-2, f"rel err {err:.2e}")
                        if xm == 0:
                            check(tag + " codes", torch.equal(xqd.view(torch.uint8), xq_ref.view(torch.uint8)),
                                  f"{int((xqd.view(torch.uint8) != xq_ref.view(torch.uint8)).sum())} codes differ")
                        else:
                            xq_k = op.scratch.xq[: M * K].view(M, K)
                            check(tag + " codes", torch.equal(xq_k.view(torch.uint8), xq_ref.view(torch.uint8)),
                                  f"{int((xq_k.view(torch.uint8) != xq_ref.view(torch.uint8)).sum())} codes differ")
                            check(tag + " scales", torch.equal(op.scratch.sx[:M], sx_ref), "scales differ")
                        check(tag + " counters", int(wk.cnt.abs().sum()) == 0 and int(wk.sync.abs().sum()) == 0,
                              f"cnt {wk.cnt.tolist()} sync {wk.sync.tolist()}")
                        y2 = H.run_pdl(op, x, margs, cfg=cfg).float()
                        check(tag + " repeat", torch.equal(y, y2), "a second launch differs")
                        if fmt == "bf16" and xm != 2:
                            y3 = H.run_pdl(op, x, margs, cfg=cfg, shrink=False).float()   # h from the launch above
                            check(tag + " no-shrink", torch.equal(y, y3), "differs from the PDL path")
                            yc = H.run_pdl(op, x, margs, cfg=cfg, coop=True).float()
                            check(tag + " coop", torch.equal(y, yc),
                                  f"max |coop - pdl| {float((yc - y).abs().max()):.3e}")
                            check(tag + " coop counters",
                                  int(wk.cnt.abs().sum()) == 0 and int(wk.sync.abs().sum()) == 0,
                                  f"cnt {wk.cnt.tolist()} sync {wk.sync.tolist()}")
                        if fmt == "bf16" and kind in ("one", "mixed") and sk == 1:
                            y0 = H.run_pdl(op, x, margs, cfg=cfg, lora=False).float()
                            e0 = float((y0 - ref0).norm() / ref0.norm())
                            worst = max(worst, e0)
                            check(tag + " no-LoRA", e0 < 1e-2, f"rel err {e0:.2e}")
            print(f"K={K} widths={widths} R={R} L={L} M={M:2d} {kind:17s} formats {'/'.join(ops)}: worst so far "
                  f"{worst:.2e}", flush=True)
    print(("PASS" if not bad else f"FAIL ({len(bad)})") + f"  worst rel err {worst:.2e}")
    for b in bad[:20]:
        print("  ", b)
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------------ offline compile
PTR = {"X": "*bf16", "XQ": "*fp8e4nv", "SX": "*fp32", "W": "*fp8e4nv", "WP": "*fp8e4nv", "SW": "*fp32", "HP": "*fp32",
       "A_PTRS": "*i64", "SA_PTRS": "*i64", "AG": "*fp32", "B_PTRS": "*i64", "SB_PTRS": "*i64", "BG": "*fp32",
       "Y": "*bf16", "P": "*fp32", "CNT": "*i32", "SYNC": "*i32", "TOK": "*i32", "SORTED": "*i32", "SEG_IDS": "*i32",
       "SEG_N": "*i32", "SEG_START": "*i32", "XQD": "*fp8e4nv"}
DIV16 = {"N", "K", "K_SPLIT", "stride_xm", "stride_xqm", "stride_wn", "stride_ym", "stride_al", "stride_am", "R"}


RELEASE = r"(atom|red)\.global\.gpu\.release"   # a gpu-scope release RMW


def build(fn, consts: dict, opts: dict, cc: int = 90, ptr: dict = PTR):
    """triton.compile with the runtime JIT's specialization: pointers and the multiple-of-16 integers marked 16-divisible
    (the arguments in fn's do_not_specialize list are left unspecialized, as at run time)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    consts = {**{p.name: p.default for p in fn.params if p.is_constexpr and p.has_default and p.name not in consts},
              **consts}
    sig = {a: ("constexpr" if a in consts else ptr.get(a, "i32")) for a in fn.arg_names}
    attrs = {(i,): [["tt.divisibility", 16]] for i, a in enumerate(fn.arg_names)
             if a not in consts and (sig[a].startswith("*") or a in DIV16)}
    return triton.compile(ASTSource(fn, signature=sig, constexprs=consts, attrs=attrs),
                          target=GPUTarget("cuda", cc, 32), options=opts)


def usage(k) -> dict:
    """Registers, stack (spills) and static shared memory of a compiled kernel's cubin (cuobjdump --dump-resource-usage)."""
    import re
    import subprocess
    import tempfile
    import triton
    tool = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(k.asm["cubin"])
    out = subprocess.run([str(tool), "--dump-resource-usage", f.name], capture_output=True, text=True).stdout
    os.unlink(f.name)
    m = re.search(r"REG:(\d+).*?STACK:(\d+)", out, re.S)
    return {"reg": int(m.group(1)), "stack": int(m.group(2))} if m else {"reg": -1, "stack": -1}


def k2_consts(xm=0, bm=16, bn=64, bk=256, sk=1, br=64, rck=None, bp=2, bpipe=True, wpf=2, wtma=False, lora=True,
              sync=0, split1=4, fmt="bf16", L=8, am=1, abk=4096):
    import triton
    rck = rck or br
    return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_N=8, SPLIT_K=sk, EVEN_K=True, X_MODE=xm, QRN=1, AMAX_M=am,
                AMAX_BK=abk, HAS_LORA=lora, SPLIT_K1=split1, BLOCK_R=br, R_CHUNK=rck,
                LP=triton.next_power_of_2(L + 1), B_PREFETCH=bp if rck == br else min(bp, 1), B_PIPE=bpipe,
                W_PREFETCH=wpf, W_TMA=wtma, SYNC_MODE=sync, PRECISE=True, B_FP8=fmt == "fp8", B_FP4=fmt == "fp4")


def compile_main() -> int:
    assert os.environ.get("TRITON_INTERPRET", "0") != "1", "--compile runs without the interpreter"
    import re
    from lowbit_lora.kernels import fp8_lora_pdl as P
    bad = 0

    def ok(name, cond, extra=""):
        nonlocal bad
        bad += not cond
        print(("  ok   " if cond else "  FAIL ") + name + (f"  {extra}" if extra else ""))

    print("== shrink (adapter segments; + quantization programs), sm_90")
    for qseg, qsig, fmt in ((False, False, "bf16"), (True, True, "bf16"), (True, False, "bf16"), (False, False, "fp8"),
                            (False, False, "fp4"), (True, True, "fp4")):
        for split, bk in ((4, 128), (8, 128), (16, 64)):
            k = build(P._shrink_pdl, dict(SPLIT_K=split, N_SR=3, BLOCK_M=16, BLOCK_SR=64, BLOCK_K=bk, QSEG=qseg,
                                          QSIGNAL=qsig, QBLOCK=4096, KQ=12288, QRN=1, USE_GDC=True,
                                          A_FP8=fmt == "fp8", A_FP4=fmt == "fp4"), {"num_warps": 4, "num_stages": 2})
            ptx = k.asm["ptx"]
            ok(f"shrink qseg={qseg} signal={qsig} {fmt} split {split} BLOCK_K {bk}",
               ptx.count("griddepcontrol.launch_dependents") == 1
               and (not qseg or "cvt.rn.satfinite.e4m3x2.f32" in ptx)
               and (not qsig or re.search(RELEASE, ptx) is not None),
               f"launch_dependents {ptx.count('griddepcontrol.launch_dependents')}, cp.async "
               f"{ptx.count('cp.async.cg') + ptx.count('cp.async.ca')}, smem {k.metadata.shared}, {usage(k)}")
    print("== GEMM (K2), sm_90, launch_pdl")
    grid = []
    for xm in (0, 1, 2):
        for bn, bk, sk, st in ((64, 256, 1, 4), (64, 256, 4, 4), (128, 128, 2, 5), (64, 128, 8, 6)):
            grid.append(dict(xm=xm, bn=bn, bk=bk, sk=sk, st=st))
    grid += [dict(xm=0, bn=64, bk=256, sk=2, st=4, wtma=True), dict(xm=1, bn=128, bk=128, sk=1, st=4, wtma=True),
             dict(xm=0, bn=64, bk=256, sk=1, st=4, br=128), dict(xm=1, bn=128, bk=256, sk=1, st=3, br=128, rck=64,
                                                                  bp=1),
             dict(xm=0, bn=64, bk=256, sk=2, st=4, br=16), dict(xm=1, bn=64, bk=256, sk=2, st=4, fmt="fp8"),
             dict(xm=1, bn=64, bk=256, sk=2, st=4, fmt="fp4"), dict(xm=0, bn=64, bk=256, sk=1, st=4, br=128, rck=64,
                                                                   fmt="fp4", bp=1),
             dict(xm=0, bn=64, bk=256, sk=1, st=4, br=16, fmt="fp4"),
             dict(xm=0, bn=64, bk=256, sk=1, st=4, lora=False, sync=1, bp=0, wpf=0),
             dict(xm=0, bn=64, bk=256, sk=2, st=4, bm=32, am=32, abk=512), dict(xm=1, bn=64, bk=128, sk=1, st=4, bm=64, nw=8),
             dict(xm=1, bn=128, bk=128, sk=2, st=4, bm=32, nw=8), dict(xm=1, bn=128, bk=128, sk=1, st=4, bm=64, nw=8),
             dict(xm=1, bn=64, bk=128, sk=1, st=3, bm=128, bpipe=False, nw=8)]
    for g in grid:
        st = g.pop("st")
        nw = g.pop("nw", 4)
        kc = k2_consts(**g)
        ptr = {**PTR, "W": f"tensordesc<fp8e4nv[{kc['BLOCK_N']}, {kc['BLOCK_K']}]>"} if kc["W_TMA"] else PTR
        try:
            k = build(P._gemm_pdl, kc, {"num_warps": nw, "num_stages": st, "launch_pdl": True}, ptr=ptr)
        except Exception as e:   # noqa: BLE001
            ok(f"GEMM {g} stages {st}", False, f"{type(e).__name__}: {str(e)[:300]}")
            continue
        ptx, ttg = k.asm["ptx"], k.asm["ttgir"]
        u = usage(k)
        wait_ok = (ptx.count("griddepcontrol.wait") >= 1) == (kc["SYNC_MODE"] == 0)
        wg = ptx.count("wgmma.mma_async.sync.aligned") and "e4m3.e4m3" in ptx
        pipe = ttg.count("async_copy_global_to_local") + ttg.count("async_tma_copy_global_to_local")
        atom = (kc["SPLIT_K"] == 1 or "atom.global.gpu.acq_rel.add" in ptx) and \
               (kc["X_MODE"] != 1 or "ld.global.gpu.acquire" in ptx)
        ok(f"GEMM {g} stages {st}", wait_ok and wg and pipe > 0 and atom and u["stack"] == 0 and k.metadata.launch_pdl,
           f"nw {nw}, wait {ptx.count('griddepcontrol.wait')}, FP8 wgmma {ptx.count('wgmma.mma_async')}, async copies {pipe}, "
           f"L2 prefetch {ptx.count('prefetch.global.L2')}, smem {k.metadata.shared}, {u}")
    print("== merged fallback kernel (coop), sm_90")
    for xm in (0, 1, 2):
        kc = k2_consts(xm=xm, bn=64, bk=256, sk=2)
        for key in ("HAS_LORA", "SYNC_MODE"):
            kc.pop(key)
        kc.update(S_SPLIT_K=4, N_SR=3, S_BLOCK_M=16, BLOCK_SR=64, S_BLOCK_K=128, QBLOCK=4096, KQ=4096)
        k = build(P._coop, kc, {"num_warps": 4, "num_stages": 4, "launch_cooperative_grid": True})
        ptx = k.asm["ptx"]
        u = usage(k)
        ok(f"coop X_MODE {xm}", "ld.global.gpu.acquire" in ptx and re.search(RELEASE, ptx) is not None
           and "e4m3.e4m3" in ptx and u["stack"] == 0,
           f"acquire {ptx.count('ld.global.gpu.acquire')}, smem {k.metadata.shared}, {u}")
    print("PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--compile" in sys.argv:
        raise SystemExit(compile_main())
    raise SystemExit(interp_main("--quick" in sys.argv))
