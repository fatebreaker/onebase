"""Offline compile of the vLLM-facing kernels for real GPU targets (no GPU needed; seconds on a CPU).

The CPU interpreter runs Python semantics and skips Triton's compiler, so code that interprets fine can still fail to
compile (e.g. a global used inside @jit). This lowers every pick_cfg variant to PTX/cubin for sm_89, sm_90 and sm_120.
    python tests/compile_check.py
"""
from __future__ import annotations

import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lowbit_lora.kernels import fp8_lora_vllm as F
from lowbit_lora.kernels import nvfp4_lora_vllm as F4

PTR = {"X": "*bf16", "XQ": "*fp8e4nv", "SX": "*fp32", "PMAX": "*fp32", "W": "*fp8e4nv", "SW": "*fp32", "HP": "*fp32",
       "B_PTRS": "*i64", "A_PTRS": "*i64", "SA_PTRS": "*i64", "SB_PTRS": "*i64", "AG": "*fp32", "BG": "*fp32", "Y": "*bf16", "TOK": "*i32", "SRT": "*i32", "SORTED": "*i32",
       "SEG_IDS": "*i32", "SEG_N": "*i32", "SEG_START": "*i32", "N_SEG": "*i32"}
PTR4 = {**PTR, "XQ": "*u8", "SXQ": "*fp8e4nv", "W": "*u8", "SW": "*fp8e4nv", "GS": "*fp32", "ALPHA": "*fp32"}


def build(fn, consts: dict, cc: int, opts: dict, ptr: dict = PTR):
    # constexpr parameters left at their defaults (e.g. R_CHUNK) must still be given to ASTSource, which otherwise
    # types them as i32 runtime arguments
    consts = {**{p.name: p.default for p in fn.params if p.is_constexpr and p.has_default and p.name not in consts},
              **consts}
    sig = {a: ("constexpr" if a in consts else ptr.get(a, "i32")) for a in fn.arg_names}
    return triton.compile(ASTSource(fn, signature=sig, constexprs=consts), target=GPUTarget("cuda", cc, 32),
                          options=opts)


def main() -> int:
    bad = 0
    # Llama-3.1-8B qkv projection, r64: widths (4096, 1024, 1024), K 4096
    N, K, R, widths = 6144, 4096, 64, (4096, 1024, 1024)
    for cc in (89, 90, 120):
        for M in (16, 64, 128, 256, 1024):
            cs, cg = F.pick_cfg(M, N, K, R, widths)
            for prefill, lora8, lora4 in ((False, False, False), (True, False, False), (False, True, False),
                                          (True, True, False), (False, False, True), (True, False, True)):
                split = 8 if M <= 512 else 1   # the default path splits the shrink up to SPLIT_M_MAX tokens
                try:
                    build(F._shrink_vllm, dict(SPLIT_K=split, WRITE_XQ=prefill, N_SR=3 * R // cs["BLOCK_SR"],
                                               BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"], BLOCK_K=cs["BLOCK_K"],
                                               DOT_F32=False, RN=True, A_FP8=lora8, A_FP4=lora4),
                          cc, {"num_warps": cs["num_warps"], "num_stages": cs["num_stages"]})
                    build(F._gemm_vllm, dict(X_FP8=True, SPLIT_K1=split, BLOCK_R=R, EVEN_K=True, BLOCK_M=cg["BLOCK_M"],
                                             BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"],
                                             DOT_F32=False, RN=True, PRECISE=True, B_FP8=lora8, B_FP4=lora4),
                          cc, {"num_warps": cg["num_warps"], "num_stages": cg["num_stages"]})
                    build(F._gemm_vllm, dict(X_FP8=True, SPLIT_K1=split, BLOCK_R=R, EVEN_K=True, BLOCK_M=cg["BLOCK_M"],
                                             BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"],
                                             DOT_F32=False, RN=True, PRECISE=True, B_FP8=lora8, B_FP4=lora4,
                                             BASE=False),   # the hybrid's expand-add kernel
                          cc, {"num_warps": cg["num_warps"], "num_stages": cg["num_stages"]})
                    print(f"sm_{cc} M={M} {'prefill' if prefill else 'decode '}{' fp8-lora' if lora8 else ''}"
                          f"{' fp4-lora' if lora4 else ''}: ok (+ expand-add)")
                except Exception as e:   # noqa: BLE001 -- report every failing variant
                    bad += 1
                    print(f"sm_{cc} M={M} {'prefill' if prefill else 'decode '}{' fp8-lora' if lora8 else ''}: FAIL "
                          f"{type(e).__name__}: {str(e)[:400]}")
    # the tuned shrink tiles (common.shrink_choice) on Ada, Hopper and Blackwell, both kernels' shrinks
    from lowbit_lora.kernels.common import shrink_choice
    for cc in (89, 90, 120):
        for M in (128, 256, 1024, 4096):
            cs, split = shrink_choice(M, R, 8, cc // 10)
            for lora8, lora4 in ((False, False), (True, False), (False, True)):
                try:
                    build(F._shrink_vllm, dict(SPLIT_K=split, WRITE_XQ=True, N_SR=3 * R // cs["BLOCK_SR"],
                                               BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"], BLOCK_K=cs["BLOCK_K"],
                                               DOT_F32=False, RN=True, A_FP8=lora8, A_FP4=lora4),
                          cc, {"num_warps": cs["num_warps"], "num_stages": cs["num_stages"]})
                    if cc == 120:
                        build(F4._shrink_nvfp4, dict(SPLIT_K=split, N_SR=3 * R // cs["BLOCK_SR"], BLOCK_M=cs["BLOCK_M"],
                                                     BLOCK_SR=cs["BLOCK_SR"], BLOCK_K=cs["BLOCK_K"], DOT_F32=False,
                                                     PTX=True, A_FP8=lora8, A_FP4=lora4),
                              cc, {"num_warps": cs["num_warps"], "num_stages": cs["num_stages"]}, PTR4)
                except Exception as e:   # noqa: BLE001
                    bad += 1
                    print(f"tuned shrink sm_{cc} M={M} lora8={lora8} lora4={lora4}: FAIL {type(e).__name__}: {str(e)[:300]}")
            print(f"tuned shrink sm_{cc} M={M} {cs['BLOCK_M']}x{cs['BLOCK_SR']}x{cs['BLOCK_K']} split {split}: ok")
    # NVFP4 (W4A4) kernels: sm_120 only (native block-scaled FP4 MMA); the GEMM must use it
    for M in (16, 64, 256, 1024):
        cs, cg = F4.pick_cfg(M, N, K, R, widths)
        split = 8 if M <= 512 else 1
        for fmt in ("bf16", "fp8", "nvfp4"):   # adapter storage
            ad = dict(A_FP8=fmt == "fp8", A_FP4=fmt == "nvfp4")
            bd = dict(B_FP8=fmt == "fp8", B_FP4=fmt == "nvfp4")
            try:
                build(F4._shrink_nvfp4, dict(SPLIT_K=split, N_SR=3 * R // cs["BLOCK_SR"], BLOCK_M=cs["BLOCK_M"],
                                             BLOCK_SR=cs["BLOCK_SR"], BLOCK_K=cs["BLOCK_K"], DOT_F32=False, PTX=True,
                                             **ad),
                      120, {"num_warps": cs["num_warps"], "num_stages": cs["num_stages"]}, PTR4)
                k = build(F4._gemm_nvfp4, dict(SPLIT_K1=split, BLOCK_R=R, EVEN_K=True, BLOCK_M=cg["BLOCK_M"],
                                               BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"],
                                               DOT_F32=False, **bd),
                          120, {"num_warps": cg["num_warps"], "num_stages": cg["num_stages"]}, PTR4)
                fp4 = "kind::mxf4nvf4.block_scale.scale_vec::4X" in k.asm["ptx"]
                bad += not fp4
                if M > 64:   # the hybrid's expand-add kernel (vendor GEMM output + LoRA in place)
                    build(F4._gemm_nvfp4, dict(SPLIT_K1=split, BLOCK_R=R, EVEN_K=True, BLOCK_M=cg["BLOCK_M"],
                                               BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"],
                                               DOT_F32=False, BASE=False, **bd),
                          120, {"num_warps": cg["num_warps"], "num_stages": cg["num_stages"]}, PTR4)
                print(f"nvfp4 sm_120 M={M} {fmt}-adapters: ok, native FP4 MMA {fp4}"
                      f"{', expand-add ok' if M > 64 else ''}")
            except Exception as e:   # noqa: BLE001
                bad += 1
                print(f"nvfp4 sm_120 M={M} {fmt}-adapters: FAIL {type(e).__name__}: {str(e)[:400]}")
    # rank 128 in 64-wide epilogue chunks (the hosts' rule: NVFP4-stored B and the expand-add), every storage format
    R2 = 128
    for cc in (89, 90, 120):
        _, cg = F.pick_cfg(1024, N, K, R2, widths, cc // 10)
        for fmt in ("bf16", "fp8", "nvfp4"):
            for base in (True, False):
                try:
                    build(F._gemm_vllm, dict(X_FP8=True, SPLIT_K1=1, BLOCK_R=R2, EVEN_K=True, BLOCK_M=cg["BLOCK_M"],
                                             BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"],
                                             DOT_F32=False, RN=True, PRECISE=True, B_FP8=fmt == "fp8",
                                             B_FP4=fmt == "nvfp4", BASE=base, R_CHUNK=64),
                          cc, {"num_warps": cg["num_warps"], "num_stages": cg["num_stages"]})
                except Exception as e:   # noqa: BLE001
                    bad += 1
                    print(f"rank-128 chunks sm_{cc} {fmt} base={base}: FAIL {type(e).__name__}: {str(e)[:300]}")
        print(f"rank-128 chunks sm_{cc} (fp8 op, 3 formats, fused and expand-add): ok")
    _, cg = F4.pick_cfg(256, N, K, R2, widths)
    for fmt in ("bf16", "nvfp4"):
        for base in (True, False):
            try:
                build(F4._gemm_nvfp4, dict(SPLIT_K1=8, BLOCK_R=R2, EVEN_K=True, BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"],
                                           BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"], DOT_F32=False, BASE=base,
                                           B_FP4=fmt == "nvfp4", R_CHUNK=64),
                      120, {"num_warps": cg["num_warps"], "num_stages": cg["num_stages"]}, PTR4)
            except Exception as e:   # noqa: BLE001
                bad += 1
                print(f"rank-128 chunks nvfp4 op {fmt} base={base}: FAIL {type(e).__name__}: {str(e)[:300]}")
    print("rank-128 chunks sm_120 (nvfp4 op): ok")
    print("PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
