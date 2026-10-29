"""Shared Triton helpers.

to_e4m3: fp32 -> FP8 E4M3 with ONE round-to-nearest-even step (`cvt.rn.satfinite.e4m3x2.f32`, sm_89+), as vLLM's
scaled_fp8_quant and torch do. Triton 3.7's own `.to(tl.float8e4nv)` lowers to `cvt.rz.f16.f32` then
`cvt.rn.satfinite.e4m3x2.f16x2`; the round-toward-zero step pulls values just above an e4m3 midpoint down onto it, where
the tie then goes to even, so ~1% of elements land one e4m3 step away from vLLM's quantization of the same input. That
alone made our GEMM disagree with vLLM's FP8 path end to end. Inputs must already be clamped to +-448.
RN=False keeps Triton's conversion (the CPU interpreter cannot run inline PTX).
"""
import os

import triton
import triton.language as tl


@triton.jit
def to_e4m3(x, RN: tl.constexpr):
    if RN:
        return tl.inline_asm_elementwise(
            asm="cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",     # element 0 ($1) -> low byte, element 1 ($2) -> high
            constraints="=h,r,r", args=[x.to(tl.float32)], dtype=tl.float8e4nv, is_pure=True, pack=2)
    else:
        return x.to(tl.float8e4nv)


def shrink_choice(M: int, R: int, split_k: int, cc_major: int) -> tuple[dict, int]:
    """LoRA shrink tiles and split-K by architecture and batch size, from bench/tune_shrink.py (results/
    shrink_tune.jsonl; the plugin's hybrid path, Llama-3.1-8B projections, r 16/64): 16-row tiles with split-K up to 64
    tokens. Above that, Hopper and Blackwell (H100, RTX PRO 6000) split K up to 512 tokens, with 32-row 256-wide tiles
    below rank 64 and 16-row 128-wide 3-stage tiles from rank 64. Ada (L40S; jobs 65487272, 65487577, 65487737; bf16, FP8- and NVFP4-stored adapters alike) splits K up to 512
    tokens and reads A in 128-wide K blocks from rank 64: within ~1% of the best measured tiles wherever they were
    measured, where split 1 took up to 1.57x as long at 128-256 tokens."""
    sr = min(64, R)
    if M <= 64:
        return dict(BLOCK_M=16, BLOCK_SR=sr, BLOCK_K=256, num_warps=4, num_stages=3), split_k
    if cc_major < 9:
        bk = 128 if R >= 64 else 256
        if M <= 256:
            return dict(BLOCK_M=16, BLOCK_SR=sr, BLOCK_K=bk, num_warps=4, num_stages=3), min(split_k, 8)
        if M <= 512:
            return dict(BLOCK_M=32, BLOCK_SR=sr, BLOCK_K=bk, num_warps=4, num_stages=3), min(split_k, 4)
        # Not split above 512 tokens on Ada: the hybrid path's tile search (job 65496306) favoured split 4 / 2 at
        # 1024 / 2048 tokens (worst regret 1.02 vs 1.31), but end to end that rule cost one rank-64/128 adapter 10
        # points of W2 throughput on both models (jobs 65496377-82 vs 65491612 / 65488620: the timed dispatch moved
        # those buckets to the fused path), so Ada keeps split 1 there
        if M <= 2048:
            return dict(BLOCK_M=16, BLOCK_SR=sr, BLOCK_K=bk, num_warps=4, num_stages=3), 1
        return dict(BLOCK_M=32, BLOCK_SR=sr, BLOCK_K=128, num_warps=4, num_stages=3), 1
    if R >= 64:   # Hopper and Blackwell at rank >= 64: 128-wide K blocks, 3 stages (job 65489389: the 256-wide
        # 2-stage tiles took 1.2-1.6x the best tile's time at 128-1024 tokens)
        if M <= 512:
            return dict(BLOCK_M=16, BLOCK_SR=sr, BLOCK_K=128, num_warps=4, num_stages=3), max(1, split_k // 2)
        if cc_major == 9 and os.environ.get("LOWBIT_SHRINK_LARGE", "0") not in ("0", "", "false", "False"):
            # Large-batch tiles (opt-in, off by default): bench/tune_shrink.py on Qwen3-8B's projections (Qwen3.5-9B's
            # dense shapes), rank 64, on the H200 (sm_90 with the H100's 132 SMs): per weight width and batch size the
            # tile with the least summed hybrid-path time over the projections. The layer took 0.98x / 0.94x / 0.92x
            # the stock path's time at 1,024 / 2,048 / 4,096 tokens with these tiles, against 1.08x / 1.03x / 1.01x
            # with the default tiles below (the down projection 1.21-1.24x stock). split_k >= 16 marks K >= 8192
            # (the down projection).
            if split_k >= 16:
                bk, sp = (128, 8) if M <= 1024 else (256, 4) if M <= 2048 else (256, 2)
            else:
                bk, sp = 128, (2 if M <= 2048 else 1)
            return dict(BLOCK_M=32, BLOCK_SR=sr, BLOCK_K=bk, num_warps=4, num_stages=3), min(split_k, sp)
        if M <= 2048:
            # split K up to 2048 tokens too (r64, bf16 and NVFP4-stored adapters; worst regret over projections and
            # formats): Blackwell, RTX PRO 6000 slice (job 65495078), 32-row tiles, 1.06 at 1024 tokens against 1.45 for
            # split 1 (the down projection's 32 thread blocks took 442 us against stock's 353, now 340-351); H100
            # (job 65496305), 64-row tiles, 1.20 at 1024 and 1.21 at 2048 tokens against 2.31 and 1.58
            return (dict(BLOCK_M=64 if cc_major == 9 else 32, BLOCK_SR=sr, BLOCK_K=128, num_warps=4, num_stages=3),
                    min(split_k, 4 if M <= 1024 else 2))
        return dict(BLOCK_M=32, BLOCK_SR=sr, BLOCK_K=128, num_warps=4, num_stages=3), 1
    if M <= 256:
        return dict(BLOCK_M=32, BLOCK_SR=sr, BLOCK_K=256, num_warps=4, num_stages=2), min(split_k, 8)
    if M <= 512:
        return dict(BLOCK_M=32, BLOCK_SR=sr, BLOCK_K=256, num_warps=4, num_stages=3), min(split_k, 4)
    if M <= 2048:
        return dict(BLOCK_M=32, BLOCK_SR=sr, BLOCK_K=256, num_warps=4, num_stages=3), 1
    return dict(BLOCK_M=32, BLOCK_SR=sr, BLOCK_K=128, num_warps=4, num_stages=3), 1
