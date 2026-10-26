"""Which fp32 division reproduces vLLM's per-token FP8 scale (amax / 448) bit for bit? (GPU)"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scales(AMAX, OUT, N, MODE: tl.constexpr):
    i = tl.arange(0, 1024)
    m = i < N
    a = tl.load(AMAX + i, mask=m, other=1.0)
    if MODE == 0:
        s = tl.inline_asm_elementwise("div.rn.f32 $0, $1, 0f43E00000;", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1)
    elif MODE == 1:
        s = tl.inline_asm_elementwise("div.full.f32 $0, $1, 0f43E00000;", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1)
    elif MODE == 2:
        s = tl.inline_asm_elementwise("div.approx.f32 $0, $1, 0f43E00000;", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1)
    elif MODE == 3:
        s = tl.inline_asm_elementwise("div.approx.ftz.f32 $0, $1, 0f43E00000;", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1)
    else:
        s = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, 0f3B124925;", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(OUT + i, s, mask=m)


def main():
    from vllm import _custom_ops as ops
    torch.manual_seed(0)
    M, K = 1024, 4096
    x = (torch.randn(M, K, device="cuda") * torch.rand(M, 1, device="cuda") * 4).to(torch.bfloat16)
    _, s_v = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    s_v = s_v.view(-1)
    amax = x.float().abs().amax(1).contiguous()
    for mode, name in enumerate(["div.rn", "div.full", "div.approx", "div.approx.ftz", "mul.rn by fp32(1/448)"]):
        out = torch.empty_like(amax)
        _scales[(1,)](amax, out, M, MODE=mode)
        print(f"{name:24s}: identical to vLLM in {int((out == s_v).sum())}/{M} rows")
    s_alt = amax / torch.tensor(448.0, device="cuda")
    print(f"{'torch amax/448':24s}: identical in {int((s_alt == s_v).sum())}/{M} rows")


if __name__ == "__main__":
    main()
