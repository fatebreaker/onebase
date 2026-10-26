"""Simulated rounding for the on-grid / off-grid study.

Every quantizer takes a 2-D weight [out, in] and returns its dequantized value in float32, so that callers can
compare the delivered update  Q(W + dW) - Q(W)  with the true update dW entry by entry.

Grids are per output row (per-channel) or per group of `group` consecutive input columns within a row. vLLM fuses
q/k/v and gate/up by stacking ROWS, so row-wise and row-group-wise grids give the same result on fused tensors.

Two ways to round an update onto a grid (docs/THEORY.md):
  * ON-GRID: the base already sits on the grid (a bf16 checkpoint merged in bf16; a model quantized once and then
    re-quantized with the same scales). Nearest rounding is a dead zone: |dW| < step/2 is erased.
  * OFF-GRID ("fresh"): the base is quantized from higher precision, so its position inside a cell is effectively
    uniform. Nearest rounding then delivers dW unbiasedly, as sparse +-step flips (the same law as stochastic
    rounding of dW).
"""
from __future__ import annotations

import torch

FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ---------------------------------------------------------------------------------------------------- helpers

def _pad_cols(x: torch.Tensor, group: int) -> tuple[torch.Tensor, int]:
    pad = (-x.shape[-1]) % group
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, pad


def _grouped(x: torch.Tensor, group: int | None) -> tuple[torch.Tensor, int]:
    """[out, in] -> [out, n_groups, group] (zero-padded); group=None -> one group per row."""
    if group is None:
        return x.unsqueeze(1), 0
    x, pad = _pad_cols(x, group)
    return x.reshape(x.shape[0], -1, group), pad


def _ungrouped(x: torch.Tensor, pad: int) -> torch.Tensor:
    x = x.reshape(x.shape[0], -1)
    return x[:, : x.shape[1] - pad] if pad else x


def _round(x: torch.Tensor, mode: str, gen: torch.Generator | None) -> torch.Tensor:
    if mode == "nearest":
        return torch.round(x)                        # ties to even
    if mode == "stochastic":
        return torch.floor(x + torch.rand(x.shape, generator=gen, dtype=x.dtype, device=x.device))
    raise ValueError(f"unknown rounding mode {mode!r}")


def _round_to_levels(x: torch.Tensor, levels: torch.Tensor, mode: str, gen: torch.Generator | None,
                     even: torch.Tensor | None = None) -> torch.Tensor:
    """Round to the nearest value of a sorted 1-D grid `levels` (ties -> the level flagged `even`), or
    stochastically between the two neighbours. Values outside the grid saturate."""
    levels = levels.to(device=x.device, dtype=x.dtype)
    even = even.to(x.device) if even is not None else None
    x = x.clamp(levels[0], levels[-1])
    hi = torch.searchsorted(levels, x.contiguous()).clamp(1, len(levels) - 1)
    lo = hi - 1
    a, b = levels[lo], levels[hi]
    frac = (x - a) / (b - a)
    if mode == "stochastic":
        up = torch.rand(x.shape, generator=gen, dtype=x.dtype, device=x.device) < frac
    elif mode == "nearest":
        up = frac > 0.5
        tie = frac == 0.5
        if even is not None:
            up = torch.where(tie, even[hi], up)
    else:
        raise ValueError(f"unknown rounding mode {mode!r}")
    return torch.where(up, b, a)


# ---------------------------------------------------------------------------------------------------- formats

def bf16(x: torch.Tensor, mode: str = "nearest", gen: torch.Generator | None = None) -> torch.Tensor:
    """Round float32 to bfloat16 (as float32). Stochastic mode adds uniform noise below the kept 16 bits."""
    x = x.float()
    if mode == "nearest":
        return x.to(torch.bfloat16).float()
    if mode != "stochastic":
        raise ValueError(f"unknown rounding mode {mode!r}")
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, generator=gen, dtype=torch.int32, device=bits.device)
    # Adding to the raw bits grows the magnitude for either sign; masking truncates the magnitude.
    return ((bits + noise) & ~0xFFFF).view(torch.float32)


def fp16(x: torch.Tensor) -> torch.Tensor:
    return x.float().to(torch.float16).float()


def e8m(x: torch.Tensor, m: int) -> torch.Tensor:
    """Round float32 to m stored mantissa bits and 8 exponent bits (round to nearest, ties to even), as float32.
    m = 7 is bf16; m < 7 gives coarser float grids with the same binades (relative step 2^-m)."""
    if not 0 <= m <= 23:
        raise ValueError(f"mantissa bits must be in [0, 23], got {m}")
    if m == 23:
        return x.float()
    bits = x.float().contiguous().view(torch.int32)
    drop = 23 - m
    lsb = (bits >> drop) & 1                                          # ties to even
    bits = (bits + ((1 << (drop - 1)) - 1) + lsb) & ~((1 << drop) - 1)
    return bits.view(torch.float32)


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of bfloat16 values at |x| (7 stored mantissa bits); normal range only."""
    e = torch.floor(torch.log2(x.float().abs().clamp_min(torch.finfo(torch.float32).tiny)))
    return torch.exp2(e - 7)


def int_scale(w: torch.Tensor, bits: int, group: int | None) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    g, _ = _grouped(w.float(), group)
    return (g.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-12)


def int_quant(w: torch.Tensor, bits: int, group: int | None = None, mode: str = "nearest",
              scale: torch.Tensor | None = None, gen: torch.Generator | None = None,
              jitter: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT-`bits` with levels [-qmax, qmax]. Returns (dequantized float32, scale [out, n_groups, 1]).

    scale: reuse a grid (frozen-grid mode). jitter: multiply a freshly computed scale by (1 + U[0, jitter)) per
    group -- a valid INT format that re-randomises every entry's position inside its cell."""
    qmax = 2 ** (bits - 1) - 1
    g, pad = _grouped(w.float(), group)
    if scale is None:
        scale = int_scale(w, bits, group)
        if jitter:
            scale = scale * (1 + jitter * torch.rand(scale.shape, generator=gen, dtype=scale.dtype,
                                                     device=scale.device))
    q = _round(g / scale, mode, gen).clamp(-qmax, qmax) * scale
    return _ungrouped(q, pad), scale


def fp8_scale(w: torch.Tensor, group: int | None = None) -> torch.Tensor:
    g, _ = _grouped(w.float(), group)
    return (g.abs().amax(dim=-1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-12)


_E4M3 = None


def _e4m3_levels() -> torch.Tensor:
    global _E4M3
    if _E4M3 is None:
        v = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        v = v[torch.isfinite(v)].unique()
        _E4M3 = v.sort().values
    return _E4M3


def fp8_quant(w: torch.Tensor, group: int | None = None, mode: str = "nearest",
              scale: torch.Tensor | None = None, gen: torch.Generator | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 with a per-row (or per-group) float32 scale mapping the row max to 448."""
    g, pad = _grouped(w.float(), group)
    if scale is None:
        scale = fp8_scale(w, group)
    y = (g / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    if mode == "nearest":
        q = y.to(torch.float8_e4m3fn).float()
    else:
        q = _round_to_levels(y, _e4m3_levels(), mode, gen)
    return _ungrouped(q * scale, pad), scale


def _e2m1(y: torch.Tensor, mode: str, gen: torch.Generator | None) -> torch.Tensor:
    levels = torch.cat([-_E2M1_POS.flip(0)[:-1], _E2M1_POS])         # -6 ... 0 ... 6 (15 values)
    mant_even = torch.tensor([True, False, True, False, True, False, True, False])   # 0,.5,1,1.5,2,3,4,6
    even = torch.cat([mant_even.flip(0)[:-1], mant_even])
    return _round_to_levels(y, levels, mode, gen, even=even)


# bitsandbytes NF4 (QLoRA, Dettmers et al. 2023): the 16 normal-quantile levels of bnb's lookup table
NF4_LEVELS = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                           -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                           0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                           0.7229568362236023, 1.0])


def nf4_quant(w: torch.Tensor, block: int = 64, mode: str = "nearest", gen: torch.Generator | None = None
              ) -> torch.Tensor:
    """bitsandbytes NF4 as QLoRA stores a base: blocks of 64 consecutive entries (row-local for the widths here), an
    absmax scale per block (kept in float32: no double quantization), each entry rounded to the nearest level."""
    g, pad = _grouped(w.float(), block)
    amax = g.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax, torch.ones_like(amax))
    return _ungrouped(_round_to_levels(g / scale, NF4_LEVELS, mode, gen) * scale, pad)


def nvfp4_quant(w: torch.Tensor, mode: str = "nearest", gen: torch.Generator | None = None
                ) -> torch.Tensor:
    """NVFP4 emulation: E2M1 elements in blocks of 16, E4M3 block scales, one float32 tensor scale."""
    x = w.float()
    g, pad = _grouped(x, 16)
    tensor_scale = (x.abs().max() / (FP8_E4M3_MAX * E2M1_MAX)).clamp_min(1e-30)
    block = g.abs().amax(dim=-1, keepdim=True) / (E2M1_MAX * tensor_scale)
    block = block.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    scale = (block * tensor_scale).clamp_min(1e-30)
    q = _e2m1(g / scale, mode, gen) * scale
    return _ungrouped(q, pad)


def mxfp4_quant(w: torch.Tensor, mode: str = "nearest", gen: torch.Generator | None = None) -> torch.Tensor:
    """OCP MXFP4 emulation: E2M1 elements in blocks of 32 sharing a power-of-two (E8M0) scale."""
    g, pad = _grouped(w.float(), 32)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - 2)             # E2M1's largest exponent is 2 (6 = 1.5 * 2^2)
    q = _e2m1(g / scale, mode, gen) * scale
    return _ungrouped(q, pad)


def fp4_quant_with_scale(w: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """nvfp4_quant / mxfp4_quant (identical values), also returning the block scales [out, n_blocks, 1] and the pad,
    so that an update can be rounded onto the same grid (an on-grid merge into a native FP4 checkpoint)."""
    x = w.float()
    if fmt == "nvfp4":
        g, pad = _grouped(x, 16)
        tensor_scale = (x.abs().max() / (FP8_E4M3_MAX * E2M1_MAX)).clamp_min(1e-30)
        block = g.abs().amax(dim=-1, keepdim=True) / (E2M1_MAX * tensor_scale)
        block = block.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
        scale = (block * tensor_scale).clamp_min(1e-30)
    elif fmt == "mxfp4":
        g, pad = _grouped(x, 32)
        amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        scale = torch.exp2(torch.floor(torch.log2(amax)) - 2)
    else:
        raise ValueError(f"not an FP4 format: {fmt!r}")
    return _ungrouped(_e2m1(g / scale, "nearest", None) * scale, pad), scale, pad


def fp4_on_scale(x: torch.Tensor, fmt: str, scale: torch.Tensor, mode: str = "nearest",
                 gen: torch.Generator | None = None) -> torch.Tensor:
    """Round x onto a fixed FP4 grid (block scales from fp4_quant_with_scale); values past +-6 * scale saturate.
    mode="stochastic": between the two neighbouring codes (an SR merge onto the frozen grid)."""
    g, pad = _grouped(x.float(), 16 if fmt == "nvfp4" else 32)
    y = (g / scale).clamp(-E2M1_MAX, E2M1_MAX)
    return _ungrouped(_e2m1(y, mode, gen) * scale, pad)


# ---------------------------------------------------------------------------------------------------- local step

def local_step(w: torch.Tensor, fmt: str, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Grid spacing at each entry of w (the s in |dW|/s), for the formats with a uniform or binade grid."""
    w = w.float()
    if fmt == "bf16":
        return bf16_ulp(w)
    if fmt == "fp16":
        return bf16_ulp(w) / 8                                          # 10 vs 7 stored mantissa bits
    if fmt.startswith("e8m"):
        return bf16_ulp(w) * 2.0 ** (7 - int(fmt[3:]))                  # m stored mantissa bits
    if fmt.startswith("int"):
        bits, group = parse_int(fmt)
        if scale is None:
            scale = int_scale(w, bits, group)
        g, pad = _grouped(torch.ones_like(w), group)
        return _ungrouped(g * scale, pad)
    if fmt == "fp8":
        if scale is None:
            scale = fp8_scale(w)
        g, pad = _grouped(w, None)
        y = (g / scale).abs().clamp_min(2.0 ** -6)                       # E4M3 subnormal spacing below 2^-6
        return _ungrouped(torch.exp2(torch.floor(torch.log2(y)) - 3) * scale, pad)
    raise ValueError(f"no local step for {fmt!r}")


def parse_int(fmt: str) -> tuple[int, int | None]:
    """'int8' -> (8, None); 'int4g128' -> (4, 128)."""
    body = fmt[3:]
    if "g" in body:
        b, g = body.split("g")
        return int(b), int(g)
    return int(body), None
