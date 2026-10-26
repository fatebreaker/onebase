"""ICML-E1B-FMT (docs/REGISTRY.md): where does stochastic rounding stop reducing a sampler's bias? ICML-E1B-TF (e4m3 activations
per token) found SR's noise-averaged sampler at 0.17-0.37x round to nearest's KL; ICML-E1B-FP4 (NVFP4, weights too) at
1.03-1.50x. Here only the activations' format changes, on bench/bias_noise_tf.py's path: the weight-matched model (vLLM's FP8
weight codes, mlsys.matched_fp8 parts="w"), the same contexts and estimators, every FP8 layer's input rounded to a block-scaled
floating-point grid before its GEMM (the LoRA branch sees the unrounded input):
  per token, a float32 scale mapping the token's max to 448 x the format's max; per block of 16 along K, an E4M3 scale (the
  NVFP4 recipe); elements on the format's grid, to nearest (ties to even) or stochastically (fresh noise per forward)
  --fmt e4m3 (FP8, 3 mantissa bits), e2m3 (FP6, 3), e3m2 (FP6, 2), e2m1 (FP4, 1): OCP formats without inf/nan except e4m3fn's.
    PYTHONPATH=$LB/pylib_fla:$LB:$ML python bench/bias_noise_tf_fmt.py --fmt e3m2 --model M --adapter A \
        --data results/icml_e1_bf16_65181900.json --K 32 --label e1bf --out results/bias_noise_tf_fmt.jsonl
CPU check of the quantizer: python bench/bias_noise_tf_fmt.py --selftest"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))

MODE = {"m": "none", "fmt": "e4m3"}
FMTS = {"e4m3": (4, 3, 448.0), "e2m3": (2, 3, 7.5), "e3m2": (3, 2, 28.0), "e2m1": (2, 1, 6.0)}   # exponent bits, mantissa bits, max
_LEVELS: dict = {}


def levels(fmt: str):
    """The format's representable values, sorted (negatives, 0, positives), up to its max."""
    import torch
    if fmt not in _LEVELS:
        E, M, mx = FMTS[fmt]
        bias = 2 ** (E - 1) - 1
        pos = {0.0}
        for e in range(2 ** E):
            for m in range(2 ** M):
                v = (m / 2 ** M) * 2.0 ** (1 - bias) if e == 0 else (1 + m / 2 ** M) * 2.0 ** (e - bias)
                if v <= mx:
                    pos.add(v)
        p = sorted(pos)
        _LEVELS[fmt] = torch.tensor([-x for x in reversed(p[1:])] + p, dtype=torch.float32)
    return _LEVELS[fmt]


def fq_rows(x2, fmt: str, mode: str):
    """Block-scaled round trip of x2 [T, K] (see the module doc); ties to even on the element grid."""
    import torch
    from mlsys.quant import _round_to_levels
    mx = FMTS[fmt][2]
    x = x2.float()
    T, K = x.shape
    pad = (-K) % 16
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    g = x.reshape(T, -1, 16)
    ts = (x.abs().amax(dim=-1) / (448.0 * mx)).clamp_min(1e-30)[:, None, None]
    block = (g.abs().amax(dim=-1, keepdim=True) / (mx * ts)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    scale = (block * ts).clamp_min(1e-30)
    lv = levels(fmt).to(x.device)
    even = torch.zeros(len(lv), dtype=torch.bool)                  # ties to the level with an even code (mantissa lsb 0)
    n = len(lv)
    for i in range(n):
        even[i] = (abs(i - (n - 1) // 2) % 2) == 0
    q = _round_to_levels(g / scale, lv, "nearest" if mode == "rtn" else "stochastic", None, even=even) * scale
    q = q.reshape(T, -1)
    return (q[:, :K] if pad else q).to(torch.bfloat16)


def hook(module, args):
    if MODE["m"] == "none":
        return None
    x = args[0]
    x2 = x.reshape(-1, x.shape[-1])
    return (fq_rows(x2, MODE["fmt"], MODE["m"]).reshape(x.shape),) + tuple(args[1:])


def selftest() -> int:
    import torch
    torch.manual_seed(0)
    assert [round(float(v), 4) for v in levels("e2m1") if v >= 0] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert float(levels("e3m2").max()) == 28.0 and float(levels("e2m3").max()) == 7.5 and float(levels("e4m3").max()) == 448.0
    x = torch.randn(4, 96) * torch.tensor([1.0, 0.01, 30.0, 3.0])[:, None]
    for fmt in FMTS:
        rtn = fq_rows(x, fmt, "rtn").float()
        S = 1000
        mean = sum(fq_rows(x, fmt, "sr").float() for _ in range(S)) / S
        e_sr = float((mean - x).abs().mean() / x.abs().mean())
        e_rn = float((rtn - x).abs().mean() / x.abs().mean())
        print(f"{fmt}: relative mean |E[q] - x| SR {e_sr:.4f}, round to nearest {e_rn:.4f}")
        assert e_sr < 0.5 * e_rn, fmt
    print("ok")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    import bias_noise_tf as B                                      # its main(), with this module's hook and format
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fmt", required=True, choices=tuple(FMTS))
    a, rest = ap.parse_known_args()
    MODE["fmt"] = a.fmt
    B.MODE = MODE                                                  # bias_noise_tf's hook reads B.MODE; replace the hook itself
    B.hook = hook
    sys.argv = [sys.argv[0]] + rest
    import json
    out_i = rest.index("--out") + 1 if "--out" in rest else None
    rc = B.main()
    if out_i:                                                      # tag the row bias_noise_tf.py appended with the format
        f = Path(rest[out_i])
        lines = f.read_text().splitlines()
        row = json.loads(lines[-1])
        row["act_format"] = f"{a.fmt}, blocks of 16 with E4M3 scales, a float32 scale per token"
        lines[-1] = json.dumps(row)
        f.write_text("\n".join(lines) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
