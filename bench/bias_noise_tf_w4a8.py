"""ICML-E1B-W4A8 (docs/REGISTRY.md): does stochastic rounding's bias reduction at FP8 activations survive 4-bit weights?
ICML-E1B-TF (FP8 weight codes, e4m3 activations per token) found SR's noise-averaged sampler at 0.17-0.37x round to nearest's
KL to the trainer; ICML-E1B-FP4 (NVFP4 weights AND activations) at 1.03-1.50x. Here the weights are NVFP4 (bias_noise_tf_fp4.py's
p: every quantized linear layer's weight rounded to NVFP4, bf16 activations) and the activations FP8, as bias_noise_tf.py rounds
them (per token e4m3, vLLM's scale rule; round to nearest, or stochastically with fresh noise per forward via
lowbit_lora/w16_engine_fqsr.sr_rows; the LoRA branch sees the unrounded input): a W4A8 sampler against its own trainer.
bias_noise_tf_fp4.main() runs unchanged with bias_noise_tf's hook; the row it appends is tagged with this format.
    PYTHONPATH=$LB/pylib_fla:$LB:$ML python bench/bias_noise_tf_w4a8.py --model M --adapter A \
        --data results/icml_e1_bf16_65181900.json --K 32 --label e1bw4a8 --out results/bias_noise_tf_w4a8.jsonl
CPU check of the activation quantizers: python bench/bias_noise_tf_w4a8.py --selftest"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
sys.path.insert(0, str(LB))


def selftest() -> int:
    import torch

    import bias_noise_tf as B
    from lowbit_lora.w16_engine_fqsr import sr_rows
    torch.manual_seed(0)
    x = (torch.randn(4, 96) * torch.tensor([1.0, 0.01, 30.0, 3.0])[:, None]).to(torch.bfloat16)
    rtn = B.fq_rtn(x).float()
    S = 2000
    mean = sum(sr_rows(x).float() for _ in range(S)) / S
    xf = x.float()
    e_sr = float((mean - xf).abs().mean() / xf.abs().mean())
    e_rn = float((rtn - xf).abs().mean() / xf.abs().mean())
    print(f"e4m3 per token: relative mean |E[q] - x| SR {e_sr:.5f}, round to nearest {e_rn:.5f}")
    assert e_sr < 0.5 * e_rn
    print("ok")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    import bias_noise_tf as B
    import bias_noise_tf_fp4 as F
    F.MODE = B.MODE                     # F.main's logp sets MODE["m"]; B.hook reads B.MODE: one dict
    F.hook = B.hook                     # F.main registers the global `hook` on every quantized layer
    out_i = sys.argv.index("--out") + 1 if "--out" in sys.argv else None
    rc = F.main()
    if out_i:                           # tag the row F.main appended
        f = Path(sys.argv[out_i])
        lines = f.read_text().splitlines()
        row = json.loads(lines[-1])
        row["format"] = "W4A8: NVFP4 weights (E2M1, E4M3 scales per 16, a float32 tensor scale), e4m3 activations per token"
        lines[-1] = json.dumps(row)
        f.write_text("\n".join(lines) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
