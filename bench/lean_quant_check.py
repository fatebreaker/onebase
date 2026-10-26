"""Diagnostic (not registered data): is LOWBIT_LEAN_BASE's quantization vLLM's? vLLM's per-token dynamic FP8 quantization as its
linear layers run it eagerly (QuantFP8.forward_native under torch.compile, inductor) against the CUDA op the lean base calls
(ops.scaled_fp8_quant, per token), on bf16 activations of the 9B's projection widths; then each through vLLM's CUTLASS GEMM on
one random per-tensor FP8 weight. Prints, per shape, the share of identical codes, the largest relative scale gap, and the
largest GEMM output gap relative to the output's RMS.
    python bench/lean_quant_check.py [--out results/lean_quant_check.jsonl]"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):                   # a CustomOp reads the current config when built
        q = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN, compile_native=False)
    native = torch.compile(q.forward_native, dynamic=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    for M in (33, 64, 512, 2048, 4096):
        for K, N in ((4096, 6144), (4096, 24576), (12288, 4096)):
            x = (torch.randn(M, K, device="cuda", generator=g) * torch.rand(M, 1, device="cuda", generator=g) * 4).bfloat16()
            x[0, :7] = 0.0
            w = (torch.randn(N, K, device="cuda", generator=g) * 0.02)
            sw = (w.abs().max() / 448.0).float().reshape(1)
            wq = (w / sw).clamp(-448, 448).to(torch.float8_e4m3fn).t()          # [K, N], column-major as vLLM stores it
            q1, s1 = native(x)
            q2, s2 = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
            same = (q1.view(torch.uint8) == q2.view(torch.uint8)).float().mean().item()
            sgap = ((s1.float() - s2.float()).abs() / s1.float().clamp_min(1e-30)).max().item()
            y1 = ops.cutlass_scaled_mm(q1, wq, scale_a=s1, scale_b=sw, out_dtype=torch.bfloat16)
            y2 = ops.cutlass_scaled_mm(q2, wq, scale_a=s2, scale_b=sw, out_dtype=torch.bfloat16)
            rms = y1.float().pow(2).mean().sqrt().item()
            ygap = (y1.float() - y2.float()).abs().max().item() / rms
            r = {"M": M, "K": K, "N": N, "codes_identical": same, "scale_rel_gap_max": sgap, "out_gap_max_over_rms": ygap,
                 "outputs_identical": bool(torch.equal(y1, y2))}
            rows.append(r)
            print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "gpu": torch.cuda.get_device_name()}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
