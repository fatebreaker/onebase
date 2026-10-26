"""Does lowbit_lora/w16_engine_fqsr.py draw fresh stochastic-rounding noise at every CUDA-graph replay? (ICML-E2-PILOT,
amendment -GRAPHS.) Its noise comes from torch.rand inside a custom op; PyTorch's graph-safe RNG should advance the
generator's offset at every replay, but if the noise were captured once, every replay would round the same way (a fixed
perturbation: biased, not stochastic rounding). Generates the same prompts R times greedily with per-token log-probabilities,
with CUDA graphs and, as the control, eager; reports, per mode, the fraction of repeats whose token sequence differs from the
first and the max |d logprob| of the first generated token across repeats (0 means frozen noise).
    VLLM_ENABLE_V1_MULTIPROCESSING=0 LOWBIT_FUSED=0 python bench/fqsr_graph_check.py --model M [--mode graphs|eager]"""
from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="graphs", choices=("graphs", "eager"))
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--engine", default="sr", choices=("sr", "rtn"), help="rtn: w16_engine_fq, the deterministic control")
    a = ap.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if a.engine == "sr":
        from lowbit_lora import w16_engine_fqsr as eng
    else:
        from lowbit_lora import w16_engine_fq as eng
    eng.install()
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, quantization="fp8", enforce_eager=(a.mode == "eager"), max_model_len=1024, gpu_memory_utilization=0.5,
              max_num_seqs=8, seed=0)
    prompts = ["The derivative of x^3 + 2x is", "List the first ten prime numbers:", "A train leaves at 3 pm at 60 km/h. How far",
               "Explain why the sky is blue in one sentence:"]
    sp = SamplingParams(temperature=0.0, max_tokens=48, logprobs=1)
    runs = []
    for _ in range(a.repeats):
        outs = llm.generate(prompts, sp, use_tqdm=False)
        runs.append([(list(o.outputs[0].token_ids), [next(iter(lp.values())).logprob for lp in o.outputs[0].logprobs])
                     for o in outs])
    differ = sum(runs[r][i][0] != runs[0][i][0] for r in range(1, a.repeats) for i in range(len(prompts)))
    dlp = max(abs(runs[r][i][1][0] - runs[0][i][1][0]) for r in range(1, a.repeats) for i in range(len(prompts)))
    row = {"engine": a.engine, "mode": a.mode, "repeats": a.repeats, "prompts": len(prompts), "sequences_differing": differ,
           "of": (a.repeats - 1) * len(prompts), "max_abs_dlogprob_first_token": dlp, "stats": eng.stats}
    print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
