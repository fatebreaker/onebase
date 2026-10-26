"""Diagnostic (not registered data): LOWBIT_LEAN_BASE parity. Greedy outputs and log-probs of one engine setting (from the
environment, as for bench/prefill_profile.py) on fixed random prompts long enough that prefill steps exceed vLLM's CUDA-graph
sizes (so the plugin's hybrid paths run eagerly), each prompt with one of N resident adapters. One JSON line per run;
--compare LABEL_A LABEL_B prints how many prompts' outputs match token for token and the largest log-prob gap on the
common prefix.
    python bench/lean_parity.py --model <dir> --adapter <dir> --label <mode> --out <jsonl>
    python bench/lean_parity.py --out <jsonl> --compare <label a> <label b>"""
from __future__ import annotations

import argparse
import json
import os
import random


def compare(out: str, la: str, lb: str) -> int:
    runs = {}
    for line in open(out):
        r = json.loads(line)
        runs[r["label"]] = r
    A, B = runs[la]["outputs"], runs[lb]["outputs"]
    same, gap, first = 0, 0.0, []
    for x, y in zip(A, B):
        n = 0
        while n < min(len(x["ids"]), len(y["ids"])) and x["ids"][n] == y["ids"][n]:
            n += 1
        same += n == len(x["ids"]) == len(y["ids"])
        first.append(n)
        gap = max([gap] + [abs(p - q) for p, q in zip(x["lp"][:n], y["lp"][:n])])
    print(json.dumps({"a": la, "b": lb, "prompts": len(A), "identical": same, "first_divergence_min": min(first),
                      "max_logprob_gap_on_common_prefix": gap}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-adapters", type=int, default=8)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--label")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        return compare(a.out, *a.compare)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, quantization="fp8", enable_lora=True, max_lora_rank=a.rank, max_loras=a.n_adapters,
              max_cpu_loras=a.n_adapters, max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=False)
    reqs = [LoRARequest(f"a{i}", i + 1, a.adapter) for i in range(a.n_adapters)]
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(100, 30000) for _ in range(a.prompt_len)])
               for _ in range(a.prompts)]
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0, logprobs=0, ignore_eos=True)
    res = llm.generate(prompts, sp, lora_request=[reqs[i % a.n_adapters] for i in range(a.prompts)], use_tqdm=False)
    outs = []
    for r in res:
        o = r.outputs[0]
        outs.append({"ids": list(o.token_ids), "lp": [d[t].logprob for t, d in zip(o.token_ids, o.logprobs)]})
    with open(a.out, "a") as fh:
        fh.write(json.dumps({"label": a.label, "lean": os.environ.get("LOWBIT_LEAN_BASE", "0"), "outputs": outs}) + "\n")
    print(json.dumps({"label": a.label, "prompts": len(outs), "tokens": sum(len(o["ids"]) for o in outs)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
