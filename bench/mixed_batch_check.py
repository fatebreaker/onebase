"""Mixed-adapter batch check (MLSys v2 mock review: S9B's batched jobs degraded late with a rising trainer-engine gap; verify that
every request in a batch of several adapters is served with its own adapter). One engine (vLLM, online FP8, LoRA with
--max-loras = the adapter count, CUDA graphs) per process; LOWBIT_FUSED selects the plugin (1) or vLLM's own LoRA path (0).
  prefill  each fixed text (a DAPO prompt plus a fixed continuation) scored under each adapter with prompt_logprobs, all
           (text, adapter) pairs in one call (mixed) and one call per adapter (alone): per-token |logp difference|
  decode   greedy 64 new tokens per (prompt, adapter), mixed and alone: share of identical outputs, first divergence
Also the spread between adapters (alone), so a misrouted request is distinguishable from batch-composition noise.
Writes one JSON per run (--out); compare the plugin's mixed-vs-alone numbers with the stock path's.
    LOWBIT_FUSED=1 python bench/mixed_batch_check.py --model <dir> --adapters a0 a1 a2 --out results/mixed_batch_plugin.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

DATA = Path("$SCRATCH/data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=24, help="texts / prompts")
    ap.add_argument("--quantization", default="fp8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rng = random.Random(0)
    qs = []
    pq = DATA / "dapo_dl/en/train.parquet"                     # the RL runs' training problems (train/grpo_q.py --task dapo)
    if pq.exists():
        import pandas as pd
        df = pd.read_parquet(pq)
        idx = list(range(len(df)))
        rng.shuffle(idx)
        qs = [str(df.iloc[i]["prompt"]) for i in idx[: a.n]]
    if len(qs) < a.n:
        qs += [f"Compute {i * 7 + 3} times {i + 11}, then add {i * i}. Put the final answer in \\boxed{{}}." for i in range(a.n - len(qs))]
    L = len(a.adapters)
    llm = LLM(model=a.model, quantization=a.quantization or None, enable_lora=True, max_lora_rank=64, max_loras=L, max_cpu_loras=L,
              max_model_len=4096, gpu_memory_utilization=0.85, enable_prefix_caching=False, seed=0)
    tok = llm.get_tokenizer()
    reqs = [LoRARequest(f"ad{k}", k + 1, p) for k, p in enumerate(a.adapters)]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for q in qs]
    cont = ("Let me work through this step by step. First, I identify the quantities involved and write down what is given. "
            "Then I compute each part carefully and check the arithmetic before stating the result. ")
    texts = [p + cont for p in prompts]
    out: dict = {"model": a.model, "adapters": a.adapters, "lowbit_fused": os.environ.get("LOWBIT_FUSED", "0"), "n": len(texts),
                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # prefill: prompt logprobs of each text under each adapter
    sp_pl = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    def plp(results):
        return [[next(iter(d.values())).logprob if d else None for d in (o.prompt_logprobs or [])] for o in results]
    pairs = [(i, k) for k in range(L) for i in range(len(texts))]
    rng.shuffle(pairs)                                           # interleave the adapters in the batch
    mixed = plp(llm.generate([texts[i] for i, _ in pairs], sp_pl, lora_request=[reqs[k] for _, k in pairs], use_tqdm=False))
    mixed_by = {pk: v for pk, v in zip(pairs, mixed)}
    alone_by = {}
    for k in range(L):
        res = plp(llm.generate(texts, sp_pl, lora_request=[reqs[k]] * len(texts), use_tqdm=False))
        for i, v in enumerate(res):
            alone_by[(i, k)] = v

    def mad(x, y):
        d = [abs(p - q) for p, q in zip(x, y) if p is not None and q is not None]
        return statistics.fmean(d) if d else None
    out["prefill_mixed_vs_alone"] = {f"ad{k}": statistics.fmean(mad(mixed_by[(i, k)], alone_by[(i, k)]) for i in range(len(texts)))
                                     for k in range(L)}
    out["prefill_between_adapters"] = {f"ad{k}-ad{k + 1}": statistics.fmean(mad(alone_by[(i, k)], alone_by[(i, k + 1)])
                                                                             for i in range(len(texts))) for k in range(L - 1)}
    out["prefill_worst_request"] = max(mad(mixed_by[pk], alone_by[pk]) for pk in pairs)
    # decode: greedy 64 tokens
    sp_g = SamplingParams(max_tokens=64, temperature=0.0, logprobs=0)
    gm = llm.generate([prompts[i] for i, _ in pairs], sp_g, lora_request=[reqs[k] for _, k in pairs], use_tqdm=False)
    gm_by = {pk: list(o.outputs[0].token_ids) for pk, o in zip(pairs, gm)}
    ga_by = {}
    for k in range(L):
        for i, o in enumerate(llm.generate(prompts, sp_g, lora_request=[reqs[k]] * len(prompts), use_tqdm=False)):
            ga_by[(i, k)] = list(o.outputs[0].token_ids)

    def first_div(x, y):
        for j, (p, q) in enumerate(zip(x, y)):
            if p != q:
                return j
        return None if len(x) == len(y) else min(len(x), len(y))
    same = {f"ad{k}": sum(gm_by[(i, k)] == ga_by[(i, k)] for i in range(len(prompts))) / len(prompts) for k in range(L)}
    out["decode_identical_share"] = same
    divs = [first_div(gm_by[pk], ga_by[pk]) for pk in pairs]
    out["decode_first_divergence_median"] = statistics.median([d for d in divs if d is not None]) if any(d is not None for d in divs) else None
    out["decode_between_adapters_identical_share"] = {f"ad{k}-ad{k + 1}": sum(ga_by[(i, k)] == ga_by[(i, k + 1)] for i in range(len(prompts))) / len(prompts)
                                                      for k in range(L - 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
