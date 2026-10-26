"""Mixed-adapter batch check, version 2 (diagnostic; MLSys v2 mock review: S9B's batched jobs degraded late with a rising
trainer-engine gap; verify that every request in a batch of several adapters is served with its own adapter).
Version 1 (bench/mixed_batch_check.py, 2026-10-17) compared a mixed batch with each adapter served alone; batch size and
chunking differ between the two, so its differences mix batch-composition noise with any cross-adapter effect, and one
plugin request differed by 1.39 nats per token, inside the range between adapters. Version 2 adds a control with the
mixed batch's exact layout: the same requests in the same order, every one assigned adapter k. Rows are computed
independently of each other in every kernel on both paths (per-token activation quantization; GEMM rows, LoRA segments),
and the scheduler sees the same token counts, so the rows that carry adapter k in the mixed batch must match the control
bit for bit unless other adapters' requests leak into them. Decode uses ignore_eos so the layout stays the same at every
step. Also per request: the adapter whose alone result is nearest to the mixed result (a misrouted request sits nearest
another adapter's).
One engine (vLLM, online FP8, LoRA with --max-loras = the adapter count, CUDA graphs) per process; LOWBIT_FUSED selects the
plugin (1) or vLLM's own LoRA path (0).
    LOWBIT_FUSED=1 python bench/mixed_batch_check2.py --model <dir> --adapters a0 a1 a2 --out results/mixed_batch2_plugin.json
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
    ap.add_argument("--max-tokens", type=int, default=64)
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
                 "version": 2, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pairs = [(i, k) for k in range(L) for i in range(len(texts))]
    rng.shuffle(pairs)                                           # interleave the adapters in the batch

    def mad(x, y):
        d = [abs(p - q) for p, q in zip(x, y) if p is not None and q is not None]
        return statistics.fmean(d) if d else None

    # ---- prefill: prompt logprobs
    sp_pl = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    def plp(results):
        return [[next(iter(d.values())).logprob if d else None for d in (o.prompt_logprobs or [])] for o in results]
    mixed = plp(llm.generate([texts[i] for i, _ in pairs], sp_pl, lora_request=[reqs[k] for _, k in pairs], use_tqdm=False))
    ctrl = {k: plp(llm.generate([texts[i] for i, _ in pairs], sp_pl, lora_request=[reqs[k]] * len(pairs), use_tqdm=False))
            for k in range(L)}
    alone = {}
    for k in range(L):
        for i, v in enumerate(plp(llm.generate(texts, sp_pl, lora_request=[reqs[k]] * len(texts), use_tqdm=False))):
            alone[(i, k)] = v
    # mixed vs the same-layout control: an exact match expected
    pos_k = {k: [p for p, (_, kk) in enumerate(pairs) if kk == k] for k in range(L)}
    out["prefill_mixed_eq_control_share"] = {f"ad{k}": sum(mixed[p] == ctrl[k][p] for p in pos_k[k]) / len(pos_k[k]) for k in range(L)}
    out["prefill_mixed_vs_control_mean"] = {f"ad{k}": statistics.fmean(mad(mixed[p], ctrl[k][p]) for p in pos_k[k]) for k in range(L)}
    out["prefill_mixed_vs_control_worst"] = max(mad(mixed[p], ctrl[k][p]) for k in range(L) for p in pos_k[k])
    # mixed vs alone (version 1's comparison), per request, with the nearest adapter
    rows = []
    for p, (i, k) in enumerate(pairs):
        d = [mad(mixed[p], alone[(i, j)]) for j in range(L)]
        dc = [mad(mixed[p], ctrl[j][p]) for j in range(L)]
        rows.append({"pos": p, "text": i, "adapter": k, "vs_alone": d, "vs_control": dc,
                     "nearest_alone": min(range(L), key=lambda j: d[j]), "nearest_control": min(range(L), key=lambda j: dc[j])})
    out["prefill_requests"] = rows
    out["prefill_nearest_other"] = [r for r in rows if r["nearest_alone"] != r["adapter"] or r["nearest_control"] != r["adapter"]]
    own = sorted(r["vs_alone"][r["adapter"]] for r in rows)
    out["prefill_mixed_vs_alone_mean"] = {f"ad{k}": statistics.fmean(r["vs_alone"][k] for r in rows if r["adapter"] == k) for k in range(L)}
    out["prefill_mixed_vs_alone_quantiles"] = {str(q): own[min(len(own) - 1, int(q * len(own)))] for q in (0.5, 0.9, 0.99)}
    out["prefill_mixed_vs_alone_worst"] = own[-1]
    out["prefill_between_adapters"] = {f"ad{j}-ad{k}": statistics.fmean(mad(alone[(i, j)], alone[(i, k)]) for i in range(len(texts)))
                                       for j in range(L) for k in range(j + 1, L)}
    out["prefill_control_vs_alone_mean"] = {f"ad{k}": statistics.fmean(mad(ctrl[k][p], alone[(pairs[p][0], k)]) for p in range(len(pairs)))
                                            for k in range(L)}
    # ---- decode: greedy, fixed length (ignore_eos keeps the batch layout the same at every step)
    sp_g = SamplingParams(max_tokens=a.max_tokens, temperature=0.0, ignore_eos=True)

    def ids(results):
        return [list(o.outputs[0].token_ids) for o in results]
    gm = ids(llm.generate([prompts[i] for i, _ in pairs], sp_g, lora_request=[reqs[k] for _, k in pairs], use_tqdm=False))
    gc = {k: ids(llm.generate([prompts[i] for i, _ in pairs], sp_g, lora_request=[reqs[k]] * len(pairs), use_tqdm=False))
          for k in range(L)}
    ga = {}
    for k in range(L):
        for i, v in enumerate(ids(llm.generate(prompts, sp_g, lora_request=[reqs[k]] * len(prompts), use_tqdm=False))):
            ga[(i, k)] = v

    def lcp(x, y):
        for j, (p, q) in enumerate(zip(x, y)):
            if p != q:
                return j
        return min(len(x), len(y))
    out["decode_mixed_eq_control_share"] = {f"ad{k}": sum(gm[p] == gc[k][p] for p in pos_k[k]) / len(pos_k[k]) for k in range(L)}
    out["decode_mixed_eq_alone_share"] = {f"ad{k}": sum(gm[p] == ga[(pairs[p][0], k)] for p in pos_k[k]) / len(pos_k[k]) for k in range(L)}
    out["decode_control_eq_alone_share"] = {f"ad{k}": sum(gc[k][p] == ga[(pairs[p][0], k)] for p in range(len(pairs))) / len(pairs)
                                            for k in range(L)}
    dec = []
    for p, (i, k) in enumerate(pairs):
        lc = [lcp(gm[p], gc[j][p]) for j in range(L)]
        if lc[k] < a.max_tokens:
            dec.append({"pos": p, "prompt": i, "adapter": k, "lcp_control": lc, "lcp_alone": [lcp(gm[p], ga[(i, j)]) for j in range(L)]})
    out["decode_mixed_ne_control"] = dec
    out["decode_between_adapters_identical_share"] = {f"ad{j}-ad{k}": sum(ga[(i, j)] == ga[(i, k)] for i in range(len(prompts))) / len(prompts)
                                                      for j in range(L) for k in range(j + 1, L)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    summ = {k: v for k, v in out.items() if k not in ("prefill_requests",)}
    print(json.dumps(summ)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
