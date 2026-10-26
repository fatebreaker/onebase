"""bench/rl_rollout_bench2.py (kept unchanged) with --prompts dapo (RL-ROLLOUT-QMATH): N DAPO-Math-17k problems (as
RL-FP8-QMATH trains on, seeded shuffle) through the model's chat template in Qwen's non-thinking mode, with the
boxed-answer instruction. Otherwise rl_rollout_bench2.py's doc:

bench/rl_rollout_bench.py (kept unchanged) with --prompts math: the prompts are N MATH questions (MetaMathQA's
MATH_AnsAug, as RL-FP8-MATH trains on, seeded shuffle) through the model's chat template with the boxed-answer
instruction, instead of agentic transcripts' first turns; everything else as before (RL-ROLLOUT-MATH).

Rollout throughput of LoRA RL's generation engine, replayed under fixed conditions (lowbit-lora P2, descriptive).

The G3 runs' step times mix engine speed with what each policy writes, so they cannot rank engines. Here every engine
generates the same tokens' worth of work: the first-turn prompts of N agentic transcripts (the policy's own task), one
RL adapter (rank 128), temperature 1, a fixed max_tokens with ignore_eos, all N at once. The engine settings are G3's
rollout server's: max_model_len 16384, max_num_seqs 48, and a GPU share that leaves room for the colocated trainer
(--gpu-util 0.31 by default), so memory that a smaller base frees for the KV cache counts, as it does in training.

    python bench/rl_rollout_bench.py --model <Qwen3.5-9B-text> --adapter <step100> --results <transcripts dir> \
        --label fp8_plugin [--fp8] --out results/rl_rollout_bench.jsonl
The plugin and its storage format come from the environment (LOWBIT_FUSED, LOWBIT_LORA_FP4), as in serving; run each
configuration in its own process (jobs/rl_rollout_bench.sbatch).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path


def prompts(tok, results: str, n: int) -> list[list[int]]:
    """Token ids of each transcript's prefix up to its first assistant turn (the rollout's first prompt)."""
    from mlsys.fidelity import build
    files = sorted(p for p in Path(results).glob("*.json") if p.name not in ("summary.json", "harness_v2.json"))
    random.Random(0).shuffle(files)
    out = []
    for f in files:
        rec = json.loads(f.read_text())
        if not rec.get("transcript"):
            continue
        ids, mask = build(tok, rec, 16384, "chatml")
        first = next((i for i, m in enumerate(mask) if m), None)
        if first and first >= 16:
            out.append(ids[:first])
        if len(out) >= n:
            break
    return out


def math_prompts(tok, n: int) -> list[list[int]]:
    """Token ids of N MATH questions with the generation prompt (RL-FP8-MATH's training distribution)."""
    sfx = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    rows, seen = [], set()
    data = json.loads(Path("$SCRATCH/data/std_bench_dl/metamathqa_40k.json").read_text())
    for r in data:
        if r["type"] == "MATH_AnsAug" and r["query"] not in seen:
            seen.add(r["query"])
            rows.append(r["query"])
    random.Random(0).shuffle(rows)
    return [tok.apply_chat_template([{"role": "user", "content": q + sfx}], add_generation_prompt=True, tokenize=True)
            for q in rows[:n]]


def dapo_prompts(tok, n: int) -> list[list[int]]:
    """Token ids of N DAPO-Math-17k problems, non-thinking chat mode (RL-FP8-QMATH's training distribution)."""
    import pandas as pd
    sfx = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    df = pd.read_parquet("$SCRATCH/data/dapo_dl/en/train.parquet")
    rows = list(df.prompt)
    random.Random(0).shuffle(rows)
    # the template rendered as text, then tokenized: transformers 5's apply_chat_template(tokenize=True) returns a
    # BatchEncoding, not the ids (RL-ROLLOUT-QMATH's first run, 10-16 00:00-00:37, failed on every configuration)
    return [tok(tok.apply_chat_template([{"role": "user", "content": q + sfx}], add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False), add_special_tokens=False)["input_ids"] for q in rows[:n]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="", help="empty: no LoRA")
    ap.add_argument("--adapters", nargs="*", default=[], help="several adapters, assigned round-robin to the prompts "
                    "(multi-tenant rollouts: K RL jobs sharing one engine); overrides --adapter")
    ap.add_argument("--results", default="", help="transcripts dir (--prompts transcripts)")
    ap.add_argument("--prompts", default="transcripts", choices=("transcripts", "math", "dapo"))
    ap.add_argument("--max-lora-rank", type=int, default=128)
    ap.add_argument("--label", required=True)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.31)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--linear-backend", default="auto", help="vLLM's --linear-backend (marlin: weight-only FP8)")
    ap.add_argument("--logprobs", type=int, default=0, help="return the top-k log-probabilities of every token (0: none)")
    ap.add_argument("--max-num-seqs", type=int, default=48, help="48: G3's rollout server; 128: the RL-FP8-QMATH runs (WEIGHT-ONLY-FP8)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(a.model)
    ps = (dapo_prompts(tok, a.n) if a.prompts == "dapo" else math_prompts(tok, a.n) if a.prompts == "math"
          else prompts(tok, a.results, a.n))
    kw = {"quantization": "fp8"} if a.fp8 else {}
    if a.linear_backend != "auto":                       # MISMATCH-DECOMP: marlin = weight-only FP8
        kw["linear_backend"] = a.linear_backend
    if a.logprobs:                                       # MISMATCH-DECOMP (E3): the cost of returning the sampler's top-k
        kw["max_logprobs"] = max(20, a.logprobs)
    ads = a.adapters or ([a.adapter] if a.adapter else [])
    if ads:
        kw.update(enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=max(2, len(ads) + 1))
    llm = LLM(model=a.model, max_model_len=16384, max_num_seqs=a.max_num_seqs, gpu_memory_utilization=a.gpu_util, seed=0, **kw)
    reqs = [LoRARequest(f"pol{i}", i + 1, d) for i, d in enumerate(ads)]
    req = [reqs[i % len(reqs)] for i in range(len(ps))] if reqs else None
    sp = SamplingParams(temperature=1.0, max_tokens=a.max_tokens, ignore_eos=True, seed=0,
                        **({"logprobs": a.logprobs} if a.logprobs else {}))
    feed = [{"prompt_token_ids": p} for p in ps]
    llm.generate(feed[:4], SamplingParams(max_tokens=16, ignore_eos=True), lora_request=req[:4] if req else None,
                 use_tqdm=False)  # warm-up
    times, toks = [], 0
    for _ in range(a.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(feed, sp, lora_request=req, use_tqdm=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
    best = min(times)
    row = {"label": a.label, "gpu": torch.cuda.get_device_name(), "model": Path(a.model).name, "fp8": a.fp8,
           "adapter": a.adapter, "adapters": len(ads), "prompts": a.prompts, "lowbit_fused": os.environ.get("LOWBIT_FUSED", "0"),
           "lowbit_lora_fp4": os.environ.get("LOWBIT_LORA_FP4", "0"), "n": len(ps),
           "prompt_tokens_mean": sum(map(len, ps)) / len(ps), "gen_tokens": toks, "seconds": times,
           "tok_per_s": toks / best, "gpu_util": a.gpu_util, "job": os.environ.get("SLURM_JOB_ID", ""),
           "max_num_seqs": a.max_num_seqs, "linear_backend": a.linear_backend, "logprobs": a.logprobs}
    with open(a.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps({k: row[k] for k in ("label", "gpu", "gen_tokens", "seconds", "tok_per_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
