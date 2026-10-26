"""ICML E1 (docs/PLAN_ICML_V2.md; registered before its data): is a low-precision sampler's mismatch bias or noise?
Engines (vLLM, one RL adapter, in-process, prefix caching off): bf16; fq, the matched W8A16 GEMM fed per-token
FP8-rounded activations (round to nearest: lowbit_lora/w16_engine_fq.py, MISMATCH-DECOMP-2's control); fqsr, the same with
stochastic rounding (lowbit_lora/w16_engine_fqsr.py: fresh noise at every call, eager).
  --phase gen       sample --group completions for each of --prompts DAPO-Math prompts (temperature 1, non-thinking) and keep
                    the engine's log-probability of every sampled token (realization 0 of the noise)
  --phase resample  K more passes over the same sequences (vLLM's prompt log-probabilities; fresh noise per pass for fqsr)
  --phase score     the twin W8A16 trainer's (w16fused) log-probabilities of the sampled tokens (as bench/mismatch_bench2.py)
  --phase report    per token: the per-draw KL estimate log q0(y) - log p(y) (the realized sampler; MISMATCH-DECOMP-2's
                    statistic) and the bias KL estimate log qbar(y) - log p(y), qbar the mean over the K resample passes'
                    probabilities (an unbiased estimate of the noise-averaged sampler at y; y ~ q0 is a draw from qbar), each with
                    a 95% bootstrap interval over prompts -> results/mismatch_sr.jsonl
For a deterministic engine (bf16, fq) the K passes repeat q0, so the two statistics coincide (a check).
    VLLM_ENABLE_V1_MULTIPROCESSING=0 LOWBIT_FUSED=0 python bench/mismatch_sr.py --phase gen --engine fqsr --model M --adapter A --data D"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "train"))
sys.path.insert(0, str(LB / "bench"))
DL = Path("$SCRATCH/data/std_bench_dl")


def engine(a):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if a.engine == "fq":
        from lowbit_lora import w16_engine_fq as E
        E.install()
    elif a.engine == "fqsr":
        from lowbit_lora import w16_engine_fqsr as E
        E.install()
    from vllm import LLM
    return LLM(model=a.model, enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=2, max_model_len=4096,
               gpu_memory_utilization=a.gpu_util, enforce_eager=(a.engine == "fqsr"), enable_prefix_caching=False, max_num_seqs=64,
               seed=0, quantization=None if a.engine == "bf16" else "fp8")


def gen(a) -> None:
    import pandas as pd
    import std_bench
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    df = pd.read_parquet(DL.parent / "dapo_dl/en/train.parquet")
    data = [r.prompt + std_bench.SUFFIX for r in df.itertuples()]
    random.Random(1234).shuffle(data)
    llm = engine(a)
    sp = SamplingParams(n=a.group, temperature=1.0, top_p=1.0, max_tokens=a.max_new, seed=7, logprobs=0)
    msgs = [[{"role": "user", "content": q}] for q in data[:a.prompts]]
    outs = llm.chat(msgs, sp, lora_request=LoRARequest("policy", 1, a.adapter), use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})
    rows = [{"prompt": list(o.prompt_token_ids), "completion": list(c.token_ids),
             "elp": [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]} for o in outs for c in o.outputs]
    Path(a.data).write_text(json.dumps({"engine": a.engine, "rows": rows}))
    print(f"wrote {len(rows)} sequences, {sum(len(r['completion']) for r in rows)} completion tokens to {a.data}", flush=True)


def resample(a) -> None:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    d = json.loads(Path(a.data).read_text())
    assert d["engine"] == a.engine, (d["engine"], a.engine)
    rows = [r for r in d["rows"] if r["completion"]]
    llm = engine(a)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    lr = LoRARequest("policy", 1, a.adapter)
    for r in rows:
        r["qk"] = []
    for k in range(a.passes):
        outs = llm.generate([{"prompt_token_ids": r["prompt"] + r["completion"]} for r in rows], sp, lora_request=lr, use_tqdm=False)
        for r, o in zip(rows, outs):
            n0, ids = len(r["prompt"]), r["prompt"] + r["completion"]
            r["qk"].append([o.prompt_logprobs[i][ids[i]].logprob for i in range(n0, len(ids))])
        print(f"pass {k + 1}/{a.passes}", flush=True)
    d["rows"] = rows
    d["passes"] = a.passes
    Path(a.data).write_text(json.dumps(d))


def score(a) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from token_lp import token_logprobs
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    if a.trainer == "w16fused":                                          # the twin W8A16 trainer
        from mlsys.matched_fp8 import patch_model
        patch_model(model, parts="w", use_scaled_mm=True)
        from lowbit_lora import fp8_train
        fp8_train.fp8_backward(model, kind="w8a16")
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    if a.trainer == "w16fused":
        from lowbit_lora import train_lora
        train_lora.install(model, kind="w16")
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.eval()
    d = json.loads(Path(a.data).read_text())
    with torch.no_grad():
        for r in d["rows"]:
            p, c = r["prompt"], r["completion"]
            ids = torch.tensor([p + c], device="cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            r["tlp"] = token_logprobs(logits, ids)[0, len(p) - 1:len(p) + len(c) - 1].float().cpu().tolist()
            del logits
    d["trainer"] = a.trainer + (" (fla)" if importlib.util.find_spec("fla") else "")
    Path(a.data).write_text(json.dumps(d))
    print(f"scored {len(d['rows'])} sequences", flush=True)


def report(a) -> None:
    from mismatch_bench2 import boot
    d = json.loads(Path(a.data).read_text())
    groups: dict = {}
    for r in d["rows"]:
        if not r["completion"] or "tlp" not in r or "qk" not in r:
            continue
        g = groups.setdefault(tuple(r["prompt"]), {"draw": 0.0, "bias": 0.0, "fresh": 0.0, "n": 0})
        K = len(r["qk"])
        for t, (e, p) in enumerate(zip(r["elp"], r["tlp"])):
            qs = [r["qk"][k][t] for k in range(K)]
            m = max(qs)
            qbar = m + math.log(sum(math.exp(q - m) for q in qs) / K)
            g["draw"] += e - p
            g["bias"] += qbar - p
            g["fresh"] += qs[0] - p
            g["n"] += 1
    G = list(groups.values())
    tot = lambda k: sum(g[k] for g in G) / max(sum(g["n"] for g in G), 1)  # noqa: E731
    row = {"label": a.label, "engine": d["engine"], "trainer": d.get("trainer"), "passes": d.get("passes"), "prompts": len(G),
           "tokens": sum(g["n"] for g in G), "kl_draw": tot("draw"), "kl_draw_ci95": boot([(g["draw"], g["n"]) for g in G]),
           "kl_bias": tot("bias"), "kl_bias_ci95": boot([(g["bias"], g["n"]) for g in G]),
           "kl_fresh": tot("fresh"), "kl_fresh_ci95": boot([(g["fresh"], g["n"]) for g in G])}
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("gen", "resample", "score", "report"))
    ap.add_argument("--engine", choices=("bf16", "fq", "fqsr"))
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--trainer", default="w16fused", choices=("w16fused", "bf16"))
    ap.add_argument("--gpu-util", type=float, default=0.45)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(LB / "results/mismatch_sr.jsonl"))
    a = ap.parse_args()
    {"gen": gen, "resample": resample, "score": score, "report": report}[a.phase](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
