"""Serving cost of the remedies (paper Sec. 7): decode throughput and memory of one vLLM configuration (GPU).

A merged model (bf16_merge or bf16_sr_merge: same tensors, same cost), the base with the RL adapter unmerged
(the exact remedy), and real-kernel FP8 (vLLM online FP8 quantization) with and without the adapter. One process per
configuration, so engines never share a GPU:

    venv_vllm/bin/python -m mlsys.bench_serve --model <dir> [--adapter <dir>] [--quantization fp8] --out x.jsonl

Workload: --batch prompts of --in-len random tokens, --out-len generated tokens each (ignore_eos), after a warm-up;
reports generated tokens/s, plus single-sequence decode latency per token (batch 1).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--quantization", default="")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--in-len", type=int, default=1024)
    ap.add_argument("--out-len", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import subprocess
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    kw = {"enable_lora": True, "max_lora_rank": 128, "max_loras": 1} if a.adapter else {}
    if a.quantization:
        kw["quantization"] = a.quantization
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util, seed=0, **kw)
    load_s = time.time() - t0
    lora = LoRARequest("rl", 1, a.adapter) if a.adapter else None
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)
    prompts = [{"prompt_token_ids": [rng.randrange(1000, vocab - 1000) for _ in range(a.in_len)]} for _ in range(a.batch)]
    sp = SamplingParams(max_tokens=a.out_len, ignore_eos=True, temperature=0.0)
    llm.generate(prompts[:8], SamplingParams(max_tokens=32, ignore_eos=True), lora_request=lora)   # warm-up
    t0 = time.time()                                  # generate() returns when every sequence has finished
    outs = llm.generate(prompts, sp, lora_request=lora)
    dt = time.time() - t0
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    t1 = time.time()
    one = llm.generate(prompts[:1], SamplingParams(max_tokens=256, ignore_eos=True, temperature=0.0), lora_request=lora)
    lat = (time.time() - t1) / max(1, len(one[0].outputs[0].token_ids))
    rec = {"label": a.label or ("+".join(x for x in (a.quantization or "bf16", "lora" if a.adapter else "") if x)),
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
                                 capture_output=True, text=True).stdout.strip(), "batch": a.batch, "in_len": a.in_len, "out_len": a.out_len,
           "gen_tok_per_s": round(n_out / dt, 1), "batch1_ms_per_tok": round(1000 * lat, 2), "load_s": round(load_s, 1)}
    print(json.dumps(rec), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
