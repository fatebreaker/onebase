"""Greedy sanity check of a quantized checkpoint against its bf16 source (GPU): the same 4 chat prompts, 48 tokens.
    python bench/sanity_generate.py <bf16 dir> <quantized dir>
"""
import gc
import json
import sys

import torch
from vllm import LLM, SamplingParams

PROMPTS = ["What is the capital of France? Answer in one word.", "Compute 17 * 23. Show the result only.",
           "Name three primary colors.", "Write the first line of a short poem about the sea."]


def main() -> int:
    outs = {}
    for tag, path in (("bf16", sys.argv[1]), ("quant", sys.argv[2])):
        llm = LLM(model=path, max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
        print(tag, "quantization:", llm.llm_engine.model_config.quantization, flush=True)
        res = llm.chat([[{"role": "user", "content": p}] for p in PROMPTS],
                       SamplingParams(max_tokens=48, temperature=0.0), use_tqdm=False)
        outs[tag] = [r.outputs[0].text for r in res]
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    for p, a, b in zip(PROMPTS, outs["bf16"], outs["quant"]):
        print(json.dumps({"prompt": p, "bf16": a, "quant": b}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
