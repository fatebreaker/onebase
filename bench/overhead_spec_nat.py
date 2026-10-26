"""SPEC-DECODE-NAT-H200 (docs/REGISTRY.md): SPEC-DECODE-H200 on natural prompts (mock review R1 v37, missing 3). bench/overhead_spec.py's
engine settings with WildChat's first user turns (results/traces/wildchat_sharegpt_4000.json, the first 64 of at most 512 prompt tokens,
in Qwen3's chat template with thinking off) instead of the copy prompt; 256 greedy tokens per request (ignore_eos):
  N<B> for B in --batches (default 1,4,8): 8 / B rounds of B concurrent requests (prompts 0-7), wall time per generated token per
  request (TPOT, ms), the median of three repetitions of the rounds; vLLM's draft and acceptance counters over them.
--spec-ngram 0 runs the same workloads without speculation. One adapter (slot 1) when --adapter is given.
    python bench/overhead_spec_nat.py --model <dir> --quantization fp8 [--adapter <dir> --rank 64] --spec-ngram 4 --out x.jsonl"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overhead import dispatch_summary, gpu_info, kernel_sha  # noqa: E402
from overhead_spec import counters  # noqa: E402

LB = Path(__file__).resolve().parents[1]
TRACE = LB / "results/traces/wildchat_sharegpt_4000.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--spec-ngram", type=int, default=4)
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    kw: dict = {"enable_prefix_caching": False, "disable_log_stats": False}
    if a.adapter:
        kw.update(enable_lora=True, max_lora_rank=a.rank, max_loras=1, max_cpu_loras=1)
    if a.quantization:
        kw["quantization"] = a.quantization
    if a.spec_ngram:
        kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": a.spec_ngram, "prompt_lookup_max": 4,
                                    "prompt_lookup_min": 2}
    fused = os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False")
    if fused and not os.environ.get("LOWBIT_REPORT"):
        os.environ["LOWBIT_REPORT"] = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"lowbit_layers_{os.getpid()}.txt")
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.85, seed=0, **kw)
    load_s = time.time() - t0
    lora = LoRARequest("a0", 1, a.adapter) if a.adapter else None
    tok = llm.get_tokenizer()
    prompts = []
    for c in json.loads(TRACE.read_text()):
        text = tok.apply_chat_template([{"role": "user", "content": c["conversations"][0]["value"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        if len(tok(text, add_special_tokens=False)["input_ids"]) <= 512:
            prompts.append(text)
        if len(prompts) >= 64:
            break
    sp = SamplingParams(max_tokens=256, ignore_eos=True, temperature=0.0)

    def rounds(b: int) -> float:
        t = time.perf_counter()
        for i in range(0, 8, b):
            outs = llm.generate(prompts[i:i + b], sp, lora_request=lora, use_tqdm=False)
            assert all(len(o.outputs[0].token_ids) == 256 for o in outs)
        return (time.perf_counter() - t) / (256 * (8 // b))      # per generated token per request

    llm.generate(prompts[8:16], sp, lora_request=lora, use_tqdm=False)   # warm-up on other prompts
    llm.generate(prompts[16:17], sp, lora_request=lora, use_tqdm=False)
    rec = {"label": a.label, "model": Path(a.model).name, "quantization": a.quantization or "bf16", "rank": a.rank,
           "lora": bool(a.adapter), "spec_ngram": a.spec_ngram, "prompts": "wildchat first turns (chat template, thinking off)",
           "lowbit_fused": fused, "kernel_sha": kernel_sha(), "tuned_configs": os.environ.get("VLLM_TUNED_CONFIG_FOLDER", ""),
           **gpu_info(), "vllm": vllm.__version__, "torch": torch.__version__, "load_s": round(load_s, 1),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": os.environ.get("SLURM_JOB_ID", "")}
    for b in (int(x) for x in a.batches.split(",")):
        c0 = counters(llm)
        tp = [rounds(b) for _ in range(3)]
        time.sleep(1.0)
        c1 = counters(llm)
        rec[f"N{b}_tpot_ms"] = round(1000 * statistics.median(tp), 3)
        rec[f"N{b}_tpot_ms_all"] = [round(1000 * x, 3) for x in tp]
        rec[f"N{b}_spec"] = {k: (c1[k] - c0.get(k, 0)) if isinstance(c1.get(k), (int, float)) else c1.get(k) for k in c1}
    rec["dispatch_hybrid"] = dispatch_summary(os.environ.get("LOWBIT_REPORT", "")) if fused else None
    print(json.dumps(rec), flush=True)
    with Path(a.out).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
