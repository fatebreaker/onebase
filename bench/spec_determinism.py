"""Diagnostic (not a registered test; SPEC-DECODE-H200 follow-up): do repeated greedy runs of one prompt give the same tokens?
SPEC-DECODE-H200's plugin rows had runs with far fewer accepted n-gram drafts than the others (3 of 12 runs, 3.97-5.01 against
1.14 ms per token), vLLM's none. Either the outputs differ between runs (run-to-run nondeterminism in the engine's arithmetic),
or they agree and only the drafts differ (the proposer's timing). One engine per process, bench/overhead_spec.py's settings and
copy prompt: --runs repetitions of one request (256 greedy tokens), each run's full token ids, wall time and vLLM's draft /
acceptance counters; then --runs repetitions of four concurrent requests (tokens of every request).
    python bench/spec_determinism.py --model <dir> --quantization fp8 [--adapter <dir> --rank 64] --spec-ngram 4 --out x.jsonl"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overhead import gpu_info, kernel_sha  # noqa: E402
from overhead_spec import DOC, counters  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--spec-ngram", type=int, default=4)
    ap.add_argument("--runs", type=int, default=8)
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
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.85, seed=0, **kw)
    lora = LoRARequest("a0", 1, a.adapter) if a.adapter else None
    tok = llm.get_tokenizer()
    ids = tok(DOC * 6, add_special_tokens=False)["input_ids"][:384]
    body = tok.decode(ids)

    def prompts(n: int) -> list[str]:
        return [f"Request {i}.\nBelow is a document. Copy it exactly, word for word.\n\nDocument:\n{body}\n\nCopy:\n" for i in range(n)]

    sp = SamplingParams(max_tokens=256, ignore_eos=True, temperature=0.0)
    llm.generate(prompts(4), sp, lora_request=lora, use_tqdm=False)   # warm-up, as overhead_spec.py
    llm.generate(prompts(1), sp, lora_request=lora, use_tqdm=False)
    runs = []
    for b in (1, 4):
        for i in range(a.runs):
            c0 = counters(llm)
            t = time.perf_counter()
            outs = llm.generate(prompts(b), sp, lora_request=lora, use_tqdm=False)
            dt = time.perf_counter() - t
            time.sleep(1.0)                                            # let the engine's counters catch up
            c1 = counters(llm)
            runs.append({"batch": b, "run": i, "ms_per_token": round(1000 * dt / 256, 3),
                         "spec": {k: (c1[k] - c0.get(k, 0)) if isinstance(c1.get(k), (int, float)) else c1.get(k) for k in c1},
                         "tokens": [list(o.outputs[0].token_ids) for o in outs]})
    ref = {b: next(r["tokens"] for r in runs if r["batch"] == b) for b in (1, 4)}
    for r in runs:   # first position where a request's tokens differ from run 0's (None: identical)
        r["first_diff"] = [next((j for j, (u, v) in enumerate(zip(t, t0)) if u != v), None) for t, t0 in zip(r["tokens"], ref[r["batch"]])]
    rec = {"label": a.label, "model": Path(a.model).name, "spec_ngram": a.spec_ngram, "lora": bool(a.adapter),
           "lowbit_fused": os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False"),
           "lowbit_pdl": os.environ.get("LOWBIT_PDL", "0"), "kernel_sha": kernel_sha(), **gpu_info(), "vllm": vllm.__version__,
           "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": os.environ.get("SLURM_JOB_ID", ""),
           "runs": runs}
    print(json.dumps({"label": a.label, "summary": [(r["batch"], r["run"], r["ms_per_token"], r["first_diff"],
                                                     r["spec"].get("accepted_tokens")) for r in runs]}), flush=True)
    with Path(a.out).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
