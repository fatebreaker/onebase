"""Diagnostic (not registered data): where the plugin's prefill time goes on one GPU. Offline vLLM engine (as bench/overhead.py
builds it: online FP8, CUDA graphs, LoRA with N resident adapters of rank R), then:
  ttft      median over --reps of one request's prefill (a --prompt-len-token prompt, max_tokens=1) with adapter i (cycling)
  mixed     median over --reps of a step mix: --mixed requests with distinct adapters, each a --prompt-len prompt, max_tokens=1
  profile   one torch.profiler trace of a single prefill (CPU and CUDA time per op, top --top of each, default 25), as JSON lines
The serving mode comes from the environment, as for bench/overhead.py (LOWBIT_FUSED, VLLM_TUNED_CONFIG_FOLDER, LOWBIT_PDL, ...).
    python bench/prefill_profile.py --model <dir> --adapter <dir> --rank 64 --n-adapters 32 --label <mode> --out <jsonl>"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-adapters", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--mixed", type=int, default=4)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--profile-mixed", action="store_true", help="profile the mixed batch instead of one prefill")
    ap.add_argument("--top", type=int, default=25, help="ops listed per profile (KERNEL-DIFF-H100: 400, every op)")
    a = ap.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    llm = LLM(model=a.model, quantization="fp8", enable_lora=True, max_lora_rank=a.rank, max_loras=a.n_adapters,
              max_cpu_loras=a.n_adapters, max_model_len=4096, gpu_memory_utilization=a.gpu_util, seed=0,
              enable_prefix_caching=False)
    reqs = [LoRARequest(f"a{i}", i + 1, a.adapter) for i in range(a.n_adapters)]
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    import random
    rng = random.Random(0)
    vocab = 30000

    def prompt():
        return TokensPrompt(prompt_token_ids=[rng.randrange(100, vocab) for _ in range(a.prompt_len)])

    # warm-up: every adapter loaded into its slot, every path compiled
    for i in range(a.n_adapters):
        llm.generate([prompt()], sp, lora_request=reqs[i], use_tqdm=False)
    llm.generate([prompt() for _ in range(a.mixed)], sp, lora_request=reqs[:a.mixed], use_tqdm=False)
    rows = []

    def timed(fn) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0)

    ttft = [timed(lambda i=i: llm.generate([prompt()], sp, lora_request=reqs[i % a.n_adapters], use_tqdm=False))
            for i in range(a.reps)]
    mixed = [timed(lambda i=i: llm.generate([prompt() for _ in range(a.mixed)], sp,
                                            lora_request=[reqs[(i * a.mixed + k) % a.n_adapters] for k in range(a.mixed)],
                                            use_tqdm=False)) for i in range(a.reps)]
    rows.append({"label": a.label, "what": "ttft", "median_ms": st.median(ttft), "min_ms": min(ttft), "n": len(ttft)})
    rows.append({"label": a.label, "what": f"mixed{a.mixed}", "median_ms": st.median(mixed), "min_ms": min(mixed), "n": len(mixed)})
    print(json.dumps(rows[-2]), json.dumps(rows[-1]), flush=True)
    # one profiled prefill, in this process: with VLLM_ENABLE_V1_MULTIPROCESSING=0 the model runs here
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "0":
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            if a.profile_mixed:
                llm.generate([prompt() for _ in range(a.mixed)], sp, lora_request=reqs[:a.mixed], use_tqdm=False)
            else:
                llm.generate([prompt()], sp, lora_request=reqs[0], use_tqdm=False)
            torch.cuda.synchronize()
        ev = prof.key_averages()
        for key, n in (("self_cpu_time_total", "cpu"), ("self_device_time_total", "cuda")):
            top = sorted(ev, key=lambda e: getattr(e, key, 0), reverse=True)[:a.top]
            rows.append({"label": a.label, "what": f"profile_{n}" + ("_mixed" if a.profile_mixed else ""), "top": [
                {"op": e.key[:120], "calls": e.count, "self_cpu_ms": e.self_cpu_time_total / 1000,
                 "self_cuda_ms": getattr(e, "self_device_time_total", 0) / 1000} for e in top]})
        rows.append({"label": a.label, "what": "profile_totals" + ("_mixed" if a.profile_mixed else ""),
                     "cpu_ms": sum(e.self_cpu_time_total for e in ev) / 1000,
                     "cuda_ms": sum(getattr(e, "self_device_time_total", 0) for e in ev) / 1000})
    with open(a.out, "a") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "gpu": torch.cuda.get_device_name(0), "rank": a.rank, "n_adapters": a.n_adapters,
                                 "prompt_len": a.prompt_len, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
