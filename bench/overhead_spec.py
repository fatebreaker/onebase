"""SPEC-DECODE-H200 (docs/REGISTRY.md): the LoRA path under vLLM's n-gram speculative decoding, whose verify steps carry 1 + k tokens
per sequence (GPU). bench/overhead.py's engine settings and helpers; one engine configuration per process:
    python bench/overhead_spec.py --model <dir> --quantization fp8 [--adapter <dir> --rank 64] --spec-ngram 4 --out x.jsonl
Workloads (a copy prompt, so that n-gram drafts are accepted; greedy, ignore_eos, prefix caching off):
  S<B> for B in --batches (default 1,4): B requests, each the same 384-token document and "Copy it" instruction with a distinct
  leading line, 256 generated tokens each, submitted together; median of 3 runs of wall time per generated token per request
  (TPOT, ms) and, from vLLM's counters over the three runs, drafts (verify steps per request), draft tokens and accepted tokens.
--spec-ngram 0 runs the same workloads without speculation. All requests use the one adapter (slot 1) when --adapter is given.
"""
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

DOC = ("The river town kept its records in a stone house by the bridge. Every spring the clerk opened the shutters, aired the "
       "ledgers, and copied the winter's entries into the great book: births and deaths, the price of grain at each market day, "
       "the height of the water at the bridge pier, and the names of the boats that passed downstream with timber. ")


def counters(llm) -> dict:
    out = {}
    try:
        for m in llm.get_metrics():
            name = m.name.removesuffix("_total")
            if name in ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens"):
                out[name.rsplit("num_", 1)[1]] = getattr(m, "value", None)
    except Exception as e:   # noqa: BLE001 (counters are descriptive)
        out["error"] = repr(e)[:120]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--spec-ngram", type=int, default=4, help="draft tokens per step (0: no speculation)")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    kw: dict = {"enable_prefix_caching": False, "disable_log_stats": False}
    if a.adapter:
        assert a.rank > 0
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
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util, seed=0, **kw)
    load_s = time.time() - t0
    lora = LoRARequest("a0", 1, a.adapter) if a.adapter else None
    tok = llm.get_tokenizer()
    body = DOC * 6
    ids = tok(body, add_special_tokens=False)["input_ids"][:384]
    body = tok.decode(ids)

    def prompts(n: int) -> list[str]:
        return [f"Request {i}.\nBelow is a document. Copy it exactly, word for word.\n\nDocument:\n{body}\n\nCopy:\n" for i in range(n)]

    sp = SamplingParams(max_tokens=256, ignore_eos=True, temperature=0.0)

    def run(ps: list[str]) -> tuple[float, list]:
        t = time.perf_counter()
        outs = llm.generate(ps, sp, lora_request=lora, use_tqdm=False)
        return time.perf_counter() - t, outs

    run(prompts(4))   # warm-up: compiles and captures the verify steps
    run(prompts(1))
    rec = {"label": a.label, "model": Path(a.model).name, "quantization": a.quantization or "bf16", "rank": a.rank,
           "lora": bool(a.adapter), "spec_ngram": a.spec_ngram,
           "lowbit_fused": fused, "kernel_sha": kernel_sha(),
           "dispatch_k": int(os.environ.get("LOWBIT_DISPATCH_K", "1")),
           "tuned_configs": os.environ.get("VLLM_TUNED_CONFIG_FOLDER", ""),
           **gpu_info(), "vllm": vllm.__version__, "torch": torch.__version__, "load_s": round(load_s, 1),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": os.environ.get("SLURM_JOB_ID", "")}
    for b in (int(x) for x in a.batches.split(",")):
        c0 = counters(llm)
        tp = []
        for _ in range(3):
            dt, outs = run(prompts(b))
            assert all(len(o.outputs[0].token_ids) == 256 for o in outs)
            tp.append(dt / 256)
        c1 = counters(llm)
        rec[f"S{b}_tpot_ms"] = round(1000 * statistics.median(tp), 3)
        rec[f"S{b}_tpot_ms_all"] = [round(1000 * x, 3) for x in tp]
        rec[f"S{b}_spec"] = {k: (c1[k] - c0.get(k, 0)) if isinstance(c1.get(k), (int, float)) else c1.get(k) for k in c1}
        rec[f"S{b}_first_tokens"] = list(outs[0].outputs[0].token_ids[:24])
    rec["dispatch_hybrid"] = dispatch_summary(os.environ.get("LOWBIT_REPORT", "")) if fused else None
    print(json.dumps(rec), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
