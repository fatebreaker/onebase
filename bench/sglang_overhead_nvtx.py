"""PROFILE-STEP-H100 copy (docs/REGISTRY.md): sglang_overhead.py with W1's two timed runs marked as NVTX ranges (W1_1, W1_257)
for bench/prof_steps.py; otherwise unchanged. Original docstring:
Review W4 (descriptive): a second serving system -- what SGLang's LoRA path costs over the same SGLang engine without
LoRA, with the workloads of bench/overhead.py (GPU):
  W1 batch-1 decode: 1 request, 512 prompt tokens. TPOT = (t(257 tokens) - t(1 token)) / 256, median of 3 pairs.
  W2 throughput: 256 requests, 512 in / 256 out, submitted together -> generated tokens/s.
  W3 prefill-heavy: 64 requests, 2048 in / 16 out -> (prompt + generated) tokens/s.
Random prompt tokens, ignore_eos, greedy, radix (prefix) cache off. Requests cycle through the adapters round-robin; the
n adapters are n names for one PEFT dir (symlinked copies, so SGLang treats them as distinct adapters). One engine
configuration per process.
    python bench/sglang_overhead.py --model <dir> [--quantization fp8] [--adapter <dir> --rank 64 --n-adapters 1]
        [--lora-backend csgmv|triton] --label L --out x.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


def gpu_info() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader", "-i", "0"],
                       capture_output=True, text=True).stdout.strip()
    name, _, cc = q.partition(",")
    return {"gpu": name.strip(), "cc": cc.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-adapters", type=int, default=1)
    ap.add_argument("--lora-backend", default="csgmv")
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--workloads", default="W1,W2,W3")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import sglang as sgl
    import torch
    kw = dict(model_path=a.model, disable_radix_cache=True, mem_fraction_static=a.mem_fraction, random_seed=a.seed,
              log_level="error")
    if a.quantization:
        kw["quantization"] = a.quantization
    names: list[str] = []
    if a.adapter:
        names = [f"a{i}" for i in range(a.n_adapters)]
        root = Path(os.environ.get("TMPDIR", "/tmp")) / f"sgl_adapters_{os.getpid()}"
        root.mkdir(parents=True, exist_ok=True)
        paths = {}
        for nm in names:
            d = root / nm
            if not d.exists():
                d.symlink_to(Path(a.adapter).resolve())
            paths[nm] = str(d)
        kw.update(enable_lora=True, lora_paths=paths, max_lora_rank=a.rank, lora_backend=a.lora_backend,
                  max_loras_per_batch=max(8, a.n_adapters), max_loaded_loras=max(8, a.n_adapters))
    t0 = time.time()
    llm = sgl.Engine(**kw)
    load_s = time.time() - t0
    vocab = json.loads((Path(a.model) / "config.json").read_text())["vocab_size"]
    rng = random.Random(a.seed)

    def prompts(n: int, length: int) -> list[list[int]]:
        return [[rng.randrange(1000, vocab - 1000) for _ in range(length)] for _ in range(n)]

    def run(ps: list[list[int]], max_new: int):
        lp = [names[i % len(names)] for i in range(len(ps))] if names else None
        sp = {"max_new_tokens": max_new, "temperature": 0.0, "ignore_eos": True}
        t = time.perf_counter()
        outs = llm.generate(input_ids=ps, sampling_params=sp, lora_path=lp)
        return time.perf_counter() - t, outs

    n_warm = max(8, len(names))
    run(prompts(n_warm, 64), 16)                        # loads every adapter, captures the decode graphs
    run(prompts(n_warm, 64), 16)
    wl = set(a.workloads.split(","))
    rec = {"label": a.label, "system": "sglang", "sglang": sgl.__version__, "torch": torch.__version__,
           "model": Path(a.model).name, "quantization": a.quantization or "bf16", "rank": a.rank if names else None,
           "n_adapters": len(names), "lora": bool(names), "lora_backend": a.lora_backend if names else None,
           **gpu_info(), "load_s": round(load_s, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "job": os.environ.get("SLURM_JOB_ID", "")}
    if "W1" in wl:
        p = prompts(1, 512)
        tp = []
        for _ in range(3):
            import torch.cuda.nvtx as _nv   # PROFILE-STEP-H100: the two timed runs as NVTX ranges for nsys
            _nv.range_push("W1_1"); t1, _ = run(p, 1); _nv.range_pop()
            _nv.range_push("W1_257"); t257, o = run(p, 257); _nv.range_pop()
            assert o[0]["meta_info"]["completion_tokens"] == 257, o[0]["meta_info"]
            tp.append((t257 - t1) / 256)
        rec["W1_tpot_ms"] = round(1000 * statistics.median(tp), 3)
        rec["W1_tpot_ms_all"] = [round(1000 * x, 3) for x in tp]
    if "W2" in wl:
        dt, outs = run(prompts(256, 512), 256)
        n = sum(o["meta_info"]["completion_tokens"] for o in outs)
        rec["W2_gen_tok_s"] = round(n / dt, 1)
        rec["W2_s"] = round(dt, 2)
    if "W3" in wl:
        dt, outs = run(prompts(64, 2048), 16)
        n = sum(o["meta_info"]["prompt_tokens"] + o["meta_info"]["completion_tokens"] for o in outs)
        rec["W3_tok_s"] = round(n / dt, 1)
        rec["W3_s"] = round(dt, 2)
    print(json.dumps(rec), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    llm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
