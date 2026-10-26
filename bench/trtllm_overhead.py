"""TRTLLM-H200 (docs/REGISTRY.md; mock reviews R1 v26 change 2, R3 v27 missing 7): what TensorRT-LLM's LoRA path costs, with the
workloads of bench/overhead.py and bench/sglang_overhead.py (GPU; run inside NVIDIA's TensorRT-LLM release container):
  W1 batch-1 decode: 1 request, 512 prompt tokens. TPOT = (t(257 tokens) - t(1 token)) / 256, median of 3 pairs.
  W2 throughput: 256 requests, 512 in / 256 out, submitted together -> generated tokens/s.
Random prompt tokens, ignore_eos, greedy, KV block reuse (prefix cache) off, CUDA graphs on for batch sizes 1-256 (padded; with
LoRA, TensorRT-LLM's CUDA-graph LoRA manager), LoRA on the main stream or overlapped with the base on a second one (--overlap).
Requests cycle through the adapters round-robin; the n adapters are n names for one PEFT dir (symlinked copies, so the engine
treats them as distinct adapters), as in sglang_overhead.py. One engine configuration per process. The PyTorch backend (the
default LLM API backend), the checkpoint's own quantization (a ModelOpt FP8 checkpoint: per-tensor weight and static activation
scales; or the official block-scaled FP8 checkpoint).
    python3 bench/trtllm_overhead.py --model <dir> [--adapter <dir> --rank 64 --n-adapters 1] --label L --out x.jsonl
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
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-adapters", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="KV-cache share of free GPU memory; default 0.85, 0.5 above 8 adapters (TensorRT-LLM's LoRA device cache, "
                         "about 21 GB, does not fit beside an 0.85 KV cache on an H200; 256 x 768 tokens need about 29 GB of KV)")
    ap.add_argument("--no-cuda-graphs", action="store_true")
    ap.add_argument("--overlap", action="store_true", help="LoraConfig.overlap_lora_and_base: LoRA on a second stream")
    ap.add_argument("--workloads", default="W1,W2")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    # Inside the Apptainer image MPI's dynamic spawn fails (MPI_ERR_SPAWN, also on a login node), so TensorRT-LLM's default TP1
    # executor (one MPI-spawned worker process) cannot start; its own single-process TP1 worker runs instead (it warns that this
    # may hurt streaming generation; these workloads do not stream). Caches go to the job's TMPDIR (Apptainer keeps $HOME).
    os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    tmp = os.environ.get("TMPDIR", "/tmp")
    for k, sub in (("XDG_CACHE_HOME", "xdg"), ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                   ("TORCH_EXTENSIONS_DIR", "torch_ext"), ("HF_HOME", "hf")):
        os.environ.setdefault(k, os.path.join(tmp, sub))
    if a.mem_fraction is None:
        a.mem_fraction = 0.5 if (a.adapter and a.n_adapters > 8) else 0.85
    import tensorrt_llm
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    kw = dict(model=a.model, kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=a.mem_fraction, enable_block_reuse=False),
              max_batch_size=256, max_seq_len=1024, max_num_tokens=8192)
    if not a.no_cuda_graphs:
        kw["cuda_graph_config"] = CudaGraphConfig(batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256], enable_padding=True)
    names: list[str] = []
    reqs: list = []
    if a.adapter:
        try:
            from tensorrt_llm.lora_helper import LoraConfig
        except ImportError:
            from tensorrt_llm.lora_manager import LoraConfig
        from tensorrt_llm.executor.request import LoRARequest
        names = [f"a{i}" for i in range(a.n_adapters)]
        root = Path(os.environ.get("TMPDIR", "/tmp")) / f"trt_adapters_{os.getpid()}"
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        for nm in names:
            d = root / nm
            if not d.exists():
                d.symlink_to(Path(a.adapter).resolve())
            paths.append(str(d))
        n_slots = max(8, a.n_adapters)
        kw["lora_config"] = LoraConfig(lora_dir=paths[:1], max_lora_rank=a.rank, max_loras=n_slots, max_cpu_loras=n_slots,
                                       overlap_lora_and_base=a.overlap)
        reqs = [LoRARequest(nm, i + 1, p) for i, (nm, p) in enumerate(zip(names, paths))]
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    vocab = json.loads((Path(a.model) / "config.json").read_text())["vocab_size"]
    rng = random.Random(a.seed)

    def prompts(n: int, length: int) -> list[list[int]]:
        return [[rng.randrange(1000, vocab - 1000) for _ in range(length)] for _ in range(n)]

    def run(ps: list[list[int]], max_new: int):
        lr = [reqs[i % len(reqs)] for i in range(len(ps))] if reqs else None
        sp = SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True)
        t = time.perf_counter()
        outs = llm.generate(ps, sampling_params=sp, lora_request=lr, use_tqdm=False)
        return time.perf_counter() - t, outs

    n_warm = max(8, len(names))
    run(prompts(n_warm, 64), 16)                        # loads every adapter, captures the decode graphs
    run(prompts(n_warm, 64), 16)
    wl = set(a.workloads.split(","))
    rec = {"label": a.label, "system": "tensorrt_llm", "tensorrt_llm": tensorrt_llm.__version__, "torch": torch.__version__,
           "backend": "pytorch", "kv_mem_fraction": a.mem_fraction, "model": Path(a.model).name, "rank": a.rank if names else None, "n_adapters": len(names),
           "lora": bool(names), "cuda_graphs": not a.no_cuda_graphs,
           "overlap": bool(names) and a.overlap, "single_process_worker": os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS") == "1", **gpu_info(), "load_s": round(load_s, 1),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": os.environ.get("SLURM_JOB_ID", "")}
    qc = Path(a.model) / "hf_quant_config.json"
    rec["quantization"] = (json.loads(qc.read_text()).get("quantization", {}).get("quant_algo") if qc.exists() else
                           (json.loads((Path(a.model) / "config.json").read_text()).get("quantization_config") or {}).get(
                               "quant_method", "bf16"))
    if "W1" in wl:
        p = prompts(1, 512)
        tp = []
        for _ in range(3):
            t1, _ = run(p, 1)
            t257, o = run(p, 257)
            assert len(o[0].outputs[0].token_ids) == 257, len(o[0].outputs[0].token_ids)
            tp.append((t257 - t1) / 256)
        rec["W1_tpot_ms"] = round(1000 * statistics.median(tp), 3)
        rec["W1_tpot_ms_all"] = [round(1000 * x, 3) for x in tp]
    if "W2" in wl:
        dt, outs = run(prompts(256, 512), 256)
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        rec["W2_gen_tok_s"] = round(n / dt, 1)
        rec["W2_s"] = round(dt, 2)
    print(json.dumps(rec), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    llm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
