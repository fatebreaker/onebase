"""PROFILE-STEP-H100 copy (docs/REGISTRY.md): overhead.py with W1's two timed runs marked as NVTX ranges (W1_1, W1_257)
for bench/prof_steps.py; otherwise unchanged. Original docstring:
P0-OVERHEAD (docs/REGISTRY.md): what vLLM's stock LoRA path costs over the same engine without LoRA (GPU).

One engine configuration per process, so engines never share a GPU:
    python bench/overhead.py --model <dir> [--quantization fp8] [--adapter <dir> --rank 64 --n-adapters 1] --out x.jsonl
--adapter is one PEFT dir; the n adapters are n LoRA slots loaded from it under distinct ids. A LoRA path's cost does
not depend on the values, only on the rank and on how many distinct adapters a batch touches.

Registered workloads (random prompt tokens, ignore_eos, greedy, prefix caching off):
  W1 batch-1 decode: 1 request, 512 prompt tokens. TPOT = (t(257 tokens) - t(1 token)) / 256, median of 3 pairs.
  W2 throughput: 256 requests, 512 in / 256 out, submitted together -> generated tokens/s.
  W3 prefill-heavy: 64 requests, 2048 in / 16 out -> (prompt + generated) tokens/s.
Requests cycle through the adapters round-robin. VLLM_LORA_ENABLE_DUAL_STREAM=1 in the environment selects vLLM's
aux-stream LoRA path (recorded as dual_stream).
"""
from __future__ import annotations

import os as _os_nvtx   # PROFILE-STEP-H100 amendment (2026-10-25 10:2x): nsys does not trace vLLM's engine core, which its client forks
_os_nvtx.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # without exec; run the engine in this process so its CUDA work is traced

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


def dispatch_summary(report: str) -> dict | None:
    """From the plugin's layer report: per M bucket, the share of fused layers whose timed dispatch chose a hybrid
    path (vendor GEMM + our LoRA kernels: "h", or "q" with our activation codes) over the fused one, with the share
    of "q" in brackets when nonzero; {"layers": n} counts the layers."""
    try:
        lines = [ln.split() for ln in open(report) if " dispatch=" in ln]
    except OSError:
        return None
    hyb: dict[str, list[str]] = {}
    for ln in lines:
        for item in ln[-1].removeprefix("dispatch=").split(","):
            b, _, v = item.partition(":")
            if b.isdigit():
                hyb.setdefault(b, []).append(v)
    out = {}
    for b, v in sorted(hyb.items(), key=lambda kv: int(kv[0])):
        h = sum(x in ("h", "q", "s") for x in v) / len(v)
        sub = {c: sum(x == c for x in v) / len(v) for c in ("q", "s")}
        extra = ", ".join(f"{sub[c]:.2f} {c}" for c in ("q", "s") if sub[c])
        out[b] = round(h, 2) if not extra else f"{h:.2f} ({extra})"
    return {**out, "layers": len(lines)}


def kernel_sha() -> str:
    """Version of the fused path (kernels + plugin), so rows from different kernel revisions never mix."""
    import hashlib
    import lowbit_lora   # the imported copy (jobs run a snapshot taken at job start, jobs/lib.sh)
    root = Path(lowbit_lora.__file__).resolve().parent
    h = hashlib.sha256()
    for f in ("kernels/fp8_lora_vllm.py", "kernels/nvfp4_lora_vllm.py", "kernels/adapter_store.py", "kernels/common.py",
              "plugin.py"):
        h.update((root / f).read_bytes())
    return h.hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--n-adapters", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="cap on concurrent sequences (0: vLLM's default); a large hybrid model needs it below its "
                         "linear-attention state blocks (Qwen3.6-27B: E2E-27B amendment)")
    ap.add_argument("--lora-target-modules", nargs="*", default=None,
                    help="vLLM's deployment-time LoRA target modules (adapter-visible names, e.g. q_proj ... down_proj): "
                         "only those layers get LoRA wrappers; default all supported modules")
    ap.add_argument("--workloads", default="W1,W2,W3")
    ap.add_argument("--label", default="")
    ap.add_argument("--no-adapter-requests", action="store_true",
                    help="SERVE-DECOMP: load the adapters (LoRA enabled, slots filled) but send every workload request without "
                         "one, so the LoRA layers run with no adapter: the engine's base path inside LoRA layers")
    ap.add_argument("--profile-dir", default="", help="also profile a decode window (torch profiler traces)")
    ap.add_argument("--profile-batch", type=int, default=8)
    ap.add_argument("--linear-backend", default="auto",
                    help="vLLM's --linear-backend (torch: FP8 GEMMs through torch._scaled_mm, i.e. cuBLASLt; STOCK-CUBLAS-L40S)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    kw: dict = {"enable_prefix_caching": False}
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    if a.adapter:
        assert a.rank > 0 and a.n_adapters > 0
        kw.update(enable_lora=True, max_lora_rank=a.rank, max_loras=a.n_adapters, max_cpu_loras=a.n_adapters)
        if a.lora_target_modules:
            kw["lora_target_modules"] = list(a.lora_target_modules)
    if a.quantization:
        kw["quantization"] = a.quantization
    if a.linear_backend != "auto":
        kw["linear_backend"] = a.linear_backend
    if a.profile_dir:
        kw["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": a.profile_dir,
                                 "torch_profiler_with_stack": False, "torch_profiler_use_gzip": False}
    fused = os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False")
    if fused and not os.environ.get("LOWBIT_REPORT"):   # the plugin's per-layer decisions (written by the workers)
        os.environ["LOWBIT_REPORT"] = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"lowbit_layers_{os.getpid()}.txt")
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util, seed=0, **kw)
    load_s = time.time() - t0
    loras = [LoRARequest(f"a{i}", i + 1, a.adapter) for i in range(a.n_adapters)] if a.adapter else []

    def lr(n: int):
        return [loras[i % len(loras)] for i in range(n)] if loras else None

    lr_wl = (lambda n: None) if a.no_adapter_requests else lr   # the workloads' requests (the warm-up still loads the slots)
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)

    def prompts(n: int, length: int) -> list[dict]:
        return [{"prompt_token_ids": [rng.randrange(1000, vocab - 1000) for _ in range(length)]} for _ in range(n)]

    def sp(n_out: int) -> SamplingParams:
        return SamplingParams(max_tokens=n_out, ignore_eos=True, temperature=0.0)

    def run(ps: list[dict], n_out: int, warm: bool = False) -> tuple[float, list]:
        t = time.perf_counter()
        outs = llm.generate(ps, sp(n_out), lora_request=(lr if warm else lr_wl)(len(ps)), use_tqdm=False)
        return time.perf_counter() - t, outs

    n_warm = max(8, len(loras))
    run(prompts(n_warm, 64), 16, warm=True)            # loads every adapter slot, compiles/captures the decode path
    run(prompts(n_warm, 64), 16, warm=True)
    wl = set(a.workloads.split(","))
    try:   # what vLLM actually runs (a ModelOpt NVFP4 checkpoint is detected from its config, with no flag)
        quant = a.quantization or llm.llm_engine.model_config.quantization or "bf16"
    except AttributeError:
        quant = a.quantization or "bf16"
    try:   # the KV cache vLLM sized after loading the model, LoRA slots and the plugin's buffers
        cc = llm.llm_engine.vllm_config.cache_config
        kv_tokens = int(cc.num_gpu_blocks) * int(cc.block_size)
    except (AttributeError, TypeError):
        kv_tokens = None
    rec = {"label": a.label, "model": Path(a.model).name, "quantization": quant, "max_num_seqs": a.max_num_seqs or None,
           "rank": a.rank, "n_adapters": a.n_adapters, "lora": bool(a.adapter), "no_adapter_requests": a.no_adapter_requests,
           "lora_target_modules": (list(a.lora_target_modules) if (a.adapter and a.lora_target_modules) else None),
           "dual_stream": os.environ.get("VLLM_LORA_ENABLE_DUAL_STREAM", "0") not in ("0", "", "false", "False"),
           "lowbit_fused": os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False"),
           "kernel_sha": kernel_sha(),
           "lora_fp8": os.environ.get("LOWBIT_LORA_FP8", "0") not in ("0", "", "false", "False"),
           "lora_fp4": os.environ.get("LOWBIT_LORA_FP4", "0") not in ("0", "", "false", "False"),
           "dispatch": os.environ.get("LOWBIT_DISPATCH", "auto"),
           "dispatch_k": int(os.environ.get("LOWBIT_DISPATCH_K", "1")),
           "release_slots": os.environ.get("LOWBIT_RELEASE_SLOTS", "0") not in ("0", "", "false", "False"),
           "kv_tokens": kv_tokens,
           "dispatch_hybrid": dispatch_summary(os.environ.get("LOWBIT_REPORT", "")) if fused else None,
           **gpu_info(), "vllm": vllm.__version__, "torch": torch.__version__, "load_s": round(load_s, 1),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": os.environ.get("SLURM_JOB_ID", "")}
    if "W1" in wl:
        p = prompts(1, 512)
        tp = []
        for _ in range(3):
            import torch.cuda.nvtx as _nv   # PROFILE-STEP-H100: the two timed runs as NVTX ranges for nsys
            _nv.range_push("W1_1"); t1, _ = run(p, 1); _nv.range_pop()
            _nv.range_push("W1_257"); t257, o = run(p, 257); _nv.range_pop()
            assert len(o[0].outputs[0].token_ids) == 257
            tp.append((t257 - t1) / 256)
        rec["W1_tpot_ms"] = round(1000 * statistics.median(tp), 3)
        rec["W1_tpot_ms_all"] = [round(1000 * x, 3) for x in tp]
    if "W2" in wl:
        dt, outs = run(prompts(256, 512), 256)
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        rec["W2_gen_tok_s"] = round(n / dt, 1)
        rec["W2_s"] = round(dt, 2)
    if "W3" in wl:
        dt, outs = run(prompts(64, 2048), 16)
        n = sum(len(o.prompt_token_ids) + len(o.outputs[0].token_ids) for o in outs)
        rec["W3_tok_s"] = round(n / dt, 1)
        rec["W3_s"] = round(dt, 2)
    if a.profile_dir:   # a short decode window: B requests x 64 generated tokens after a 512-token prompt
        llm.start_profile()
        run(prompts(a.profile_batch, 512), 64)
        llm.stop_profile()
        rec["profile_dir"] = a.profile_dir
    print(json.dumps(rec), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
