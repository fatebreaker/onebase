"""bench/serve_bench.py (kept unchanged for the streams that run it) plus --server-args: extra arguments for `vllm serve`
(SERVE-GRAPHS: CUDA-graph capture sizes). Otherwise serve_bench.py's doc:

Online serving benchmark (GPU): vLLM's OpenAI server with n LoRA adapters, driven by vLLM's own `vllm bench serve`
(random prompts, 512 tokens in and 256 out with ignore_eos; Poisson arrivals at each request rate; each request on a
uniformly random adapter), for one serving mode. One server per call; one JSON line per request rate.

    python bench/serve_bench.py --model <dir> [--quantization fp8] --mode stock|fused|fused-fp8ad|fused-fp4ad|nolora \
        [--adapter <dir> --rank 16 --n-adapters 32] --rates 2,6,10,inf --num-prompts 300 --out results/serve_bench.jsonl
Modes: nolora = the same server without LoRA (the merged-model cost); stock = vLLM's LoRA path; fused* = the lowbit
plugin with bf16 / FP8-stored / NVFP4-stored adapters.
Adapter pools (S-LoRA style): --max-loras S < --n-adapters N keeps all N in the host cache (--max-cpu-loras N) and S
in GPU slots, which vLLM swaps; --zipf A gives request i the adapter of a fixed seeded draw from Zipf(A) over the N
(sent in that order with round-robin assignment, so every mode sees the same sequence); the server's KV-cache size is
recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

VLLM = "$AGENT_HARNESS/venv_vllm/bin/vllm"
ROOT = Path(__file__).resolve().parents[1]
PKG = Path(importlib.util.find_spec("lowbit_lora").origin).parent   # the copy `vllm serve` imports


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1500.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:   # noqa: BLE001 -- not up yet
            pass
        time.sleep(5)
    return False


def kernel_sha() -> str:
    h = hashlib.sha256()   # of the snapshot `vllm serve` imports (first lowbit_lora on PYTHONPATH, jobs/lib.sh)
    for f in ("kernels/fp8_lora_vllm.py", "kernels/nvfp4_lora_vllm.py", "kernels/adapter_store.py", "kernels/common.py",
              "plugin.py"):
        h.update((PKG / f).read_bytes())
    return h.hexdigest()[:12]


def gpu_name() -> str:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"], capture_output=True,
                       text=True).stdout.strip()
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--mode", required=True, choices=("nolora", "stock", "fused", "fused-fp8ad", "fused-fp4ad"))
    ap.add_argument("--adapter", default="")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--n-adapters", type=int, default=1)
    ap.add_argument("--max-loras", type=int, default=0, help="GPU adapter slots (default: --n-adapters, all resident)")
    ap.add_argument("--zipf", type=float, default=0.0, help="adapter popularity exponent (0: uniformly random)")
    ap.add_argument("--rates", default="2,6,10,inf")
    ap.add_argument("--seed", type=int, default=0, help="SERVE-POOL-LONG: the request sequence's seed (prompts, arrivals, the Zipf draw); 0 = as before")
    ap.add_argument("--num-prompts", type=int, default=300)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--workdir", required=True, help="scratch dir for logs and raw result files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-args", default="", help="extra `vllm serve` arguments, split on spaces")
    a = ap.parse_args()
    port = a.port or (20000 + int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 20000)
    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    slots = a.max_loras or a.n_adapters
    tag = f"{a.mode}_r{a.rank}_n{a.n_adapters}" if a.mode != "nolora" else "nolora"
    if a.mode != "nolora" and (slots != a.n_adapters or a.zipf):
        tag += f"_s{slots}_z{a.zipf:g}"
    if a.seed:
        tag += f"_seed{a.seed}"
    env = dict(os.environ)
    fused = a.mode.startswith("fused")
    env["LOWBIT_FUSED"] = "1" if fused else "0"
    env["LOWBIT_LORA_FP8"] = "1" if a.mode == "fused-fp8ad" else "0"
    env["LOWBIT_LORA_FP4"] = "1" if a.mode == "fused-fp4ad" else "0"
    if fused:   # never reuse a compiled graph across modes (the compile cache does not key on the plugin)
        env["VLLM_CACHE_ROOT"] = str(wd / f"vllm_cache_{tag}")
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        env["LOWBIT_REPORT"] = str(wd / f"layers_{tag}.txt")
    cmd = [VLLM, "serve", a.model, "--served-model-name", "base", "--host", "127.0.0.1", "--port", str(port),
           "--max-model-len", "4096", "--gpu-memory-utilization", "0.85", "--seed", "0", "--no-enable-prefix-caching"]
    if a.quantization:
        cmd += ["--quantization", a.quantization]
    if a.server_args:
        cmd += a.server_args.split()
    names = [f"a{i}" for i in range(a.n_adapters)]
    if a.mode != "nolora":
        cmd += ["--enable-lora", "--max-loras", str(slots), "--max-cpu-loras", str(a.n_adapters),
                "--max-lora-rank", str(a.rank), "--lora-modules", *[f"{n}={a.adapter}" for n in names]]
    log = open(wd / f"server_{tag}.log", "w")
    server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rc = 0
    try:
        if not wait_ready(port, server):
            print(f"server for {tag} did not come up; see {wd / f'server_{tag}.log'}", flush=True)
            return 1
        kv = [ln.strip() for ln in (wd / f"server_{tag}.log").read_text(errors="replace").splitlines()
              if "KV cache size" in ln][-1:]
        seq = names
        if a.zipf and a.mode != "nolora":     # one seeded draw per request, the same for every mode and rate
            import random
            rng = random.Random(1234 + a.seed)
            w = [1.0 / (i + 1) ** a.zipf for i in range(a.n_adapters)]
            seq = rng.choices(names, weights=w, k=a.num_prompts)
        for rate in a.rates.split(","):
            fn = f"{tag}_rate{rate}.json"
            bench = [VLLM, "bench", "serve", "--backend", "vllm", "--base-url", f"http://127.0.0.1:{port}",
                     "--model", "base", "--tokenizer", a.model, "--dataset-name", "random",
                     "--random-input-len", str(a.input_len), "--random-output-len", str(a.output_len), "--ignore-eos",
                     "--num-prompts", str(a.num_prompts), "--request-rate", rate, "--seed", str(a.seed), "--save-result",
                     "--result-dir", str(wd), "--result-filename", fn, "--percentile-metrics", "ttft,tpot,itl,e2el",
                     "--metric-percentiles", "50,90,99", "--disable-tqdm"]
            if a.mode != "nolora":
                bench += ["--lora-modules", *seq, "--lora-assignment", "round-robin" if a.zipf else "random"]
            r = subprocess.run(bench, env={k: v for k, v in env.items() if not k.startswith("LOWBIT_")},
                               capture_output=True, text=True, timeout=3600)
            if r.returncode != 0 or not (wd / fn).exists():
                print(f"bench {tag} rate {rate} failed rc={r.returncode}: {r.stderr[-800:]}", flush=True)
                rc = 1
                continue
            res = json.loads((wd / fn).read_text())
            keep = {k: v for k, v in res.items() if isinstance(v, (int, float)) and
                    any(s in k for s in ("throughput", "ttft", "tpot", "itl", "e2el", "completed", "duration"))}
            rec = {"mode": a.mode, "server_args": a.server_args, "model": Path(a.model).name, "quantization": a.quantization or "auto",
                   "rank": a.rank if a.mode != "nolora" else 0, "n_adapters": a.n_adapters if a.mode != "nolora" else 0,
                   "max_loras": slots if a.mode != "nolora" else 0, "zipf": a.zipf, "seed": a.seed, "kv_cache": kv[0] if kv else "",
                   "rate": rate, "num_prompts": a.num_prompts, "input_len": a.input_len, "output_len": a.output_len,
                   "gpu": gpu_name(), "job": os.environ.get("SLURM_JOB_ID", ""), "kernel_sha": kernel_sha(),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **keep}
            print(json.dumps({k: rec[k] for k in ("mode", "rank", "n_adapters", "rate")} |
                             {k: round(rec[k], 2) for k in ("request_throughput", "output_throughput", "median_ttft_ms",
                                                            "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms") if k in rec}),
                  flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    finally:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(5)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
