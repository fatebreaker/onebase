"""KV-gap probe (diagnostic, not a registered test; mock review R1 v22, minor point and question 5): where the plugin's ~65k
missing KV-cache tokens with 32 rank-64 bf16 adapters go (Table S-absolute: 357k against vLLM's 422k on the L40S). One engine
start per mode, built as bench/overhead.py builds it (Qwen3.5-9B online FP8, rank 64, 32 slots, max length 4,096, utilization
0.85), in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the probe can wrap the plugin's per-layer build: the CUDA memory
still allocated after each layer's set_mapping (vLLM's own, plus the plugin's op construction, dispatch timing and warm-up),
the shared scratch's sizes, and vLLM's KV-cache size. vLLM's own profiling lines (model loading, peak activation, CUDA graphs)
are in the log.
    MODE=stock|plugin|plugin-nowarm python bench/kv_gap_probe.py --model <dir> --out results/kv_gap_probe.jsonl"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-adapters", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    mode = os.environ.get("MODE", "stock")
    rec: dict = {"mode": mode, "layers": 0, "set_mapping_bytes": 0, "op_init_bytes": 0, "warmup_bytes": 0, "warmup_peak_bytes": 0}

    def alloc() -> int:
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()

    from lowbit_lora import plugin
    plugin.install()                                   # a no-op unless LOWBIT_FUSED=1 (idempotent; vLLM calls it again)
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    sm = BaseLinearLayerWithLoRA.set_mapping

    def set_mapping(self, pw):
        m0 = alloc()
        sm(self, pw)
        rec["set_mapping_bytes"] += alloc() - m0
        rec["layers"] += 1
    BaseLinearLayerWithLoRA.set_mapping = set_mapping
    if mode != "stock":
        from lowbit_lora.kernels import fp8_lora_vllm as F
        init, warm = F.VllmFusedFP8LoRA.__init__, F.VllmFusedFP8LoRA.warmup

        def init_w(self, *x, **kw):
            m0 = alloc()
            init(self, *x, **kw)
            rec["op_init_bytes"] += alloc() - m0

        def warm_w(self):
            m0 = alloc()
            torch.cuda.reset_peak_memory_stats()
            warm(self)
            rec["warmup_bytes"] += alloc() - m0
            rec["warmup_peak_bytes"] = max(rec["warmup_peak_bytes"], torch.cuda.max_memory_allocated() - m0)
        F.VllmFusedFP8LoRA.__init__, F.VllmFusedFP8LoRA.warmup = init_w, warm_w
    if os.environ.get("MEMHIST", "0") == "1":     # allocation stack traces of what stays allocated after the engine start
        torch.cuda.memory._record_memory_history(max_entries=500000)
    from vllm import LLM
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util, seed=0, enable_prefix_caching=False,
              quantization="fp8", enable_lora=True, max_lora_rank=a.rank, max_loras=a.n_adapters, max_cpu_loras=a.n_adapters)
    rec["load_s"] = time.time() - t0
    cc = llm.llm_engine.vllm_config.cache_config
    rec["kv_tokens"] = int(cc.num_gpu_blocks) * int(cc.block_size)
    if mode != "stock":
        from lowbit_lora.kernels import fp8_lora_vllm as F
        rec["scratch_bytes"] = sum(t.numel() * t.element_size() for s in F._SCRATCH.values() for t in (s.hp, s.pmax, s.xq, s.sx))
        rec["fused_layers"] = len(plugin._REGISTRY)
    rec["allocated_after_start"] = alloc()
    rec["gpu"] = torch.cuda.get_device_name()
    if os.environ.get("MEMHIST", "0") == "1":
        snap = torch.cuda.memory._snapshot()
        agg: dict = {}
        for seg in snap["segments"]:
            for b in seg["blocks"]:
                if b["state"] != "active_allocated":
                    continue
                fr = [f for f in b.get("frames", []) if f.get("filename", "").endswith(".py")]
                key = next((f"{f['filename'].split('site-packages/')[-1].split('code/')[-1]}:{f['line']}:{f['name']}" for f in fr
                            if "lowbit_lora" in f["filename"] or "/lora/" in f["filename"]), None) or \
                      (f"{fr[0]['filename'].split('site-packages/')[-1]}:{fr[0]['line']}:{fr[0]['name']}" if fr else "?")
                agg[key] = agg.get(key, 0) + b["size"]
        rec["live_by_site_mib"] = {k: round(v / 2**20, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:25]}
        lb: dict = {}                                  # vLLM's LoRA buffers: (site, MiB) -> [count, one call stack]
        for seg in snap["segments"]:
            for b in seg["blocks"]:
                fr = [f for f in b.get("frames", []) if f.get("filename", "").endswith(".py")]
                if b["state"] != "active_allocated" or not any("/lora/layers/" in f["filename"] for f in fr):
                    continue
                site = next(f"{f['filename'].split('site-packages/')[-1]}:{f['line']}" for f in fr if "/lora/layers/" in f["filename"])
                k = f"{site} {b['size'] / 2**20:.2f}"
                if k not in lb:
                    lb[k] = [0, [f"{f['filename'].split('site-packages/')[-1].split('code/')[-1]}:{f['line']}:{f['name']}" for f in fr[:18]]]
                lb[k][0] += 1
        rec["lora_blocks"] = lb
    print("KVPROBE", json.dumps(rec), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
