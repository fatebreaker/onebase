"""Where the plugin's memory goes at engine start (diagnostic; the E2E grids show the plugin with less KV cache than vLLM's
LoRA path): builds the plugin's op (fp8_lora_vllm.VllmFusedFP8LoRA, bf16 adapters in vLLM-style stacked buffers) for
each projection of one decoder layer, runs its warmup (the timed dispatch and every kernel variant), and reports the
device memory outside PyTorch's allocator before and after (CUDA modules of the compiled kernels, their local-memory
reservation), PyTorch's own peak, and the compiled Triton kernels with their registers and spills. One process per
configuration, since loaded modules are not unloaded:
    python bench/plugin_mem.py --model qwen3_8b --rank 64 --slots 32 --max-m 8192 [--paths fused|all] [--out f.jsonl]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_bench import SHAPES  # noqa: E402


def used() -> tuple[int, int]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return total - free, torch.cuda.memory_reserved()


def triton_kernels() -> list[dict]:
    """Every compiled kernel of the lowbit kernels' Triton functions (Triton 3.x JITFunction.device_caches)."""
    import triton
    from lowbit_lora.kernels import fp8_lora_vllm, fp8_lora_multi, common
    out = []
    for mod in (fp8_lora_vllm, fp8_lora_multi, common):
        for name, fn in vars(mod).items():
            jit = getattr(fn, "fn", fn)          # autotuners/heuristics wrap the JITFunction
            caches = getattr(jit, "device_caches", None)
            if not isinstance(jit, triton.runtime.jit.JITFunction) or not caches:
                continue
            for dev, entry in caches.items():
                cache = entry[0] if isinstance(entry, tuple) else entry
                for k in cache.values():
                    out.append({"fn": f"{mod.__name__.split('.')[-1]}.{name}", "n_regs": getattr(k, "n_regs", None),
                                "n_spills": getattr(k, "n_spills", None),
                                "shared": getattr(getattr(k, "metadata", None), "shared", None),
                                "bytes": len(getattr(k, "kernel", b"") or b"")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--max-m", type=int, default=8192)
    ap.add_argument("--paths", default="all", choices=("fused", "all"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from lowbit_lora.kernels.fp8_lora_vllm import VllmFusedFP8LoRA
    torch.manual_seed(0)
    torch.empty(1, device="cuda")                       # the context
    u0, r0 = used()
    L, r, ops_ = a.slots + 1, a.rank, []
    for proj, (K, slices) in SHAPES[a.model].items():
        N, S = sum(slices), len(slices)
        w_q = (torch.randn(N, K, device="cuda") * 0.02 / 0.01).to(torch.float8_e4m3fn)
        s_w = torch.full((1,), 0.01, device="cuda")
        a_st = tuple((torch.randn(L, 1, r, K, device="cuda") / K ** 0.5).to(torch.bfloat16) for _ in range(S))
        b_st = tuple((torch.randn(L, 1, n, r, device="cuda") * 0.02).to(torch.bfloat16) for n in slices)
        op = VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, a.max_m)
        if a.paths == "all":                            # vLLM's quantize + CUTLASS: the hybrid paths are timed too

            def vendor(x, _w=w_q.t(), _s=s_w):
                xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(xq, _w, sx, _s, torch.bfloat16)
            op.base_fn = vendor
        ops_.append(op)
    u1, r1 = used()
    torch.cuda.reset_peak_memory_stats()
    for op in ops_:
        op.warmup()
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    u2, r2 = used()
    ks = triton_kernels()
    rec = {"model": a.model, "rank": r, "slots": a.slots, "max_m": a.max_m, "paths": a.paths,
           "gpu": torch.cuda.get_device_name(), "nontorch_before_MiB": (u1 - r1) / 2**20,
           "nontorch_after_MiB": (u2 - r2) / 2**20, "nontorch_delta_MiB": ((u2 - r2) - (u1 - r1)) / 2**20,
           "context_MiB": (u0 - r0) / 2**20, "torch_peak_warmup_MiB": peak / 2**20,
           "torch_reserved_after_MiB": r2 / 2**20, "triton_kernels": len(ks),
           "max_spills": max((k["n_spills"] or 0) for k in ks) if ks else None,
           "kernels_with_spills": sum(1 for k in ks if (k["n_spills"] or 0) > 0),
           "cubin_MiB": sum(k["bytes"] for k in ks) / 2**20,
           "dispatch": {str(b): v for b, v in sorted(ops_[0].dispatch.items())}}
    print(json.dumps(rec))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
