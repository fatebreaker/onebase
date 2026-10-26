"""Sum GPU kernel time by category in a vLLM torch-profiler trace (bench/overhead.py --profile-dir).

    python3 bench/parse_trace.py <trace dir or .json[.gz]>
Categories: base GEMM (cutlass / our _gemm), LoRA (Punica shrink/expand, our shrink), FP8 quantization, attention,
norm/activation/other. Busy = sum of kernel durations; span = first kernel start to last kernel end; idle = 1 - busy/span.
"""
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CATS = [("lora", r"lora_shrink|lora_expand|_shrink_vllm|_shrink_nvfp4|sgmv|bgmv|punica"),
        ("base_gemm", r"cutlass|_gemm_vllm|_gemm_nvfp4|_gemm_lora|gemm|Gemm|sm90_xmma|sm89_xmma|nvjet|fp4_mm|mm_fp4"),
        ("act_quant", r"scaled_fp8_quant|dynamic_per_token|fp8_quant|fp4_quant|cvt_fp16_to_fp4|quant"),
        ("attention", r"flash|attn|attention|paged"),
        ("norm_act", r"rms|norm|silu|act_and_mul|rotary")]


def load(path: Path) -> list:
    if path.is_dir():
        files = sorted(path.rglob("*.json")) + sorted(path.rglob("*.json.gz"))
        files = [f for f in files if "capture" not in str(f)]
        if not files:
            raise SystemExit(f"no trace in {path}")
        path = max(files, key=lambda f: f.stat().st_size)          # the worker trace (CPU + GPU) is the largest
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh).get("traceEvents", []), path


def main() -> int:
    ev, path = load(Path(sys.argv[1]))
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    if not k:
        print("no GPU kernel events in", path)
        return 1
    by = defaultdict(float)
    for e in k:
        name = e.get("name", "")
        cat = next((c for c, pat in CATS if re.search(pat, name)), "other")
        by[cat] += e["dur"]
    busy = sum(by.values())
    span = max(e["ts"] + e["dur"] for e in k) - min(e["ts"] for e in k)
    print(f"{path}: {len(k)} GPU events; span {span / 1e3:.1f} ms, busy {busy / 1e3:.1f} ms, idle {100 * (1 - busy / span):.1f}%")
    for c, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {c:10s} {v / 1e3:8.2f} ms  {100 * v / busy:5.1f}% of busy")
    per = defaultdict(lambda: [0, 0.0])
    for e in k:
        key = (re.sub(r"[(<].*", "", e.get("name", ""))[:48], tuple(e.get("args", {}).get("grid", [])))
        per[key][0] += 1
        per[key][1] += e["dur"]
    print("  top kernels (name, grid): count, total ms, mean us")
    for (n, g), (c, t) in sorted(per.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"    {n:48s} {str(g):18s} {c:6d} {t / 1e3:8.1f} {t / c:8.1f}")
    # the largest GPU-idle gaps, and the longest CPU-side events inside each (JIT compiles, syncs, Python)
    ks = sorted(k, key=lambda e: e["ts"])
    gaps = sorted(((ks[i + 1]["ts"] - (ks[i]["ts"] + ks[i]["dur"]), ks[i]["ts"] + ks[i]["dur"]) for i in range(len(ks) - 1)),
                  reverse=True)[:5]
    cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    print("  largest GPU-idle gaps (ms) and the longest CPU events inside them:")
    for g, t0 in gaps:
        inside = sorted((e for e in cpu if e["ts"] >= t0 and e["ts"] < t0 + g), key=lambda e: -e["dur"])[:3]
        desc = "; ".join(f"{e.get('cat', '')}:{e.get('name', '')[:50]} {e['dur'] / 1e3:.1f}ms" for e in inside)
        print(f"    {g / 1e3:8.1f}  {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
