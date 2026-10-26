"""End-to-end serving overhead tables (CPU, seconds): for each (GPU, model, base), the change of W1 (batch-1 TPOT),
W2 (throughput) and W3 (prefill-heavy throughput) against the same engine without LoRA, for the stock vLLM LoRA path
and each lowbit plugin mode (bf16 / FP8-stored / NVFP4-stored adapters), from results/*.jsonl.

    python bench/score_e2e.py [--sha <kernel_sha prefix>] [--md]
Rows of each plugin mode come from its newest job (one grid job is one consistent run), restricted to rows whose
kernel_sha starts with --sha if given.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
FILES = ("p0_overhead.jsonl", "fused_overhead.jsonl", "blackwell_overhead.jsonl", "blackwell_fused.jsonl")
WL = (("W1_tpot_ms", "W1", -1), ("W2_gen_tok_s", "W2", 1), ("W3_tok_s", "W3", 1))


def mode(r: dict) -> str:
    if not r.get("lowbit_fused"):
        return "stock+dual" if r.get("dual_stream") else "stock"
    return "fused-fp4ad" if r.get("lora_fp4") else "fused-fp8ad" if r.get("lora_fp8") else "fused"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default="")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for f in FILES if (R / f).exists() for l in (R / f).read_text().splitlines() if l.strip()]
    newest = {}
    for r in rows:   # newest job per (gpu, model, quantization, mode), by time
        if r.get("lowbit_fused"):
            k = (r["gpu"], r["model"], r["quantization"], mode(r))
            if r.get("kernel_sha", "").startswith(a.sha) and r["time"] >= newest.get(k, ("", ""))[0]:
                newest[k] = (r["time"], r.get("job"))
    table = defaultdict(dict)   # (gpu, model, quant) -> {(setting, mode): row}
    refs = defaultdict(dict)    # (gpu, model, quant) -> {mode: no-LoRA row}
    for r in rows:
        m = mode(r)
        key = (r["gpu"], r["model"].replace("-NVFP4", ""), r["quantization"])
        if r.get("lowbit_fused") and r.get("job") != newest.get((r["gpu"], r["model"], r["quantization"], m), ("", None))[1]:
            continue
        if not r["lora"]:
            refs[key][m] = r
            continue
        setting = f"r{r['rank']}:n{r['n_adapters']}"
        table[key][(setting, m)] = r
    for key in sorted(table):
        gpu, model, quant = key
        modes = sorted({m for _, m in table[key]}, key=["stock", "stock+dual", "fused", "fused-fp8ad", "fused-fp4ad"].index)
        settings = sorted({s for s, _ in table[key]}, key=lambda s: (int(s[1:].split(":")[0]), int(s.split(":n")[1])))
        print(f"\n### {gpu}, {model}, {quant} base (change vs no-LoRA; W1 latency up is worse, W2/W3 down is worse)")
        hdr = "| setting | " + " | ".join(f"{w} {m}" for _, w, _ in WL for m in modes) + " |"
        print(hdr)
        print("|" + "---|" * (1 + len(WL) * len(modes)))
        for s in settings:
            cells = []
            for k, _, _ in WL:
                for m in modes:
                    r = table[key].get((s, m))
                    ref = refs[key].get(m) or refs[key].get("stock") or refs[key].get("fused")
                    cells.append("–" if r is None or ref is None or k not in r else f"{100 * (r[k] / ref[k] - 1):+.1f}%")
            print(f"| {s} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
