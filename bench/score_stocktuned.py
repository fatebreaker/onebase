"""STOCK-TUNED (docs/REGISTRY.md): the plugin against vLLM's LoRA path with its kernels tuned for the GPU
(jobs/e2e_stocktuned.sbatch -> results/e2e_stocktuned.jsonl, labels <mode>+r<r>:n<n>#<start> and nolora#<start>; the last
three starts of each configuration):
  (i)   the plugin's (auto) mean W1 below stock-tuned's at (64, 1), (16, 32) and (64, 32)
  (ii)  the plugin's mean W2 above stock-tuned's at the same settings
Descriptive: stock-tuned against stock per setting; changes against the no-LoRA engine; results/vllm_lora_tuned's
tuning.jsonl summed best against default kernel times.
    python3 bench/score_stocktuned.py [--json results/stocktuned_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
SETTINGS = ((64, 1), (16, 32), (64, 32))


def load(path: Path, need=None) -> dict:
    """{(mode, (r, n)) or ("nolora", None): {"W1": [...], "W2": [...]}} over each configuration's last three measured starts
    within one run (job): the newest run holding three starts of every configuration in need (a run cut short by preemption
    is discarded), else the newest run."""
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    by_job: dict = {}
    for r in sorted(rows, key=lambda r: r.get("time", "")):
        lab = r.get("label", "").rsplit("#", 1)[0]
        if lab == "nolora":
            key = ("nolora", None)
        elif "+r" in lab:
            m, s = lab.split("+", 1)
            rk, nk = s.split(":")
            key = (m, (int(rk[1:]), int(nk[1:])))
        else:
            continue
        if "W1_tpot_ms" in r and "W2_gen_tok_s" in r:
            by_job.setdefault(r.get("job", ""), {}).setdefault(key, []).append(r)
    if not by_job:
        return {}
    newest = lambda jobs: max(jobs, key=lambda j: max(r["time"] for v in by_job[j].values() for r in v))  # noqa: E731
    complete = [j for j in by_job if need and all(len(by_job[j].get(k, [])) >= 3 for k in need)]
    job = newest(complete) if complete else newest(list(by_job))
    return {k: {"W1": [r["W1_tpot_ms"] for r in v[-3:]], "W2": [r["W2_gen_tok_s"] for r in v[-3:]], "job": job}
            for k, v in by_job[job].items()}


def mean(xs):
    return sum(xs) / len(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--file", default="results/e2e_stocktuned.jsonl", help="STOCK-TUNED-8B: results/e2e_stocktuned_qwen3.jsonl")
    ap.add_argument("--tuned", default="Qwen3.5-9B-text", help="results/vllm_lora_tuned/<model>")
    a = ap.parse_args()
    need = [(m, s) for m in ("stock", "stock-tuned", "auto") for s in SETTINGS] + [("nolora", None)]
    d = load(LB / a.file, need)
    miss = [k for k in need if len(d.get(k, {}).get("W1", [])) < 3]
    if miss:
        print(f"incomplete: {len(miss)} configurations missing, e.g. {miss[0]}")
        return 1
    ref = {w: mean(d[("nolora", None)][w]) for w in ("W1", "W2")}
    per = {}
    for s in SETTINGS:
        e = {m: {w: mean(d[(m, s)][w]) for w in ("W1", "W2")} for m in ("stock", "stock-tuned", "auto")}
        per[f"{s[0]}:{s[1]}"] = {
            **{f"{m}_{w}": e[m][w] for m in e for w in ("W1", "W2")},
            **{f"{m}_{w}_change": 100 * (e[m][w] / ref[w] - 1) for m in e for w in ("W1", "W2")},
            "tuned_vs_stock_W1": e["stock-tuned"]["W1"] / e["stock"]["W1"], "tuned_vs_stock_W2": e["stock-tuned"]["W2"] / e["stock"]["W2"],
            "auto_vs_tuned_W1": e["auto"]["W1"] / e["stock-tuned"]["W1"], "auto_vs_tuned_W2": e["auto"]["W2"] / e["stock-tuned"]["W2"]}
    v = {"i": {"met": all(p["auto_W1"] < p["stock-tuned_W1"] for p in per.values())},
         "ii": {"met": all(p["auto_W2"] > p["stock-tuned_W2"] for p in per.values())}}
    tj = LB / "results/vllm_lora_tuned" / a.tuned / "tuning.jsonl"
    if tj.exists():
        rows = [json.loads(x) for x in tj.read_text().splitlines()]
        v["kernels"] = {op: {"keys": sum(r["op"] == op for r in rows),
                             "sum_default_us": sum(r["t_default_us"] or 0 for r in rows if r["op"] == op),
                             "sum_best_us": sum(r["t_best_us"] or 0 for r in rows if r["op"] == op)} for op in ("shrink", "expand")}
    for s, p in per.items():
        print(s, json.dumps({k: round(x, 3) for k, x in p.items()}))
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"per_setting": per, "nolora": ref, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
