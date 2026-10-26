"""FIRSTWAVE-72B (docs/REGISTRY.md, registered 2026-10-25 before the data of its primary cases; mock review R3 v38, change 2: the
first-wave condition was fitted post hoc on 131 cases, so test it on shapes it has not seen): the decode-path benchmark's rows for
Qwen2.5-72B's projections on one H200 (E2E-72B-H200 -TILES step 2, bench/pdl_k2_bench.py --stages qwen72b -> results/
pdl_k2_72b_h200.jsonl; primary) and its TP2 shards on one H100 (E2E-TP2-TUNED step 1 -> results/pdl_k2_72b_tp2_h100.jsonl; secondary:
collected at 19:16, before this registration, and not analysed for exposure). Cases, model and filter exactly as
paper_mlsys/figs/make_fig_exposure.py (v38): one adapter, bf16, M <= 16, the file's newest job, rows with t_pdl_noshrink - t_pdl > 5 us
dropped; measured exposure t_pdl - t_pdl_noshrink; first-wave prediction max(0, t_shrink_only - t_k2_nolora * min(1, 132 / grid)), grid
= cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N) * SPLIT_K of the swept best tiles; whole-loop prediction max(0, t_shrink_only - t_k2_nolora).
  (i)  primary cases: the first-wave prediction's median absolute error at most 1.0 us
  (ii) primary cases: its median absolute error at most the whole-loop prediction's
Descriptive: per-case rows, the count of cases whose grid exceeds 132 (where the two predictions can differ), correlations, the
secondary cases' same readouts.
    python3 bench/score_firstwave_72b.py [--json results/firstwave_72b_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
NSM = 132
NEED = ("t_k2_nolora", "t_shrink_only", "t_pdl", "t_pdl_noshrink")
SETS = (("primary", "pdl_k2_72b_h200.jsonl"), ("secondary", "pdl_k2_72b_tp2_h100.jsonl"))


def cases(fn: str) -> list[dict]:
    f = LB / "results" / fn
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    rows = [r for r in rows if all(isinstance(r.get(k), (int, float)) for k in NEED) and r.get("n") == 1
            and r.get("fmt", "bf16") == "bf16" and r.get("M", 99) <= 16]
    if not rows:
        return []
    job = max(str(r.get("job", "")) for r in rows)
    out = []
    for r in rows:
        if str(r.get("job", "")) != job or r["t_pdl_noshrink"] - r["t_pdl"] > 5.0:
            continue
        cfg = r.get("pdl_best") or {}
        grid = (-(-r["M"] // cfg["BLOCK_M"]) * -(-r["N"] // cfg["BLOCK_N"]) * cfg["SPLIT_K"]
                if all(k in cfg for k in ("BLOCK_M", "BLOCK_N", "SPLIT_K")) and "N" in r else None)
        win = r["t_k2_nolora"] * min(1.0, NSM / grid) if grid else r["t_k2_nolora"]
        out.append({"proj": r["proj"], "M": r["M"], "N": r.get("N"), "K": r.get("K"), "grid": grid, "t_main": r["t_k2_nolora"],
                    "t_shrink": r["t_shrink_only"], "win": win, "meas": r["t_pdl"] - r["t_pdl_noshrink"],
                    "pred_w": max(0.0, r["t_shrink_only"] - win), "pred": max(0.0, r["t_shrink_only"] - r["t_k2_nolora"])})
    return out


def corr(a: list, b: list) -> float | None:
    if len(a) < 3:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((y - mb) ** 2 for y in b) ** 0.5
    return None if sa == 0 or sb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def med(xs: list) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out: dict = {}
    for name, fn in SETS:
        cs = cases(fn)
        if not cs:
            out[name] = {"n": 0}
            continue
        ew = med([abs(c["meas"] - c["pred_w"]) for c in cs])
        e0 = med([abs(c["meas"] - c["pred"]) for c in cs])
        out[name] = {"n": len(cs), "beyond_one_wave": sum(1 for c in cs if c["grid"] and c["grid"] > NSM),
                     "median_err_firstwave_us": ew, "median_err_wholeloop_us": e0,
                     "corr_firstwave": corr([c["pred_w"] for c in cs], [c["meas"] for c in cs]),
                     "corr_wholeloop": corr([c["pred"] for c in cs], [c["meas"] for c in cs]), "cases": cs}
        if name == "primary":
            out["verdict"] = {"i": {"met": ew <= 1.0, "median_err_us": ew}, "ii": {"met": ew <= e0, "firstwave_us": ew, "wholeloop_us": e0}}
    out["complete"] = out.get("primary", {}).get("n", 0) >= 12
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
