"""FIRSTWAVE-GRID (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data): bench/firstwave_grid.py's rows
(results/firstwave_grid_<gpu>.jsonl, the newest job), the configurations that ran and passed their check (relative error <= 2e-2).
Discriminating cases: |pred_firstwave - pred_wholeloop| >= 1.0 us.
  (i)  over the discriminating cases, the first-wave prediction's median absolute error below the whole-loop prediction's
  (ii) over every case, the Spearman rank correlation of the first-wave prediction with the measured exposure at least 0.5
Void if fewer than 8 discriminating cases. Descriptive: both models' errors over all cases, the exposure by grid / n_sm.
    python3 bench/score_firstwave_grid.py [--gpu h200] [--json results/firstwave_grid_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

LB = Path(__file__).resolve().parents[1]


def rank(xs: list) -> list:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2
        i = j + 1
    return rk


def pearson(a: list, b: list) -> float | None:
    ma, mb = st.fmean(a), st.fmean(b)
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((y - mb) ** 2 for y in b) ** 0.5
    return None if sa == 0 or sb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="h200")
    ap.add_argument("--file", default="", help="-SWEPT: results/firstwave_grid_swept_<gpu>.jsonl")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    f = Path(a.file) if a.file else LB / f"results/firstwave_grid_{a.gpu}.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    job = max((r.get("job", "") for r in rows), default="")
    cs = [r for r in rows if r.get("job", "") == job and r.get("ok")]
    out: dict = {"job": job, "n": len(cs), "n_failed": sum(1 for r in rows if r.get("job", "") == job and not r.get("ok"))}
    if cs:
        disc = [r for r in cs if abs(r["pred_firstwave"] - r["pred_wholeloop"]) >= 1.0]
        ew = lambda rs: st.median(abs(r["exposed"] - r["pred_firstwave"]) for r in rs)   # noqa: E731
        e0 = lambda rs: st.median(abs(r["exposed"] - r["pred_wholeloop"]) for r in rs)   # noqa: E731
        sp = pearson(rank([r["pred_firstwave"] for r in cs]), rank([r["exposed"] for r in cs]))
        out.update(n_discriminating=len(disc), all_err_firstwave=ew(cs), all_err_wholeloop=e0(cs), spearman_firstwave=sp,
                   spearman_wholeloop=pearson(rank([r["pred_wholeloop"] for r in cs]), rank([r["exposed"] for r in cs])))
        if len(disc) >= 8:
            out["disc_err_firstwave"], out["disc_err_wholeloop"] = ew(disc), e0(disc)
            out["verdict"] = {"i": {"met": ew(disc) < e0(disc)}, "ii": {"met": sp is not None and sp >= 0.5, "spearman": sp}}
        else:
            out["verdict"] = {"status": "void", "why": f"{len(disc)} discriminating cases (< 8)"}
        out["by_wave"] = {}
        for lo, hi, lab in ((0, 1, "<= 1 wave"), (1, 2, "1-2 waves"), (2, 4, "2-4 waves"), (4, 1e9, "> 4 waves")):
            v = [r for r in cs if lo < r["grid"] / r["n_sm"] <= hi or (lo == 0 and r["grid"] / r["n_sm"] <= hi)]
            if v:
                out["by_wave"][lab] = {"n": len(v), "exposed_median": st.median(r["exposed"] for r in v),
                                       "pred_firstwave_median": st.median(r["pred_firstwave"] for r in v)}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
