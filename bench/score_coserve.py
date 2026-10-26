"""CO-SERVE (docs/REGISTRY.md, registered before any data): ONE FP8 base on ONE GPU serving tenants, an RL rollout engine and
that engine's trainer at once (mlsys jobs/pod_runs/coserve_pod.sh). From the run's output directory:
  (i)   both trainer runs (X: RL alone; Y: RL with tenants) complete their steps with no OOM-skipped step, and every tenant
        request of A and of Y's scored chunks completes
  (ii)  RL with tenants: mean step time over steps 3..STEPS at most 1.3x RL alone's
  (iii) tenants with RL: median time per output token at most 50 ms and p99 time to first token at most 5 s, pooled over the
        Y chunks that start and end inside the trainer's run
Descriptive: A (tenants alone) against Y (median and p99 TTFT and TPOT, tenant output tokens/s), rollout and trainer time per
step, peak GPU memory per phase, and GPU-hours for the same work: co-located (one GPU for Y's duration) against separate
deployments (one GPU for RL alone, X's duration, and one serving the tenants meanwhile).
    python3 bench/score_coserve.py --dir <OUTD> [--json results/coserve_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def pct(xs: list, q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def phases(d: Path) -> dict:
    out: dict = {}
    for line in (d / "phases.jsonl").read_text().splitlines():
        r = json.loads(line)
        out.setdefault(r["phase"], {})[r["event"]] = r["t"]
    return out


def requests(d: Path, names: list) -> dict:
    """Per-request TTFT and TPOT (ms) pooled over the named chunks, with the chunks' completion counts."""
    ttft, tpot, done, sent, toks, dur = [], [], 0, 0, 0, 0.0
    for n in names:
        f = d / f"{n}.json"
        if not f.exists():
            sent += 1
            continue
        r = json.loads(f.read_text())
        errs = r.get("errors", [])
        sent += r.get("num_prompts", len(errs))
        done += r.get("completed", 0)
        toks += sum(r.get("output_lens", []))
        dur += r.get("duration", 0.0)
        for t, itl, e in zip(r.get("ttfts", []), r.get("itls", []), errs or [""] * len(r.get("ttfts", []))):
            if e:
                continue
            ttft.append(1e3 * t)
            if itl:
                tpot.append(1e3 * sum(itl) / len(itl))
    return {"chunks": len(names), "sent": sent, "completed": done, "tok_s": toks / dur if dur else float("nan"),
            "ttft_p50": pct(ttft, 0.5), "ttft_p99": pct(ttft, 0.99), "tpot_p50": pct(tpot, 0.5), "tpot_p99": pct(tpot, 0.99)}


def rl(d: Path, steps: int) -> dict:
    h = [json.loads(x) for x in (d / "history.jsonl").read_text().splitlines() if x.strip()] if (d / "history.jsonl").exists() else []
    s = [r for r in h if 3 <= r["step"] <= steps]
    return {"steps": max((r["step"] for r in h), default=0), "oom_skipped": sum(r.get("oom_skipped", 0) for r in h),
            "t_step": st.fmean(r["t_step"] for r in s) if s else float("nan"),
            "t_rollout": st.fmean(r["t_rollout"] for r in s) if s else float("nan"),
            "t_train": st.fmean(r["t_step"] - r["t_rollout"] for r in s) if s else float("nan"),
            "reward": st.fmean(r["mean_reward"] for r in s) if s else float("nan"),
            "errored": sum(r.get("errored", 0) for r in s)}


def peak_mem(d: Path, ph: dict) -> dict:
    """Peak memory.used (MiB) per phase window from smi.csv (timestamps 'YYYY/MM/DD HH:MM:SS.mmm', local time of the node)."""
    import datetime as dt
    rows = []
    for line in (d / "smi.csv").read_text().splitlines() if (d / "smi.csv").exists() else []:
        p = [x.strip() for x in line.split(",")]
        try:
            t = dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append((t, float(p[2].split()[0]), float(p[3].split()[0])))
        except (ValueError, IndexError):
            continue
    out = {}
    for name in ("A", "rl_x", "rl_y"):
        if name == "A":
            w = (ph["A_1"]["start"], max(v["end"] for k, v in ph.items() if k.startswith("A_") and "end" in v))
        elif name in ph and "end" in ph[name]:
            w = (ph[name]["start"], ph[name]["end"])
        else:
            continue
        xs = [m for t, m, _ in rows if w[0] <= t <= w[1]]
        us = [u for t, _, u in rows if w[0] <= t <= w[1]]
        out[name] = {"mem_peak_mib": max(xs) if xs else None, "util_mean": st.fmean(us) if us else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = Path(a.dir)
    ph = phases(d)
    A = requests(d, sorted((k for k in ph if k.startswith("A_") and "end" in ph[k]), key=lambda k: int(k[2:])))
    y0, y1 = ph.get("rl_y", {}).get("start"), ph.get("rl_y", {}).get("end")
    ych = sorted((k for k in ph if k.startswith("Y_") and "end" in ph[k]), key=lambda k: int(k[2:]))
    inside = [k for k in ych if y0 is not None and y1 is not None and ph[k]["start"] >= y0 and ph[k]["end"] <= y1]
    Y = requests(d, inside)
    X, Ry = rl(d / "rl_x", a.steps), rl(d / "rl_y", a.steps)
    tx = ph["rl_x"]["end"] - ph["rl_x"]["start"] if "end" in ph.get("rl_x", {}) else None
    ty = y1 - y0 if y0 is not None and y1 is not None else None
    v = {"i": {"met": X["steps"] >= a.steps and Ry["steps"] >= a.steps and X["oom_skipped"] == 0 and Ry["oom_skipped"] == 0
                      and A["completed"] == A["sent"] and Y["completed"] == Y["sent"] and Y["chunks"] > 0},
         "ii": {"ratio": Ry["t_step"] / X["t_step"], "met": Ry["t_step"] <= 1.3 * X["t_step"]},
         "iii": {"tpot_p50": Y["tpot_p50"], "ttft_p99": Y["ttft_p99"], "met": Y["tpot_p50"] <= 50 and Y["ttft_p99"] <= 5000}}
    out = {"A": A, "Y": Y, "X_rl": X, "Y_rl": Ry, "y_chunks_scored": inside, "y_chunks_all": ych,
           "wall_s": {"rl_x": tx, "rl_y": ty},
           "gpu_hours": {"colocated": ty / 3600 if ty else None, "separate": 2 * tx / 3600 if tx else None},
           "memory": peak_mem(d, ph), "verdict": v}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
