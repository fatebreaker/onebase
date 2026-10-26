"""GPTQ-ACT (docs/REGISTRY.md): act-order GPTQ merges onto the frozen grids (<adapter>_gptqact_<fmt>) against the plain GPTQ
merges (<adapter>_gptq_<fmt>), GSM8K; output-space statistics on the calibration inputs from logs/gptqact_<adapter>_<fmt>.json
(bench/gptq_merge_ckpt2.py --compare: the act-order merge, the plain one and the nearest one, per module).
  (i)  NF4, output space: the act-order merge's median module output gain exceeds the plain merge's for >= 3 of the 4
       Mistral GRPO adapters.
  (ii) NF4, behavior: pooled over the four Mistral adapters, act-order wins more discordant problems than it loses against
       the plain merge (two-sided exact binomial p <= 0.05).
Descriptive: R per adapter and grid (paired bootstrap 10,000, each adapter's own references), McNemar against the plain
merge, NVFP4, FP8 and SmolLM2 (seeds 4, 5 on the H100; 0-3 on the A100s).
    python3 bench/score_gptq_act.py [--json results/std_bench/gptq_act_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

MIS = ["mistral_gsm8k_rl", "mistral_gsm8k_rl_s1", "mistral_gsm8k_rl_s2", "mistral_gsm8k_rlb"]
SMOL = [f"smol_gsm8k_rlb_s{k}" for k in (4, 5, 0, 1, 2, 3)]


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def binom_two_sided(w: int, l: int) -> float:
    n, k = w + l, min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def row(n: str, fmt: str) -> dict | None:
    act = load(RES / f"{n}_gptqact_{fmt}/bf16_base/gsm8k.jsonl")
    if act is None:
        return None
    ref = {k: load(RES / n / k / "gsm8k.jsonl") for k in ("bf16_base", "bf16_exact", f"{fmt}_base")}
    out = {}
    ra = rr.retention(ref["bf16_base"], ref["bf16_exact"], ref[f"{fmt}_base"], act, n_boot=10000)
    out["act"] = {"R": ra["R"], "R_ci95": ra["R_ci95"]}
    plain = load(RES / f"{n}_gptq_{fmt}/bf16_base/gsm8k.jsonl")
    if plain is not None:
        rp = rr.retention(ref["bf16_base"], ref["bf16_exact"], ref[f"{fmt}_base"], plain, n_boot=10000)
        ids = sorted(set(act) & set(plain))
        w, l, p = rr.mcnemar([act[i] for i in ids], [plain[i] for i in ids])
        out["plain"] = {"R": rp["R"], "R_ci95": rp["R_ci95"]}
        out["act_vs_plain"] = {"wins": w, "losses": l, "p": p}
    st = LB / f"logs/gptqact_{n}_{fmt}.json"
    if st.exists():
        s = json.loads(st.read_text())
        out["out_gain"] = {"act": s["out_gain_median"], "plain": s["other_out_gain_median"],
                           "nearest": s["nearest_out_gain_median"]}
        out["out_err"] = {"act": s["out_err_median"], "plain": s["other_out_err_median"],
                          "nearest": s["nearest_out_err_median"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = {f"{n}:{fmt}": row(n, fmt) for n in MIS + SMOL for fmt in ("nvfp4", "nf4", "fp8")}
    rows = {k: v for k, v in rows.items() if v is not None}
    for k, r in rows.items():
        txt = f"{k}: act R {r['act']['R']:.2f} [{r['act']['R_ci95'][0]:.2f}, {r['act']['R_ci95'][1]:.2f}]"
        if "plain" in r:
            m = r["act_vs_plain"]
            txt += f", plain {r['plain']['R']:.2f}; act vs plain {m['wins']}:{m['losses']} p {m['p']:.2g}"
        if "out_gain" in r:
            g, e = r["out_gain"], r["out_err"]
            txt += (f" | output gain act {g['act']:.3f} plain {g['plain']:.3f} nearest {g['nearest']:.3f}; "
                    f"error act {e['act']:.3f} plain {e['plain']:.3f}")
        print(txt)
    verdict = {}
    nf4 = [rows.get(f"{n}:nf4") for n in MIS]
    if all(r and "out_gain" in r and "act_vs_plain" in r for r in nf4):
        k = sum(r["out_gain"]["act"] > r["out_gain"]["plain"] for r in nf4)
        verdict["i"] = f"{'MET' if k >= 3 else 'NOT MET'} ({k} of 4)"
        W, L = sum(r["act_vs_plain"]["wins"] for r in nf4), sum(r["act_vs_plain"]["losses"] for r in nf4)
        p = binom_two_sided(W, L)
        verdict["ii"] = f"{'MET' if W > L and p <= 0.05 else 'NOT MET'} (pooled {W}:{L}, exact binomial p {p:.2g})"
    else:
        verdict = {"i": "incomplete", "ii": "incomplete"}
    print("verdict:", verdict)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
