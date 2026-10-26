"""GPTQ-SELF (docs/REGISTRY.md): GPTQ merges calibrated on the model's own greedy GSM8K completions (<adapter>_gptqself_<fmt>)
against the same merges calibrated on GSM8K's reference answers (<adapter>_gptq_<fmt>), GSM8K. R = (acc - acc_<fmt>_base) /
(acc_bf16_exact - acc_bf16_base) (paired bootstrap 10,000) from each adapter's own references.
  (i)  SmolLM2 seeds 4, 5 (H100): pooled over the two seeds and the NVFP4 and NF4 grids, the self-calibrated merge wins more
       discordant problems than it loses against the reference-calibrated one (two-sided exact binomial p <= 0.05).
  (ii) the four Mistral GRPO adapters (A100): the same, pooled over the four adapters and both grids.
FP8 (where built) is descriptive. Parts whose merges are not all evaluated are reported as incomplete.
GPTQ-SELF-NF4 (confirmatory, registered 2026-10-15 01:36 UTC): --part nf4c: SmolLM2 smol_gsm8k_rlb_s0, _s1 and the four
QLoRA-trained Mistral adapters mistral_gsm8k_qlora_s0..s3, NF4 only; pooled over the six, the self-calibrated merge wins
more discordant GSM8K problems than the reference-calibrated one (two-sided exact binomial p <= 0.05).
    python3 bench/score_gptq_self.py [--json results/std_bench/gptq_self_verdict.json]"""
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

PARTS = {"i": [f"smol_gsm8k_rlb_s{k}" for k in (4, 5)],
         "ii": ["mistral_gsm8k_rl", "mistral_gsm8k_rl_s1", "mistral_gsm8k_rl_s2", "mistral_gsm8k_rlb"]}
NF4C = [f"smol_gsm8k_rlb_s{k}" for k in (0, 1)] + [f"mistral_gsm8k_qlora_s{k}" for k in range(4)]


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def binom_two_sided(w: int, l: int) -> float:
    n = w + l
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--part", default="registered", choices=("registered", "nf4c"))
    a = ap.parse_args()
    out = {"rows": {}, "verdict": {}}
    if a.part == "nf4c":                                         # GPTQ-SELF-NF4: discordant problems only (R descriptive)
        W = L = 0
        for n in NF4C:
            s_, g = load(RES / f"{n}_gptqself_nf4/bf16_base/gsm8k.jsonl"), load(RES / f"{n}_gptq_nf4/bf16_base/gsm8k.jsonl")
            if s_ is None or g is None:
                print(f"{n}: missing"); out["verdict"]["nf4c"] = {"verdict": "incomplete"}
                break
            ids = sorted(set(s_) & set(g))
            w, l, p = rr.mcnemar([s_[i] for i in ids], [g[i] for i in ids])
            acc = lambda d: 100 * sum(d[i] for i in ids) / len(ids)  # noqa: E731
            out["rows"][f"{n}:nf4"] = {"self_acc": acc(s_), "reference_acc": acc(g), "self_vs_reference": {"wins": w, "losses": l, "p": p}}
            W, L = W + w, L + l
            print(f"{n} nf4: self {acc(s_):.1f}% vs reference {acc(g):.1f}% ({w}:{l}, p {p:.2g})")
        else:
            p = binom_two_sided(W, L)
            out["verdict"]["nf4c"] = {"wins": W, "losses": L, "p": p, "verdict": "MET" if W > L and p <= 0.05 else "NOT MET"}
            print(f"(nf4c) pooled: self vs reference {W}:{L}, exact binomial p {p:.2g} -> {out['verdict']['nf4c']['verdict']}")
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))
        return 0
    for part, names in PARTS.items():
        W = L = 0
        complete = True
        for n in names:
            ref = {k: load(RES / n / k / "gsm8k.jsonl") for k in ("bf16_base", "bf16_exact", "nvfp4_base", "nf4_base", "fp8_base")}
            for fmt in ("nvfp4", "nf4", "fp8"):
                s, g = load(RES / f"{n}_gptqself_{fmt}/bf16_base/gsm8k.jsonl"), load(RES / f"{n}_gptq_{fmt}/bf16_base/gsm8k.jsonl")
                if s is None or g is None or ref[f"{fmt}_base"] is None:
                    if fmt != "fp8":
                        complete = False
                    continue
                rs = rr.retention(ref["bf16_base"], ref["bf16_exact"], ref[f"{fmt}_base"], s, n_boot=10000)
                rg = rr.retention(ref["bf16_base"], ref["bf16_exact"], ref[f"{fmt}_base"], g, n_boot=10000)
                ids = sorted(set(s) & set(g))
                w, l, p = rr.mcnemar([s[i] for i in ids], [g[i] for i in ids])
                out["rows"][f"{n}:{fmt}"] = {"self": {"R": rs["R"], "R_ci95": rs["R_ci95"]},
                                             "reference": {"R": rg["R"], "R_ci95": rg["R_ci95"]},
                                             "self_vs_reference": {"wins": w, "losses": l, "p": p}}
                if fmt != "fp8":
                    W, L = W + w, L + l
                print(f"{n} {fmt}: self R {rs['R']:.2f} [{rs['R_ci95'][0]:.2f}, {rs['R_ci95'][1]:.2f}], reference "
                      f"{rg['R']:.2f}; self vs reference {w}:{l} p {p:.2g}")
        if complete:
            p = binom_two_sided(W, L)
            out["verdict"][part] = {"wins": W, "losses": L, "p": p,
                                    "verdict": "MET" if W > L and p <= 0.05 else "NOT MET"}
            print(f"({part}) pooled NVFP4+NF4: self vs reference {W}:{L}, exact binomial p {p:.2g} -> "
                  f"{out['verdict'][part]['verdict']}")
        else:
            out["verdict"][part] = {"verdict": "incomplete"}
            print(f"({part}) incomplete")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
