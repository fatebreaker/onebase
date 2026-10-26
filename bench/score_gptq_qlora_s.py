"""GPTQ-QLORA-S (docs/REGISTRY.md): the GPTQ NF4 merge of QLoRA-trained SmolLM2 adapters (smol_gsm8k_qlora_s<k>, trained
on the NF4 base), R_q against the trained policy (nf4_exact over nf4_base), McNemar against PEFT's merge (nf4_merge).
Prediction (each judged seed, QLORA-GRPO-S / -S2's gate): R_q above PEFT's merge, p <= 0.05. Seeds whose GPTQ merge has
not been evaluated are skipped.
    python3 bench/score_gptq_qlora_s.py [--seeds 0 1 2 3] [--json results/std_bench/gptq_qlora_s_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
SB = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402


def load(f: Path) -> dict:
    return {x["id"]: int(x["correct"]) for x in map(json.loads, f.read_text().splitlines())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    judged = {}
    for f in ("qlora_grpo_smol_verdict.json", "qlora_grpo_smol2_verdict.json"):
        if (SB / f).exists():
            judged.update({int(k): bool(v and v.get("judged")) for k, v in json.loads((SB / f).read_text())["rows"].items()})
    rows, met = {}, []
    for sd in a.seeds:
        g_f = SB / f"smol_gsm8k_qlora_s{sd}_gptq_nf4/bf16_base/gsm8k.jsonl"
        if not g_f.exists() or not judged.get(sd):
            continue
        d = SB / f"smol_gsm8k_qlora_s{sd}"
        b, e, m = (load(d / v / "gsm8k.jsonl") for v in ("nf4_base", "nf4_exact", "nf4_merge"))
        g = load(g_f)
        ret = rr.retention(b, e, b, g, n_boot=10000)
        rp = rr.retention(b, e, b, m, n_boot=10000)
        ids = sorted(set(g) & set(m))
        w, l, p = rr.mcnemar([g[i] for i in ids], [m[i] for i in ids])
        rows[str(sd)] = {"acc": 100 * sum(g.values()) / len(g), "gptq": {"R": ret["R"], "R_ci95": ret["R_ci95"]},
                         "peft": {"R": rp["R"], "R_ci95": rp["R_ci95"]}, "gptq_vs_peft": {"wins": w, "losses": l, "p": p}}
        met.append(ret["R"] > rp["R"] and w > l and p <= 0.05)
        print(f"seed {sd}: GPTQ NF4 R_q {ret['R']:.2f} [{ret['R_ci95'][0]:.2f}, {ret['R_ci95'][1]:.2f}], PEFT {rp['R']:.2f}; "
              f"{w}:{l} p {p:.2g}")
    verdict = f"{'MET' if met and all(met) else 'NOT MET'} ({sum(met)} of {len(met)} judged seeds)" if met else "no data"
    print("verdict:", verdict)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
