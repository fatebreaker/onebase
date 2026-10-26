"""STD-BENCH (docs/REGISTRY.md, registered 2026-10-13 20:59 UTC, before any training or evaluation): the registered rule
on results/std_bench/mistral_<math|code>/<variant>/<bench>.jsonl (bench/std_bench.py; one line per problem: id, correct).
  R = (acc_variant - acc_base) / (acc_bf16 exact - acc_bf16 base), acc_base = the variant's own base (nvfp4_base or
  fp8_base), paired bootstrap 95% CI with 10,000 resamples (mlsys analysis/read_retention.py); a benchmark is judged
  only if the bf16 gain is at least 5 points.
  (i)   the nearest NVFP4 merge keeps less than the exact adapter on the NVFP4 base: McNemar p <= 0.05 with the exact
        adapter ahead
  (ii)  the exact adapter keeps R >= 0.8 on the NVFP4 base and R >= 0.9 on the FP8 base
  (iii) the NVFP4-stored adapter keeps R >= 0.8
  (iv)  the stochastic NVFP4 merge keeps R >= 0.8
FP8 merges are reported, not judged. The MLSys check (descriptive): the exact adapter on vLLM's FP8 engine through vLLM's
LoRA path (mistral_<w>_fp8eng_fused0) and the plugin (_fused1), accuracy difference and McNemar.
--model llama (STD-BENCH-L, registered 2026-10-13 22:30 UTC, before any data): results/std_bench/llama_<w>, plus NVIDIA's
NVFP4 release in llama_<w>_rel/{none,frozen,srfrozen}: (v) the exact adapter on the release keeps R >= 0.8, (vi) the
NVFP4-stored adapter R >= 0.8, (vii) the nearest merge on the release's scales keeps less than the exact adapter
(McNemar p <= 0.05), (viii) the stochastic merge R >= 0.8.
    python bench/score_std_bench.py [math code] [--model mistral|llama] [--json out.json] [--tex out.tex]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

RES = LB / "results/std_bench"
BENCHES = {"math": ("gsm8k", "math500"), "code": ("humaneval", "mbpp"), "gsm8k_rl": ("gsm8k", "math500"),
           "gsm8k_rlb": ("gsm8k", "math500"), "gsm8k_rl_s1": ("gsm8k", "math500"), "gsm8k_rl_s2": ("gsm8k", "math500")}
# STD-BENCH-RL (registered 2026-10-14 01:25 UTC): the GRPO adapter (mistral_gsm8k_rl), judged at a 3-point bf16 gain,
# with prediction (v) the nearest NVFP4 merge keeps R <= 0.5
JUDGE = {"gsm8k_rl": 3.0, "gsm8k_rlb": 3.0, "gsm8k_rl_s1": 3.0, "gsm8k_rl_s2": 3.0}            # STD-BENCH-RL-B (lr 5e-5): the same rule
NAMES = {"gsm8k": "GSM8K", "math500": "MATH-500", "humaneval": "HumanEval", "mbpp": "MBPP"}
ARMS = [("nvfp4_exact", "nvfp4_base"), ("nvfp4_stored", "nvfp4_base"), ("nvfp4_merge", "nvfp4_base"),
        ("nvfp4_srmerge", "nvfp4_base"), ("fp8_exact", "fp8_base"), ("fp8_merge", "fp8_base"),
        ("fp8_srmerge", "fp8_base")]
LABEL = {"bf16_base": "bf16 base", "bf16_exact": "bf16 + adapter", "nvfp4_base": "NVFP4 base",
         "nvfp4_exact": "NVFP4 + exact adapter", "nvfp4_stored": "NVFP4 + NVFP4-stored adapter",
         "nvfp4_merge": "NVFP4 nearest merge", "nvfp4_srmerge": "NVFP4 stochastic merge", "fp8_base": "FP8 base",
         "fp8_exact": "FP8 + exact adapter", "fp8_merge": "FP8 nearest merge", "fp8_srmerge": "FP8 stochastic merge"}
BAR = {"nvfp4_exact": 0.8, "fp8_exact": 0.9, "nvfp4_stored": 0.8, "nvfp4_srmerge": 0.8,
       "rel_exact": 0.8, "rel_stored": 0.8, "rel_srmerge": 0.8}
REL = {"rel_base": "none/bf16_base", "rel_exact": "none/bf16_exact", "rel_stored": "none/bf16_stored",
       "rel_merge": "frozen/bf16_base", "rel_srmerge": "srfrozen/bf16_base"}
REL_LABEL = {"rel_base": "NVIDIA NVFP4 release", "rel_exact": "release + exact adapter",
             "rel_stored": "release + NVFP4-stored adapter", "rel_merge": "release nearest merge",
             "rel_srmerge": "release stochastic merge"}
REL_ARMS = [("rel_exact", "rel_base"), ("rel_stored", "rel_base"), ("rel_merge", "rel_base"), ("rel_srmerge", "rel_base")]
MODEL = "mistral"


def load(d: Path, b: str) -> dict | None:
    f = d / f"{b}.jsonl"
    if not f.exists():
        return None
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())}


def acc(x: dict) -> float:
    return 100 * sum(x.values()) / len(x)


def score(which: str) -> dict:
    root = RES / f"{MODEL}_{which}"
    out = {}
    for b in BENCHES[which]:
        v = {name: load(root / name, b) for name in LABEL}
        v.update({k: load(RES / f"{MODEL}_{which}_rel" / sub, b) for k, sub in REL.items()})
        res = {"acc": {k: acc(x) for k, x in v.items() if x is not None}}
        arms = ARMS + REL_ARMS
        if v["bf16_base"] is None or v["bf16_exact"] is None:
            print(f"{which}/{b}: no bf16 references yet")
            out[b] = res
            continue
        gain = acc(v["bf16_exact"]) - acc(v["bf16_base"])
        res["gain"], res["judged"] = gain, gain >= JUDGE.get(which, 5.0)
        for arm, base in arms:
            if v[arm] is None or v[base] is None:
                continue
            r = rr.retention(v["bf16_base"], v["bf16_exact"], v[base], v[arm], n_boot=10000)
            res[arm] = {"R": r["R"], "R_ci95": r["R_ci95"]}
        if v["nvfp4_merge"] is not None and v["nvfp4_exact"] is not None:
            ids = sorted(set(v["nvfp4_merge"]) & set(v["nvfp4_exact"]))
            w, l, p = rr.mcnemar([v["nvfp4_exact"][i] for i in ids], [v["nvfp4_merge"][i] for i in ids])
            res["i"] = {"exact_wins": w, "merge_wins": l, "p": p, "met": p <= 0.05 and w > l}
        if v["rel_merge"] is not None and v["rel_exact"] is not None:
            ids = sorted(set(v["rel_merge"]) & set(v["rel_exact"]))
            w, l, p = rr.mcnemar([v["rel_exact"][i] for i in ids], [v["rel_merge"][i] for i in ids])
            res["vii"] = {"exact_wins": w, "merge_wins": l, "p": p, "met": p <= 0.05 and w > l}
        for arm, bar in BAR.items():
            if arm in res:
                res[arm]["met"] = res[arm]["R"] >= bar
        if which.startswith("gsm8k_rl") and "nvfp4_merge" in res:
            res["v"] = {"R": res["nvfp4_merge"]["R"], "met": res["nvfp4_merge"]["R"] <= 0.5}
        for f in (0, 1):
            e = load(LB / f"results/std_bench/{MODEL}_{which}_fp8eng_fused{f}/fp8eng_exact", b)
            if e is not None:
                res[f"fp8eng_fused{f}"] = acc(e)
                res.setdefault("fp8eng", {})[f] = e
        if "fp8eng" in res and len(res["fp8eng"]) == 2:
            e0, e1 = res["fp8eng"][0], res["fp8eng"][1]
            ids = sorted(set(e0) & set(e1))
            w, l, p = rr.mcnemar([e1[i] for i in ids], [e0[i] for i in ids])
            res["mlsys_check"] = {"plugin_minus_stock": acc(e1) - acc(e0), "plugin_wins": w, "stock_wins": l, "p": p}
        res.pop("fp8eng", None)
        out[b] = res
        verdict = "judged" if res["judged"] else f"NOT judged (gain under {JUDGE.get(which, 5.0):g} points)"
        print(f"{which}/{b}: bf16 {acc(v['bf16_base']):.1f} -> {acc(v['bf16_exact']):.1f} (gain {gain:+.1f}, {verdict})")
        for arm, _ in arms:
            if arm in res:
                m = res[arm].get("met")
                print(f"   {arm:14s} acc {res['acc'][arm]:5.1f}  R = {res[arm]['R']:.2f} "
                      f"[{res[arm]['R_ci95'][0]:.2f}, {res[arm]['R_ci95'][1]:.2f}]"
                      + ("" if m is None else f"  bar {BAR[arm]} -> {'MET' if m else 'NOT MET'}"))
        if "v" in res:
            print(f"   (v) nearest NVFP4 merge R <= 0.5: R = {res['v']['R']:.2f} -> {'MET' if res['v']['met'] else 'NOT MET'}")
        for k, what in (("i", "exact vs nearest NVFP4 merge"), ("vii", "release: exact vs nearest merge")):
            if k in res:
                print(f"   ({k}) {what}: {res[k]['exact_wins']} vs {res[k]['merge_wins']} discordant, "
                      f"p = {res[k]['p']:.3g} -> {'MET' if res[k]['met'] else 'NOT MET'}")
        if "mlsys_check" in res:
            c = res["mlsys_check"]
            print(f"   MLSys check: plugin - stock = {c['plugin_minus_stock']:+.1f} points ({c['plugin_wins']} vs "
                  f"{c['stock_wins']}, p = {c['p']:.3g})")
    return out


def tex(results: dict, path: str) -> None:
    benches = [(w, b) for w in results for b in BENCHES[w] if b in results.get(w, {})]
    lines = [r"\begin{tabular}{@{}l" + "r" * len(benches) + "@{}}", r"\toprule",
             "variant & " + " & ".join(NAMES[b] + {"gsm8k_rl": " (RL)", "gsm8k_rlb": " (RL, 5x lr)", "gsm8k_rl_s1": " (RL, seed 1)", "gsm8k_rl_s2": " (RL, seed 2)"}.get(w, "")
                                      + ("" if results[w][b].get("judged") else r"$^\dagger$")
                                      for w, b in benches) + r" \\", r"\midrule"]
    labels = dict(LABEL, **REL_LABEL)
    groups = [("bf16_base", "bf16_exact"), ("nvfp4_base", "nvfp4_exact", "nvfp4_stored", "nvfp4_merge", "nvfp4_srmerge"),
              ("fp8_base", "fp8_exact", "fp8_merge", "fp8_srmerge"), tuple(REL_LABEL)]
    blocks = []
    for g in groups:
        rows = []
        for k in g:
            if not any(k in results[w][b]["acc"] for w, b in benches):
                continue
            cells = []
            for w, b in benches:
                r = results[w][b]
                if k not in r["acc"]:
                    cells.append("--")
                elif r.get("judged") and k in r and isinstance(r[k], dict) and "R" in r[k]:
                    cells.append(f"{r['acc'][k]:.1f} ({r[k]['R']:.2f})")
                else:
                    cells.append(f"{r['acc'][k]:.1f}")
            rows.append(f"{labels[k]} & " + " & ".join(cells) + r" \\")
        if rows:
            blocks.append("\n".join(rows))
    lines += ["\n\\midrule\n".join(blocks), r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines).replace("-0.00", "0.00").replace("(-", "($-$") + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["math", "code"], help="math, code, gsm8k_rl")
    ap.add_argument("--model", default="mistral", choices=("mistral", "llama"))
    ap.add_argument("--json", default="")
    ap.add_argument("--tex", default="")
    a = ap.parse_args()
    global MODEL
    MODEL = a.model
    results = {w: score(w) for w in a.which}
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))
    if a.tex:
        tex(results, a.tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
