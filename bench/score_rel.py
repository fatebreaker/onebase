"""The release tests on popular tasks (docs/REGISTRY.md), one rule for each: MATH-RL-REL (Llama-3.1-8B GRPO on MATH, MATH-500),
CODE-RL (GRPO on MBPP; MBPP and HumanEval judged separately), GRANITE-RL (Granite-4.2-8B on DAPO-Math; IBM's NVFP4 release for
(i), (ii), (iv); (iii) is the nearest merge against the exact adapter on IBM's FP8 and MXFP4 releases, p <= 0.05, no R bar),
QMATH-CONV, TIR-RL and CODE-Q (Qwen3.5-9B adapters on our conversions; (i) our NVFP4 merge and (ii) PEFT's NF4 merge by
rule (i), (iii) the exact adapter on our NVFP4 base keeps R >= 0.8), Q27-RL (Qwen3.8-27B: (i) the nearest merge below the
exact adapter on Qwen's FP8 release and on NVIDIA's mixed release, p <= 0.05; (ii) the exact adapter's R >= 0.8 on each;
(iii) our NVFP4 merge by rule (i); (iv) GPTQ above nearest on the mixed release), TOOL-RL (GRPO on ToolACE; BFCL's pooled AST categories,
irrelevance descriptive) and HARD-MATH (b) (MATH-RL-REL's variants on OlympiadBench and Minerva pooled).
For each seed's adapter, judged on an outcome only if its bf16 gain there is at least 3 points (the gate):
  (i)   the nearest merge on NVIDIA's NVFP4 release keeps R <= 0.5 and less than the exact adapter on the release
        (McNemar p <= 0.05, exact ahead)
  (ii)  the exact adapter on the release keeps R >= 0.8
  (iii) (i) for our NVFP4 conversion (nvfp4_merge against nvfp4_exact) and PEFT's NF4 merge (nf4_merge against nf4_exact)
  (iv)  the GPTQ merge on the release keeps more than the nearest merge there (McNemar p <= 0.05, GPTQ ahead)
R = (acc_variant - acc_its_base) / (acc_bf16_exact - acc_bf16_base), paired bootstrap 95% CI (mlsys analysis/
read_retention.py, 10,000 resamples). Descriptive: FP8 exact and merge, the stochastic merge on the release, the
NVFP4-stored adapter on the release.
Files: results/std_bench/<run>/<variant>/<bench>.jsonl and <run>_rel/{none,frozen,srfrozen,gptq}/<variant>/<bench>.jsonl.
    python3 bench/score_rel.py --test <mathrel|hardmath|coderl|toolrl|qtool|qmath|granite|tir|codeq|q27> [--seeds 0 1 2] [--json out.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

# test -> (run name for seed k, release-run name, [(outcome label, [(bench file, category filter or None)])])
TESTS = {
    "mathrel": ("llama_math_S0_s{k}_a100", "llama_math_S0_s{k}_rel", [("MATH-500", [("math500", None)])]),
    "hardmath": ("llama_math_S0_s{k}_a100_hard", "llama_math_S0_s{k}_rel_hard",
                 [("OlympiadBench+Minerva", [("olympiad", None), ("minerva", None)])]),
    "coderl": ("llama_code_rl_s{k}", "llama_code_rl_s{k}_rel", [("MBPP", [("mbpp", None)]), ("HumanEval", [("humaneval", None)])]),
    "toolrl": ("llama_tool_rl_s{k}", "llama_tool_rl_s{k}_rel", [("BFCL AST", [("bfcl", "ast")])]),
    "qtool": ("qwen9_tool_s{k}", "qwen9_tool_s{k}_rel", [("BFCL AST", [("bfcl", "ast")])]),   # QTOOL-RL: no release
    # QMATH-CONV: no official release; (i) our NVFP4 merge, (ii) PEFT's NF4 merge, (iii) the exact adapter on our NVFP4
    "qmath": ("qwen9_dapo_S0G_s{k}_conv", "qwen9_dapo_S0G_s{k}_norelease",
              [("MATH-500+OlympiadBench+Minerva", [("math500", None), ("olympiad", None), ("minerva", None)])]),
    "granite": ("granite8_dapo_s{k}", "granite8_dapo_s{k}_rel_nvfp4", [("MATH-500+OlympiadBench+Minerva", [("math500", None), ("olympiad", None), ("minerva", None)])]),
    "tir": ("qwen9_tir_s{k}", "qwen9_tir_s{k}_norelease", [("MATH-500+OlympiadBench+Minerva", [("math500", None), ("olympiad", None), ("minerva", None)])]),
    "codeq": ("qwen9_code_s{k}", "qwen9_code_s{k}_norelease", [("LiveCodeBench", [("lcb", None)])]),
    # QMATH-CONV-SEEDS: RL-FP8-QMATH-TUNED's S0T adapters (S0G's recipe, the engine asleep during training)
    "qmath_s0t": ("qwen9_dapo_S0T_s{k}_conv", "qwen9_dapo_S0T_s{k}_norelease",
                  [("MATH-500+OlympiadBench+Minerva", [("math500", None), ("olympiad", None), ("minerva", None)])]),
    # RECIPE-2: the S0G recipe with a 5x learning rate; QMATH-CONV's predictions plus (iv) R of our NVFP4 merge > 0.06
    "qmathr2": ("qwen9_dapo_R2_s{k}_conv", "qwen9_dapo_R2_s{k}_norelease",
                [("MATH-500+OlympiadBench+Minerva", [("math500", None), ("olympiad", None), ("minerva", None)])]),
    # OLMO-RL: OLMo-3-7B-Instruct, no official low-precision release; predictions as QMATH-CONV's
    "olmo": ("olmo7_dapo_s{k}", "olmo7_dapo_s{k}_norelease", [("MATH-500+OlympiadBench+Minerva", [("math500", None), ("olympiad", None), ("minerva", None)])]),
    # Q27-RL: Qwen's FP8 release (rel_*), NVIDIA's mixed FP8/NVFP4 release (relnv_*), our NVFP4 conversion
    "q27": ("qwen27_dapo_s{k}", "qwen27_dapo_s{k}_rel", [("MATH-500+OlympiadBench+Minerva", [("math500", None), ("olympiad", None), ("minerva", None)])]),
}
CONV_TESTS = {"qmath", "tir", "codeq", "olmo", "qmathr2", "qmath_s0t"}   # no release: (i) our NVFP4 merge, (ii) PEFT's NF4 merge, (iii) exact on our NVFP4
# GRANITE-RL (iii): IBM's FP8 and MXFP4 releases, the nearest merge against the exact adapter on the same release (no R bar);
# for GRANITE-RL our conversions are descriptive
EXTRA_REL = {"granite": {"fp8": "granite8_dapo_s{k}_rel_fp8", "mxfp4": "granite8_dapo_s{k}_rel_mxfp4"},
             "q27": {"nv": "qwen27_dapo_s{k}_nvrel"}}


def load(d: Path, files) -> dict | None:
    """id -> 0/1 over the outcome's files (None if any is missing)."""
    out = {}
    for bench, cat in files:
        f = d / f"{bench}.jsonl"
        if not f.exists():
            return None
        for r in map(json.loads, f.read_text().splitlines()):
            if cat == "ast" and "irrelevance" in r.get("category", ""):
                continue
            out[f"{bench}:{r['id']}"] = int(r["correct"])
    return out


def ret(base, ex, vb, v):
    r = rr.retention(base, ex, vb, v, n_boot=10000)
    return {"R": r["R"], "R_ci95": r["R_ci95"], "acc": 100 * r["acc"]["rl_q"], "acc_base": 100 * r["acc"]["base_q"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, choices=sorted(TESTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--json", default="")
    ap.add_argument("--lcb-file", default="lcb", help="CODE-Q-REGRADE: read <variant>/<this>.jsonl for LiveCodeBench")
    a = ap.parse_args()
    run_t, rel_t, outcomes = TESTS[a.test]
    outcomes = [(lab, [(a.lcb_file if b == "lcb" else b, c) for b, c in files]) for lab, files in outcomes]
    res = {}
    for k in a.seeds:
        run, rel = RES / run_t.format(k=k), RES / rel_t.format(k=k)
        for label, files in outcomes:
            V = {v: load(run / v, files) for v in ("bf16_base", "bf16_exact", "nvfp4_base", "nvfp4_exact", "nvfp4_merge",
                                                   "nvfp4_srmerge", "nf4_base", "nf4_exact", "nf4_merge", "fp8_base",
                                                   "fp8_exact", "fp8_merge")}
            V.update({"rel_base": load(rel / "none/bf16_base", files), "rel_exact": load(rel / "none/bf16_exact", files),
                      "rel_stored": load(rel / "none/bf16_stored", files), "rel_merge": load(rel / "frozen/bf16_base", files),
                      "rel_sr": load(rel / "srfrozen/bf16_base", files), "rel_gptq": load(rel / "gptq/bf16_base", files)})
            for r, t in EXTRA_REL.get(a.test, {}).items():
                d = RES / t.format(k=k)
                V.update({f"rel{r}_base": load(d / "none/bf16_base", files), f"rel{r}_exact": load(d / "none/bf16_exact", files),
                          f"rel{r}_merge": load(d / "frozen/bf16_base", files), f"rel{r}_gptq": load(d / "gptq/bf16_base", files)})
            key = f"s{k}:{label}"
            if V["bf16_base"] is None or V["bf16_exact"] is None:
                print(f"{key}: bf16 runs missing")
                continue
            ids = sorted(V["bf16_base"])
            acc = lambda x: 100 * sum(x[i] for i in ids if i in x) / len(ids)  # noqa: E731
            gain = acc(V["bf16_exact"]) - acc(V["bf16_base"])
            e = {"n": len(ids), "bf16_base": acc(V["bf16_base"]), "bf16_exact": acc(V["bf16_exact"]), "gain": gain,
                 "judged": gain >= 3.0, "variants": {}, "pred": {}}
            B, X = V["bf16_base"], V["bf16_exact"]
            for v, vb in (("nvfp4_exact", "nvfp4_base"), ("nvfp4_merge", "nvfp4_base"), ("nvfp4_srmerge", "nvfp4_base"),
                          ("nf4_exact", "nf4_base"),
                          ("nf4_merge", "nf4_base"), ("fp8_exact", "fp8_base"), ("fp8_merge", "fp8_base"),
                          ("rel_exact", "rel_base"), ("rel_stored", "rel_base"), ("rel_merge", "rel_base"),
                          ("rel_sr", "rel_base"), ("rel_gptq", "rel_base")):
                if V[v] is not None and V[vb] is not None:
                    e["variants"][v] = ret(B, X, V[vb], V[v])
            for r in EXTRA_REL.get(a.test, {}):
                for v in (f"rel{r}_exact", f"rel{r}_merge", f"rel{r}_gptq"):
                    if V[v] is not None and V[f"rel{r}_base"] is not None:
                        e["variants"][v] = ret(B, X, V[f"rel{r}_base"], V[v])

            def worse(merge, exact):
                """McNemar of the merge against the exact adapter on the same base: (exact wins, merge wins, p)."""
                if V[merge] is None or V[exact] is None:
                    return None
                common = sorted(set(V[merge]) & set(V[exact]))
                return rr.mcnemar([V[exact][i] for i in common], [V[merge][i] for i in common])
            def rule_i(merge, exact):
                if merge not in e["variants"]:
                    return None
                w = worse(merge, exact)
                return {"R": e["variants"][merge]["R"], "exact_vs_merge": w,
                        "met": e["variants"][merge]["R"] <= 0.5 and w is not None and w[0] > w[1] and w[2] <= 0.05}
            if e["judged"] and a.test == "q27":                  # Q27-RL and its mixed-release amendment
                for tag, r in (("fp8_release", ""), ("nv_release", "nv")):
                    w = worse(f"rel{r}_merge", f"rel{r}_exact")
                    e["pred"][f"i_{tag}"] = None if w is None else {"exact_vs_merge": w, "met": w[0] > w[1] and w[2] <= 0.05}
                    x = e["variants"].get(f"rel{r}_exact")
                    e["pred"][f"ii_{tag}"] = None if x is None else {"R": x["R"], "met": x["R"] >= 0.8}
                e["pred"]["iii"] = rule_i("nvfp4_merge", "nvfp4_exact")
                g = worse("relnv_merge", "relnv_gptq")
                e["pred"]["iv"] = None if g is None else {"gptq_vs_nearest": g, "met": g[0] > g[1] and g[2] <= 0.05}
            elif e["judged"] and a.test in CONV_TESTS:          # QMATH-CONV, TIR-RL, CODE-Q: three predictions
                e["pred"]["i"] = rule_i("nvfp4_merge", "nvfp4_exact")
                e["pred"]["ii"] = rule_i("nf4_merge", "nf4_exact")
                e["pred"]["iii"] = ({"R": e["variants"]["nvfp4_exact"]["R"], "met": e["variants"]["nvfp4_exact"]["R"] >= 0.8}
                                    if "nvfp4_exact" in e["variants"] else None)
                if a.test == "qmathr2":                          # RECIPE-2 (iv): larger updates survive the merge more
                    m = e["variants"].get("nvfp4_merge")
                    e["pred"]["iv"] = None if m is None else {"R": m["R"], "bar": 0.06, "met": m["R"] > 0.06}
            elif e["judged"]:
                e["pred"]["i"] = rule_i("rel_merge", "rel_exact")
                e["pred"]["ii"] = ({"R": e["variants"]["rel_exact"]["R"], "met": e["variants"]["rel_exact"]["R"] >= 0.8}
                                   if "rel_exact" in e["variants"] else None)
                if a.test in EXTRA_REL:                          # GRANITE-RL (iii); our conversions descriptive
                    for r in EXTRA_REL[a.test]:
                        w = worse(f"rel{r}_merge", f"rel{r}_exact")
                        e["pred"][f"iii_{r}_release"] = None if w is None else {"exact_vs_merge": w, "met": w[0] > w[1] and w[2] <= 0.05}
                    e["desc_conv"] = {"nvfp4": rule_i("nvfp4_merge", "nvfp4_exact"), "nf4": rule_i("nf4_merge", "nf4_exact")}
                else:
                    e["pred"]["iii_nvfp4"] = rule_i("nvfp4_merge", "nvfp4_exact")
                    e["pred"]["iii_nf4"] = rule_i("nf4_merge", "nf4_exact")
                g = worse("rel_merge", "rel_gptq")                 # (gptq wins, nearest wins, p)
                e["pred"]["iv"] = None if g is None else {"gptq_vs_nearest": g, "met": g[0] > g[1] and g[2] <= 0.05}
            res[key] = e
            line = (f"{key}: n {e['n']}, bf16 {e['bf16_base']:.1f} -> {e['bf16_exact']:.1f} (gain {gain:+.1f}, "
                    f"{'judged' if e['judged'] else 'not judged'}); " + ", ".join(
                        f"{v} R {x['R']:.2f} [{x['R_ci95'][0]:.2f}, {x['R_ci95'][1]:.2f}]" for v, x in e["variants"].items()))
            print(line)
            for p, x in e["pred"].items():
                print(f"   ({p}) {'MET' if x and x['met'] else ('NOT MET' if x else 'no data')}: {json.dumps(x)}")
    if a.test in ("toolrl", "qtool"):                            # irrelevance, descriptive
        for k in a.seeds:
            for v in ("bf16_base", "bf16_exact"):
                f = RES / run_t.format(k=k) / v / "bfcl.jsonl"
                if f.exists():
                    rows = [r for r in map(json.loads, f.read_text().splitlines()) if "irrelevance" in r["category"]]
                    print(f"s{k} {v} irrelevance {100 * sum(r['correct'] for r in rows) / len(rows):.1f}% of {len(rows)}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
