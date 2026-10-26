"""Score the registered Phase-0 rules (docs/REGISTRY.md) from results/. CPU, seconds.

    python3 bench/score_p0.py [--results results]
P0-OVERHEAD: per (GPU, model, base), o_thr = 1 - thr(LoRA)/thr(ref) on W2 and W3, o_lat = TPOT(LoRA)/TPOT(ref) - 1 on
  W1, with ref = the same engine without LoRA; best stock = the lower overhead of sync and dual-stream. MET if, with an
  FP8 base on sm_89/sm_90, >= 2 (GPU, model) configurations show o_thr(W2) >= 0.10 or o_lat(W1) >= 0.20 at (64,1) or
  (16,32).
P0-KERNEL: per (GPU, model, r=64, M in 32..256), the median rho over the four projections must be >= 0.5 for every M,
  on H100 or L40S.
P0-NUMERICS: per subject, lora_fp8 kappa <= 0.05 and g in [0.95, 1.05]; the control must have kappa < 0.01.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

FP8_CC = {"8.9", "9.0"}


def load(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def score_overhead(rows: list[dict]) -> None:
    print("== P0-OVERHEAD")
    if not rows:
        print("no data")
        return
    by = defaultdict(dict)
    for r in rows:
        key = (r["gpu"], r["cc"], r["model"], r["quantization"])
        tag = "ref" if not r["lora"] else f"r{r['rank']}:n{r['n_adapters']}" + (":dual" if r["dual_stream"] else "")
        by[key][tag] = r                                  # a later repeat of a configuration overrides
    passing = set()
    for key, d in sorted(by.items()):
        ref = d.get("ref")
        if not ref:
            print(f"{key}: no reference run")
            continue
        print(f"{key[0]} (sm {key[1]}) {key[2]} {key[3]}: ref W1 {ref.get('W1_tpot_ms')} ms, W2 {ref.get('W2_gen_tok_s')} "
              f"tok/s, W3 {ref.get('W3_tok_s')} tok/s")
        for tag, r in sorted(d.items()):
            if tag == "ref":
                continue
            o_lat = r["W1_tpot_ms"] / ref["W1_tpot_ms"] - 1 if "W1_tpot_ms" in r and "W1_tpot_ms" in ref else None
            o_w2 = 1 - r["W2_gen_tok_s"] / ref["W2_gen_tok_s"] if "W2_gen_tok_s" in r and "W2_gen_tok_s" in ref else None
            o_w3 = 1 - r["W3_tok_s"] / ref["W3_tok_s"] if "W3_tok_s" in r and "W3_tok_s" in ref else None
            fmt = lambda v: "   -  " if v is None else f"{100 * v:+6.1f}%"  # noqa: E731
            print(f"    {tag:14s} o_lat(W1) {fmt(o_lat)}  o_thr(W2) {fmt(o_w2)}  o_thr(W3) {fmt(o_w3)}")
        if key[3] != "fp8" or key[1] not in FP8_CC:
            continue
        for setting in ("r64:n1", "r16:n32"):
            cands = [d[t] for t in (setting, setting + ":dual") if t in d]
            if not cands:
                continue
            o_lat = min(c["W1_tpot_ms"] / ref["W1_tpot_ms"] - 1 for c in cands if "W1_tpot_ms" in c)
            o_w2 = min(1 - c["W2_gen_tok_s"] / ref["W2_gen_tok_s"] for c in cands if "W2_gen_tok_s" in c)
            if o_w2 >= 0.10 or o_lat >= 0.20:
                passing.add((key[0], key[2]))
            print(f"    best stock {setting}: o_lat {100 * o_lat:+.1f}%  o_thr(W2) {100 * o_w2:+.1f}%")
    print(f"configurations meeting the bar: {sorted(passing)}")
    print(f"**P0-OVERHEAD {'MET' if len(passing) >= 2 else 'NOT MET (so far)'}**")


MODEL_SHAPES = {"qwen3_8b": [(4096, 6144), (4096, 4096), (4096, 24576), (12288, 4096)],      # (K, N): qkv, o, gate_up, down
                "llama31_8b": [(4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096)]}


def score_kernel(rows: list[dict]) -> None:
    """Per (GPU, model, r, M): the median rho over the model's four projections, looked up by shape (the benchmark
    measures a shape shared by two models once). MET if on one GPU (sm_89/sm_90) both models have median rho >= 0.5 at
    every M in 32..256 for r = 64."""
    print("\n== P0-KERNEL")
    if not rows:
        print("no data")
        return
    rec = {(r["gpu"], r["cc"], r["K"], r["N"], r["r"], r["M"]): r for r in rows}
    gpus = sorted({(r["gpu"], r["cc"]) for r in rows})
    met_any = False
    for gpu, cc in gpus:
        model_ok = []
        for model, shapes in MODEL_SHAPES.items():
            for rank in sorted({r["r"] for r in rows}):
                line, ok_all = [], True
                for M in sorted({r["M"] for r in rows}):
                    rs = [rec[(gpu, cc, K, N, rank, M)].get("rho") for K, N in shapes if (gpu, cc, K, N, rank, M) in rec]
                    med = statistics.median(rs) if len(rs) == 4 and None not in rs else None
                    line.append(f"M={M}: {'-' if med is None else f'{med:.2f}'}")
                    if rank == 64 and 32 <= M <= 256:
                        ok_all &= med is not None and med >= 0.5
                gated = rank == 64 and cc in FP8_CC
                print(f"{gpu} {model} r={rank}: " + "  ".join(line) + ("  <- gated" if gated else ""))
                if gated:
                    model_ok.append(ok_all)
        if len(model_ok) == 2 and all(model_ok):
            met_any = True
    print(f"**P0-KERNEL {'MET' if met_any else 'NOT MET (so far)'}**")


def score_numerics(d: Path) -> None:
    """Addendum 3: kappa is scored net of the control (an unchanged copy of the adapter), kappa_net = kappa - kappa_ctl;
    the run is valid if the control's logit gain lies in [0.98, 1.02]. Add-on runs (<S>_<tag>.json) carry their own
    control and are reported with the subject."""
    print("\n== P0-NUMERICS (kappa net of the control; addendum 3)")
    met = []
    for subj in ("S1", "S2"):
        files = sorted(d.glob(f"{subj}.json")) + sorted(d.glob(f"{subj}_*.json"))
        files = [f for f in files if not f.name.endswith("_quant.json")]
        if not files:
            print(f"{subj}: no data")
            met.append(False)
            continue
        res, valid = {}, True
        for f in files:
            j = json.loads(f.read_text())
            kb = j["variants"]["none:base"]["topk_kl"]
            rows = {}
            for v, x in j["variants"].items():
                if v.startswith("lora="):
                    rows[v.rstrip("/").rsplit("/", 1)[1]] = (x["topk_kl"] / kb, j["logit_gain"].get(v, {}))
            k_ctl, g_ctl = rows["copy"][0], rows["copy"][1].get("gain", float("nan"))
            ok_ctl = 0.98 <= g_ctl <= 1.02
            valid &= ok_ctl
            print(f"{subj} [{f.name}] control kappa {k_ctl:.4f} g {g_ctl:.3f} -> {'valid' if ok_ctl else 'INVALID'}")
            for name, (k, g) in rows.items():
                if name == "copy":
                    continue
                res[name] = (k - k_ctl, g.get("gain"))
                print(f"    {name:7s} kappa {k:.4f}  kappa_net {k - k_ctl:+.4f}  g {g.get('gain', float('nan')):.3f}  "
                      f"(CI {g.get('gain_ci95')}, corr {g.get('corr', float('nan')):.3f})")
        kn, g8 = res.get("fp8", (None, None))
        ok = valid and kn is not None and kn <= 0.05 and g8 is not None and 0.95 <= g8 <= 1.05
        print(f"{subj}: lora_fp8 {'meets' if ok else 'misses'} the bar")
        met.append(ok)
    print(f"**P0-NUMERICS {'MET' if all(met) else 'NOT MET (so far)'}**")


def score_accuracy_logits(d: Path, adapters=range(1, 7), part: str = "(b)", prefix: str = "a") -> None:
    """P0-ACCURACY (b): per registered adapter (addendum 2), the NVFP4-stored adapter on the bf16 base (g >= 0.9 and
    kappa_net <= 0.1) and the on-grid NVFP4 merge (g <= 0.5, paired with the NVFP4 base). MET on >= 5 of 6 adapters.
    Descriptive: the exact and the NVFP4-stored adapter on the NVFP4 base.
    part "(b2)" (addendum 6, SFT adapters 7-8): MET if the stored adapter passes on both; the merge is not ruled.
    part "(b4)" (addendum 8, public adapters q1 q2 q4 q5): MET if >= 3 of the 4 controls are valid and the stored
    adapter passes on every adapter with a valid control; the merge is not ruled."""
    print(f"\n== P0-ACCURACY {part}, logit level (kappa net of the run's control)")
    need = len(adapters) if part != "(b3)" else 5   # (b3): >= 5 of the 6 public adapters with a valid control
    passes, n, n_valid = 0, 0, 0
    for a in adapters:
        f = d / f"{prefix}{a}_logits.json"
        if not f.exists():
            print(f"adapter {a}: no data")
            continue
        n += 1
        j = json.loads(f.read_text())
        kb = j["variants"]["none:base"]["topk_kl"]
        def get(pred):
            for v, x in j["variants"].items():
                if pred(v):
                    return v, x["topk_kl"] / kb, j["logit_gain"].get(v, {}).get("gain")
            return None, None, None
        _, k_ctl, g_ctl = get(lambda v: v.startswith("lora=") and v.rstrip("/").endswith("/copy"))
        _, k_n4, g_n4 = get(lambda v: v.startswith("lora=") and v.rstrip("/").endswith("/nvfp4"))
        _, k_fr, g_fr = get(lambda v: v == "nvfp4_frozen:rl")
        _, k_ex, g_ex = get(lambda v: v == "nvfp4_fresh:base+lora")
        _, k_bb, g_bb = get(lambda v: v.startswith("nvfp4_fresh:base+lora="))
        _, k_bq, _ = get(lambda v: v == "nvfp4_fresh:base")
        valid = g_ctl is not None and 0.98 <= g_ctl <= 1.02
        n_valid += valid
        ok = (valid and None not in (g_n4, k_n4, g_fr) and g_n4 >= 0.9 and k_n4 - k_ctl <= 0.1
              and (part != "(b)" or g_fr <= 0.5))
        passes += ok
        fmt = lambda x: "  -  " if x is None else f"{x:.3f}"  # noqa: E731
        print(f"adapter {a} ({Path(j['adapter']).parent.name[:48]}): control k {fmt(k_ctl)} g {fmt(g_ctl)}"
              f"{'' if valid else ' INVALID'}")
        print(f"    NVFP4-stored adapter, bf16 base: k_net {fmt(None if k_n4 is None else k_n4 - k_ctl)}  g {fmt(g_n4)}")
        print(f"    on-grid NVFP4 merge:             k {fmt(k_fr)}  g {fmt(g_fr)}   (NVFP4 base alone: k {fmt(k_bq)})")
        print(f"    exact adapter, NVFP4 base:       k {fmt(k_ex)}  g {fmt(g_ex)}")
        print(f"    NVFP4 adapter, NVFP4 base:       k {fmt(k_bb)}  g {fmt(g_bb)}   -> {'passes' if ok else 'fails'}")
    if part == "(b)":
        print(f"{passes} of {n} adapters pass (b); MET needs >= 5 of 6")
    elif part == "(b4)":
        if n:
            met = n_valid >= 3 and passes == n_valid
            print(f"P0-ACCURACY (b4) {'MET' if met else 'NOT MET'} ({n_valid} of {n} controls valid, needs >= 3; "
                  f"the stored adapter passes on {passes} of the {n_valid} valid, needs all)")
    elif n:
        print(f"P0-ACCURACY {part} {'MET' if passes >= need else 'NOT MET'} ({passes} of {n} stored adapters pass; needs {need})")


def score_dose(d: Path, ks=(1, 4, 16, 64, 256)) -> dict:
    """P1-DOSE (registered 2026-10-11 06:40 UTC): adapter a1 with its update scaled by k. Per k: the on-grid NVFP4
    merge's logit gain g and weight-level gain g_W (census), and the NVFP4-stored adapter's g on the bf16 base.
    MET if, over the k with a valid control (g in [0.98, 1.02]): (a) merge g <= 0.05 at k = 1 and >= 0.5 at k = 256,
    (b) |g - g_W| <= 0.1 for the merge at every k, (c) the stored adapter keeps g >= 0.95 at every k."""
    print("\n== P1-DOSE: what the on-grid NVFP4 merge and NVFP4 storage keep as the update grows (adapter a1)")
    out = {}
    for k in ks:
        f, c = d / f"k{k}_logits.json", d / f"k{k}_census.jsonl"
        if not (f.exists() and c.exists()):
            print(f"k={k}: no data")
            continue
        j = json.loads(f.read_text())
        lg = j["logit_gain"]
        g = lambda pred: next((v["gain"] for kk, v in lg.items() if pred(kk)), None)   # noqa: E731
        g_ctl = g(lambda v: v.startswith("lora=") and v.rstrip("/").endswith("_copy"))
        g_st = g(lambda v: v.startswith("lora=") and v.rstrip("/").endswith("_nvfp4"))
        g_m = lg.get("nvfp4_frozen:rl", {}).get("gain")
        summ = [json.loads(x) for x in c.read_text().splitlines() if '"summary": true' in x]
        g_w = summ[0]["gain"] if summ else None
        moved = summ[0]["changed"] if summ else None
        valid = g_ctl is not None and 0.98 <= g_ctl <= 1.02
        out[k] = {"g_merge": g_m, "g_W": g_w, "moved": moved, "g_stored": g_st, "g_ctl": g_ctl, "valid": valid}
        fmt = lambda x: "  -  " if x is None else f"{x:.3f}"   # noqa: E731
        print(f"k={k:3d}: control g {fmt(g_ctl)}{'' if valid else ' INVALID'}  merge g {fmt(g_m)}  g_W {fmt(g_w)} "
              f"(moved {100 * (moved or 0):.3f}%)  stored g {fmt(g_st)}")
    ok = {k: v for k, v in out.items() if v["valid"] and None not in (v["g_merge"], v["g_W"], v["g_stored"])}
    if 1 in ok and 256 in ok and len(ok) == len(ks):
        a = ok[1]["g_merge"] <= 0.05 and ok[256]["g_merge"] >= 0.5
        b = all(abs(v["g_merge"] - v["g_W"]) <= 0.1 for v in ok.values())
        cc = all(v["g_stored"] >= 0.95 for v in ok.values())
        print(f"(a) {'holds' if a else 'fails'}  (b) {'holds' if b else 'fails'}  (c) {'holds' if cc else 'fails'}")
        print(f"P1-DOSE {'MET' if a and b and cc else 'NOT MET'}")
    else:
        print(f"P1-DOSE not judged yet ({len(ok)} of {len(ks)} scales with a valid control and data)")
    return out


def score_accuracy_behavior(d: Path, subject: str = "S1") -> None:
    """P0-ACCURACY (a) (addendum 4): R of the three arms on the NVFP4 base, against same-hardware references run alongside (A100-80G, addendum 4a).
    MET part (a): arm 2 R >= 0.8 and arm 3 R <= 0.5 (point estimates). Needs mlsys/analysis (agent_harness grounding).
    subject S2: the registered replication (a2) on EHRSQL (addendum 5), same rule."""
    import sys
    part = "(a)" if subject == "S1" else "(a2)"
    print(f"\n== P0-ACCURACY {part}, {subject} behavior on an NVFP4 base (same-hardware references)")
    sys.path.insert(0, "$ROUNDING/analysis")
    import read_retention as rr
    bench = {"S1": "medcalc_ver_s220", "S2": "ehrsql2024_s300"}[subject]
    path = lambda tag: d / tag / bench                                      # noqa: E731
    have = lambda tag: (path(tag) / "summary.json").exists()               # noqa: E731
    refs = ("ref_bf16base", "ref_bf16lora", "base_nvfp4")
    if not all(have(t) for t in refs):
        print("references incomplete: " + ", ".join(t for t in refs if not have(t)) + " missing")
        return
    base, ref, base_q = (rr.load(str(path(t)), bench, "verified") for t in refs)
    res = {}
    for tag, label in (("rl_nvfp4unmerged", "arm 1: exact adapter, unmerged"),
                       ("rl_nvfp4unmerged_l4", "arm 2: NVFP4-stored adapter, unmerged"),
                       ("rl_nvfp4frozen", "arm 3: on-grid NVFP4 merge")):
        if not have(tag):
            print(f"{label}: no data")
            continue
        r = rr.retention(base, ref, base_q, rr.load(str(path(tag)), bench, "verified"), n_boot=5000)
        res[tag] = r
        print(f"{label:40s} R={r['R']:.3f} [{r['R_ci95'][0]:.2f}, {r['R_ci95'][1]:.2f}]  acc base {r['acc']['base']:.3f} "
              f"ref {r['acc']['rl']:.3f} base_q {r['acc']['base_q']:.3f} arm {r['acc']['rl_q']:.3f}  n={r['n']}  "
              f"arm-vs-ref {r['rlq_vs_rl']}  arm-vs-base_q {r['rlq_vs_baseq']}")
    if "rl_nvfp4unmerged_l4" in res and "rl_nvfp4frozen" in res:
        met = res["rl_nvfp4unmerged_l4"]["R"] >= 0.8 and res["rl_nvfp4frozen"]["R"] <= 0.5
        print(f"P0-ACCURACY {part} {'MET' if met else 'NOT MET'} (arm 2 R >= 0.8 and arm 3 R <= 0.5)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
    a = ap.parse_args()
    d = Path(a.results)
    score_overhead(load(d / "p0_overhead.jsonl"))
    score_kernel(load(d / "p0_kernel.jsonl"))
    score_numerics(d / "p0_numerics")
    score_accuracy_logits(d / "p0_accuracy")
    if any((d / "p0_accuracy" / f"a{a}_logits.json").exists() for a in (7, 8)):
        score_accuracy_logits(d / "p0_accuracy", (7, 8), "(b2)")
    if any((d / "p0_accuracy" / f"p{a}_logits.json").exists() for a in range(1, 7)):
        score_accuracy_logits(d / "p0_accuracy", range(1, 7), "(b3)", prefix="p")
    if (d / "p0_dose").is_dir():
        score_dose(d / "p0_dose")
    if any((d / "p0_accuracy" / f"q{a}_logits.json").exists() for a in (1, 2, 4, 5)):
        score_accuracy_logits(d / "p0_accuracy", (1, 2, 4, 5), "(b4)", prefix="q")
    score_accuracy_behavior(d / "p0_accuracy" / "behavior" / "S1")
    if (d / "p0_accuracy" / "behavior" / "S2").exists():
        score_accuracy_behavior(d / "p0_accuracy" / "behavior" / "S2", "S2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
