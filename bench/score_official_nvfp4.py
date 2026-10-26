"""W1-NVFP4-OFFICIAL part 2 (docs/REGISTRY.md, registered 2026-10-12 20:22 UTC): the rule, and the appendix table.
For each adapter: the control's g (valid in [0.98, 1.02]); on NVIDIA's release, g of the frozen merge, the exact adapter
and the NVFP4-stored adapter, each against the same arm on our conversion (P0's results/p0_accuracy/<id>_logits.json:
nvfp4_frozen:rl, nvfp4_fresh:base+lora, nvfp4_fresh:base+lora=<nvfp4>); the re-export (absmax) merge, descriptive.
MET if every arm is within 0.05 of ours on every adapter with a valid control.
    python3 bench/score_official_nvfp4.py results/official_nvfp4/logits_<job> [--tex paper/figs/tab_official_nvfp4.tex]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = ("a5", "p1", "p2", "p3", "q1", "q2", "a7", "a8", "p4", "p5")
BASE = {i: "Llama-3.1-8B-Instruct" for i in ("a5", "p1", "p2", "p3", "q1", "q2")} | \
       {i: "Qwen3-8B" for i in ("a7", "a8", "p4", "p5")}
KIND = {"a5": "our RL", "a7": "our SFT", "a8": "our SFT"}


def ours(i: str) -> dict:
    j = json.loads((ROOT / "results/p0_accuracy" / f"{i}_logits.json").read_text())
    lg = j["logit_gain"]
    pick = lambda pred: next(v["gain"] for k, v in lg.items() if pred(k))  # noqa: E731
    return {"frozen": pick(lambda k: k == "nvfp4_frozen:rl"),
            "exact": pick(lambda k: k == "nvfp4_fresh:base+lora"),
            "stored": pick(lambda k: k.startswith("nvfp4_fresh:base+lora=")),
            "control": pick(lambda k: k.startswith("lora=") and k.rstrip("/").endswith("/copy"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tex", default="")
    a = ap.parse_args()
    d = Path(a.dir)
    rows, judged, ok_all = [], 0, True
    for i in IDS:
        if not (d / f"{i}_release.json").exists():
            continue
        rel = {k: v["g"] for k, v in json.loads((d / f"{i}_release.json").read_text()).items()}
        ctl = json.loads((d / f"{i}_control.json").read_text())["control"]["g"]
        own = ours(i)
        valid = 0.98 <= ctl <= 1.02
        diffs = {k: rel[k] - own[k] for k in ("frozen", "exact", "stored")}
        ok = all(abs(x) <= 0.05 for x in diffs.values())
        if valid:
            judged += 1
            ok_all &= ok
        rows.append({"id": i, "base": BASE[i], "control": ctl, "valid": valid, "release": rel, "ours": own,
                     "diff": diffs, "within_0.05": ok})
        print(f"{i:3s} {BASE[i]:22s} control {ctl:.3f}{'' if valid else ' (invalid)'}  "
              + "  ".join(f"{k} {rel[k]:+.3f} (ours {own[k]:+.3f})" for k in ("frozen", "exact", "stored"))
              + f"  re-export {rel['absmax']:+.3f}  {'ok' if ok else 'OUT'}")
    complete = len(rows) == len(IDS)
    verdict = ("MET" if ok_all else "NOT MET") if complete and judged else "INCOMPLETE"
    print(f"**W1-NVFP4-OFFICIAL part 2: {verdict}** ({judged} adapters with a valid control of {len(rows)})")
    (d / "score.json").write_text(json.dumps({"verdict": verdict, "judged": judged, "rows": rows}, indent=1))
    if a.tex:
        def f(x: float) -> str:
            x = 0.0 if abs(x) < 0.0005 else x
            return f"${x:.3f}$"
        pair = lambda r, k: f"{f(r['release'][k])} ({f(r['ours'][k])})"  # noqa: E731
        out = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
               r"base & adapter & control & merge, release's scales & exact adapter & NVFP4-stored adapter & re-export merge \\",
               r"\midrule"]
        for r in rows:
            dag = "" if r["valid"] else r"$^\dagger$"
            out.append(f"{r['base']} & {r['id']}, {KIND.get(r['id'], 'public')} & {f(r['control'])}{dag} & "
                       f"{pair(r, 'frozen')} & {pair(r, 'exact')} & {pair(r, 'stored')} & {f(r['release']['absmax'])} \\\\")
        out += [r"\bottomrule", r"\end{tabular}"]
        Path(a.tex).write_text("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
