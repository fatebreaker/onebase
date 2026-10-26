"""MISMATCH-DECOMP-2 (docs/REGISTRY.md, registered 2026-10-19 00:12 UTC before any data; amendment -CACHE): per adapter (S0G s0,
S8G s0, S8TL s0) on the H100, the per-token KL(engine || trainer) at the engine's own samples (bench/mismatch_bench2.py rows
decomp2_<adapter>_<engine>_<trainer>_<GPU>_<job> in results/mismatch_bench2.jsonl):
  (i)   the W8A8 engine's KL to the W8A16 trainer is at least 4x bf16's (the bf16 engine's KL to the bf16 trainer)
  (ii)  the fake-quant control's excess over bf16's is at least half the W8A8 engine's excess
  (iii) the matched engine's KL is at most 1.5x bf16's
Each held for all three adapters: met; for none: not met; else partly met, with the adapters listed. Descriptive: Marlin's KL,
the bf16 trainer on the W8A8 engine, the |dlog p| and TV analogues, the bootstrap intervals.
    python3 bench/score_decomp2.py [--json results/std_bench/decomp2_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
ADAPTERS = ("qwen9_dapo_S0G_s0", "qwen9_dapo_S8G_s0", "qwen9_dapo_S8TL_s0")
PAIRS = {"bf16": ("bf16", "bf16"), "w8a8": ("w8a8", "w16fused"), "w8a8_bf16trainer": ("w8a8", "bf16"),
         "fq": ("w8a8fq", "w16fused"), "matched": ("matched", "w16fused"), "marlin": ("marlin", "w16fused")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    f = LB / "results/mismatch_bench2.jsonl"
    rows = {}
    for line in (f.read_text().splitlines() if f.exists() else []):
        r = json.loads(line)
        lab = r.get("label", "")
        if not lab.startswith("decomp2_") or "_NVIDIA_H100" not in lab:
            continue
        for ad in ADAPTERS:
            if lab.startswith(f"decomp2_{ad}_"):
                eng, trn = lab[len(f"decomp2_{ad}_"):].split("_NVIDIA")[0].rsplit("_", 1)
                rows[(ad, eng, trn)] = r                  # the newest row per pair
    per = {}
    for ad in ADAPTERS:
        got = {k: rows.get((ad, *pair)) for k, pair in PAIRS.items()}
        per[ad] = {k: ({"kl": v["kl"], "kl_ci95": v["kl_ci95"], "mismatch": v["mismatch"], "tv": v.get("tv")} if v else None)
                   for k, v in got.items()}
    v = {"adapters": per}
    need = ("bf16", "w8a8", "fq", "matched")
    if all(all(per[ad][k] for k in need) for ad in ADAPTERS):
        def tally(test):
            ok = [ad for ad in ADAPTERS if test(per[ad])]
            return {"held_for": ok, "verdict": "MET" if len(ok) == 3 else "NOT MET" if not ok else "PARTLY MET"}
        kl = lambda p, k: p[k]["kl"]  # noqa: E731
        v["i"] = tally(lambda p: kl(p, "w8a8") >= 4 * kl(p, "bf16"))
        v["ii"] = tally(lambda p: kl(p, "fq") - kl(p, "bf16") >= 0.5 * (kl(p, "w8a8") - kl(p, "bf16")))
        v["iii"] = tally(lambda p: kl(p, "matched") <= 1.5 * kl(p, "bf16"))
        v["ratios"] = {ad: {k: kl(per[ad], k) / kl(per[ad], "bf16") for k in PAIRS if per[ad][k]} for ad in ADAPTERS}
        v["fq_share_of_excess"] = {ad: (kl(per[ad], "fq") - kl(per[ad], "bf16")) / (kl(per[ad], "w8a8") - kl(per[ad], "bf16"))
                                   for ad in ADAPTERS}
    else:
        v["status"] = "incomplete"
    print(json.dumps(v, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
