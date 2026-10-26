"""Weight-level delivered gain of low-precision adapter storage (CPU, a few minutes): for each LoRA module, round A and
B on their own grids (lowbit_lora.adapters.round_matrix) and measure what the stored product B^A^ keeps of the
update BA,
    g_BA = sum <B^A^, BA> / sum <BA, BA>      (traces of r x r products; BA itself is never formed),
with g_A = <A^, A>/<A, A> and g_B likewise, and the relative RMS error of the product. Proposition 1(c) says the OCP
MXFP4 rule shrinks the update (g < 1: clipped block maxima) while nearest rounding with a non-clipping scale does not
(g ~ 1 at a similar error), the weight-level counterpart of the logit gains of tab_numerics.
    python bench/adapter_gain.py --adapters S1=<dir>,S2=<dir> [--out results/adapter_gain.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowbit_lora.adapters import round_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters", required=True, help="NAME=DIR,NAME=DIR (PEFT adapter directories)")
    ap.add_argument("--fmts", default="fp8,nvfp4,mxfp4,mxfp4c")
    ap.add_argument("--out", default="results/adapter_gain.jsonl")
    a = ap.parse_args()
    from safetensors.torch import load_file
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for spec in a.adapters.split(","):
        name, _, d = spec.partition("=")
        sd = load_file(str(Path(d) / "adapter_model.safetensors"))
        pairs = sorted({k.replace(".lora_A.", ".lora_X.").replace(".lora_B.", ".lora_X.")
                        for k in sd if ".lora_A." in k or ".lora_B." in k})
        for fmt in a.fmts.split(","):
            t0 = time.time()
            acc = dict(ba_dot=0.0, ba_sq=0.0, ba_err=0.0, a_dot=0.0, a_sq=0.0, b_dot=0.0, b_sq=0.0)
            for kx in pairs:
                A = sd[kx.replace(".lora_X.", ".lora_A.")].float()          # [r, K]
                B = sd[kx.replace(".lora_X.", ".lora_B.")].float()          # [N, r]
                Aq, Bq = round_matrix(A, fmt), round_matrix(B, fmt)
                G, Gq = A @ A.t(), A @ Aq.t()                                # r x r: A A', A A^'
                H, Hq = B.t() @ B, Bq.t() @ B                                # B'B, B^'B
                ba_sq = float((H * G.t()).sum())                             # <BA, BA> = tr(B'B A A'), tr(XY) = sum(X*Y')
                ba_dot = float((Hq * Gq.t()).sum())                          # <B^A^, BA> = tr(A^' B^' B A) = tr(B^'B A A^')
                qq = float(((Bq.t() @ Bq) * (Aq @ Aq.t()).t()).sum())       # <B^A^, B^A^>
                acc["ba_dot"] += ba_dot
                acc["ba_sq"] += ba_sq
                acc["ba_err"] += qq - 2 * ba_dot + ba_sq                     # ||B^A^ - BA||^2
                acc["a_dot"] += float((Aq * A).sum()); acc["a_sq"] += float((A * A).sum())
                acc["b_dot"] += float((Bq * B).sum()); acc["b_sq"] += float((B * B).sum())
            rec = {"adapter": name, "src": d, "fmt": fmt, "modules": len(pairs),
                   "g_BA": acc["ba_dot"] / acc["ba_sq"], "rel_err_BA": max(acc["ba_err"], 0.0) ** 0.5 / acc["ba_sq"] ** 0.5,
                   "g_A": acc["a_dot"] / acc["a_sq"], "g_B": acc["b_dot"] / acc["b_sq"],
                   "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            print(json.dumps(rec), flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
