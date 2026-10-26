"""Merging a LoRA update into an OFFICIALLY released block-FP8 checkpoint (review W1; descriptive weight-level census).

The release stores each linear weight as FP8 (E4M3) codes with one dequantization scale per 128x128 block
(``weight_scale_inv``: W = codes * s). For each LoRA module (every 4th layer, as the census) we compute what a merge
delivers of the update dW = scaling * B @ A against the released weights W:
  frozen   codes' = rne(clamp((W + dW) / s)): the release's scales kept (a fine-tune re-exported on the native grid)
  absmax   s' = amax_block(W + dW) / 448, codes'' = rne((W + dW) / s'): scales recomputed from the merged weights
  shift    a fixed 1.1x scale (moved off the native grid)
and report the delivered gain g_W = <D, dW> / <dW, dW> (D = Q(W + dW) - W), the share of entries that move, and the
update-to-step ratio rho = |dW| / step (median over entries), pooled per format. CPU only (torch fp8 casts).
    python bench/official_fp8_census.py --release <snapshot dir> --adapter <PEFT dir> --out <jsonl> [--every 4]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

FP8_MAX = 448.0
BLK = 128


def rne_e4m3(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def expand(s: torch.Tensor, shape) -> torch.Tensor:
    n, k = shape
    return s.repeat_interleave(BLK, 0)[:n].repeat_interleave(BLK, 1)[:, :k]


def block_amax(w: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    pn, pk = (-n) % BLK, (-k) % BLK
    wp = torch.nn.functional.pad(w.abs(), (0, pk, 0, pn))
    return wp.view((n + pn) // BLK, BLK, (k + pk) // BLK, BLK).amax(dim=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args()
    rel = Path(a.release)
    wmap = json.loads((rel / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((Path(a.adapter) / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    fa = safe_open(str(Path(a.adapter) / "adapter_model.safetensors"), framework="pt")
    pat = re.compile(r"layers\.(\d+)\.(.+)\.lora_A\.weight$")
    mods = sorted({(int(m.group(1)), m.group(2)) for k in fa.keys() if (m := pat.search(k))})
    mods = [(l, p) for l, p in mods if l % a.every == 0]
    handles: dict[str, object] = {}
    tot = {f: {"num": 0.0, "den": 0.0, "moved": 0, "n": 0} for f in ("frozen", "absmax", "shift")}
    rhos = []
    t0 = time.time()
    with open(a.out, "a") as fo:
        for layer, path in mods:
            key = f"model.language_model.layers.{layer}.{path}"
            if key + ".weight" not in wmap:
                continue
            fn = wmap[key + ".weight"]
            if fn not in handles:
                handles[fn] = safe_open(str(rel / fn), framework="pt")
            f = handles[fn]
            codes = f.get_tensor(key + ".weight").to(torch.float32)
            s = f.get_tensor(key + ".weight_scale_inv").to(torch.float32)
            W = codes * expand(s, codes.shape)
            akey = next(k for k in fa.keys() if k.endswith(f"layers.{layer}.{path}.lora_A.weight"))
            A = fa.get_tensor(akey).float()
            B = fa.get_tensor(akey.replace("lora_A", "lora_B")).float()
            dW = scaling * (B @ A)
            if dW.shape != W.shape:
                raise ValueError(f"{key}: update {tuple(dW.shape)} vs weight {tuple(W.shape)}")
            se = expand(s, W.shape)
            rhos.append(float((dW.abs() / se).median()))
            Wn = W + dW
            amax_s = block_amax(Wn) / FP8_MAX
            variants = {"frozen": se, "absmax": expand(amax_s, W.shape), "shift": se * 1.1}
            rec = {"layer": layer, "module": path, "shape": list(W.shape), "rho_median": rhos[-1]}
            for name, sc in variants.items():
                D = rne_e4m3(Wn / sc) * sc - W
                num, den = float((D * dW).sum()), float((dW * dW).sum())
                moved = int((D != 0).sum())
                rec[name] = {"g_W": num / den, "moved_share": moved / D.numel()}
                t = tot[name]
                t["num"] += num
                t["den"] += den
                t["moved"] += moved
                t["n"] += D.numel()
            fo.write(json.dumps(rec) + "\n")
            print(f"{key}: rho {rhos[-1]:.2e}  " + "  ".join(f"{k} g {rec[k]['g_W']:.3f}" for k in variants), flush=True)
        summary = {"summary": True, "release": str(rel), "adapter": a.adapter, "modules": len(rhos),
                   "rho_median_of_modules": sorted(rhos)[len(rhos) // 2] if rhos else None,
                   "seconds": round(time.time() - t0, 1),
                   **{k: {"g_W": v["num"] / v["den"] if v["den"] else None,
                          "moved_share": v["moved"] / v["n"] if v["n"] else None} for k, v in tot.items()}}
        fo.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
