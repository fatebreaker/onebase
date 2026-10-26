"""Sparse weight sync: what each merge scheme sends between consecutive syncs of a real LoRA RL run (CPU or GPU).

For a sequence of adapters of one run (e.g. a G3 run's kept early policies pol1, pol2, pol3, pol5, pol10 and its
step25..step100 checkpoints) and a set of modules, every scheme merges each adapter into the bf16 base:
    bf16_merge        nearest (dead zone, the TRL / G3-S1 path)
    sr_fixed          stochastic rounding, one seed for every sync (G3-S2: a fixed dither pattern)
    sr_fresh          stochastic rounding, a new seed per sync
    fp8_fresh         FP8 per channel with the merged model's own scales (G3-S3)
and reports, per consecutive pair (t0 -> t1): payload = share of entries whose synced value changed (what a sparse
sync must send), and, per step, the delivered gain <D, dW>/|dW|^2 with D = synced - base_q.

    venv_vllm/bin/python -m mlsys.sync_payload --base <model dir> --adapters <run>/early/pol1 <run>/early/pol2 ... \
        --layers 3,15,31 --out results/census/sync_payload_<tag>.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import torch
from safetensors import safe_open

from . import merge, quant
from .census import Base, adapter_modules, select_layers

SCHEMES = ("bf16_merge", "sr_fixed", "sr_fresh", "fp8_fresh")


def synced(W: torch.Tensor, dW: torch.Tensor, scheme: str, step: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(base_q, synced weights) under a sync scheme; SR streams are per module (and per step for sr_fresh)."""
    if scheme == "bf16_merge":
        return merge.apply_scheme(W, dW, "bf16_merge")
    if scheme in ("sr_fixed", "sr_fresh"):
        seed = zlib.crc32(name.encode()) + (step * 7919 if scheme == "sr_fresh" else 0)
        return merge.apply_scheme(W, dW, "bf16_sr_merge", seed=seed)
    if scheme == "fp8_fresh":
        return merge.apply_scheme(W, dW, "fp8_fresh")
    raise ValueError(scheme)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="adapter dirs in training order")
    ap.add_argument("--steps", nargs="*", type=int, default=None, help="their steps (default: parsed from names)")
    ap.add_argument("--layers", default="3,15,31")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    ads = [Path(x) for x in a.adapters]
    steps = a.steps or [int("".join(ch for ch in p.name if ch.isdigit())) for p in ads]
    base = Base(a.base)
    scale, mods = adapter_modules(ads[0])
    keep = select_layers(a.layers, sorted({lay for lay, *_ in mods}))
    mods = [m for m in mods if m[0] in keep]
    fs = [safe_open(str(p / "adapter_model.safetensors"), framework="pt") for p in ads]
    tot: dict = {}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w") as fo:
        for lay, mod, ka, kb in mods:
            name = ka.removeprefix("base_model.model.").removesuffix(".lora_A.weight") + ".weight"   # as census.py
            W = base.get(name).to(a.device).float()
            prev = {s: None for s in SCHEMES}
            for i, (f, st) in enumerate(zip(fs, steps)):
                dW = merge.lora_delta(f.get_tensor(ka).to(a.device), f.get_tensor(kb).to(a.device), scale)
                for s in SCHEMES:
                    base_q, q = synced(W, dW, s, st, name)
                    D = (q - base_q).double()
                    d2 = float((dW.double() ** 2).sum())
                    row = {"layer": lay, "module": mod, "step": st, "scheme": s, "n": W.numel(),
                           "gain": float((D * dW.double()).sum()) / d2 if d2 else float("nan"),
                           "changed_vs_base": float((D != 0).float().mean())}
                    if prev[s] is not None:
                        row |= {"from_step": steps[i - 1], "payload": float((q != prev[s]).float().mean())}
                        k = (s, steps[i - 1], st)
                        acc = tot.setdefault(k, [0.0, 0])
                        acc[0] += row["payload"] * W.numel()
                        acc[1] += W.numel()
                    prev[s] = q
                    fo.write(json.dumps(row) + "\n")
            print(f"layer {lay} {mod} done ({time.time() - t0:.0f} s)", flush=True)
        for (s, s0, s1), (num, n) in sorted(tot.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            summ = {"summary": True, "scheme": s, "from_step": s0, "step": s1, "payload": num / n}
            fo.write(json.dumps(summ) + "\n")
            print("SUMMARY", json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
