"""Census: how much of a real RL update survives each rounding scheme, measured and predicted (docs/THEORY.md).

Extends agent_harness rl/async_pilot/merge_precision.py (same retained / changed / cos / out_ret metrics) from the bf16 merge
to every scheme in mlsys/merge.py, and puts the theory's prediction beside each measurement:

    ON-GRID nearest rounding   d = s * round(dW / s)            -> dead zone, gain < 1
    OFF-GRID, one shared grid  E[d] = dW, Var = s^2 p (1 - p)    -> gain ~ 1, noise from flips (p = frac(|dW| / s))
    (stochastic rounding obeys the same law; '_fresh' adds 'drift': the RL model's own scales rescale each group)

Metrics per (module, scheme), with d = rl_q - base_q the delivered update:
    gain      <d, dW> / |dW|^2           1 = unbiased in aggregate, < 1 = shrunk toward the base
    retained  |d| / |dW|                 (merge_precision's 'retained')
    cos       cosine(d, dW)
    noise     |d - dW| / |dW|
    changed   share of entries with d != 0
    out_gain, out_snr                    the same on layer outputs x d^T vs x dW^T for random unit-scale inputs
    r_p50, r_p90, below_half             |dW| / s quantiles and the share below half a step (the dead zone)
    pred_*                               the theory's value for gain / changed / noise from |dW| / s alone

    venv_vllm/bin/python -m mlsys.census --base <model dir> --adapter <run>/step25 [<run>/step50 ...] \
        --layers every:4 --out results/census/<tag>.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

from . import gptq, merge, quant

DEFAULT_SCHEMES = ("bf16_merge,bf16_sr_merge,fp16_merge,fp8_shared,fp8_fresh,fp8_frozen,int8_shared,int8_fresh,"
                   "int4g128_shared,int4g128_fresh,int4g128_frozen,nvfp4_fresh,mxfp4_fresh")
_STEP_FMT = {"bf16_merge": "bf16", "bf16_sr_merge": "bf16", "fp16_merge": "fp16"}


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class Base:
    """Lazy reader of a sharded safetensors checkpoint (model.safetensors.index.json)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        idx = self.path / "model.safetensors.index.json"
        self.map = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
        self._open: dict[str, object] = {}

    def names(self) -> list[str]:
        if self.map:
            return list(self.map)
        with safe_open(str(self.path / "model.safetensors"), framework="pt") as f:
            return list(f.keys())

    def get(self, name: str) -> torch.Tensor:
        shard = self.map[name] if self.map else "model.safetensors"
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.path / shard), framework="pt")
        return self._open[shard].get_tensor(name)


def adapter_modules(adapter: Path) -> tuple[float, list[tuple[int, str, str, str]]]:
    """(scale, [(layer, module, key_A, key_B)]) for every LoRA pair in a PEFT adapter."""
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"] if not cfg.get("use_rslora") else cfg["lora_alpha"] / math.sqrt(cfg["r"])
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    out = []
    for k in keys:
        m = re.match(r"base_model\.model\.(.*\.layers\.(\d+)\.(.+))\.lora_A\.weight$", k)
        if m:
            out.append((int(m.group(2)), m.group(3), k, k.replace(".lora_A.", ".lora_B."), m.group(1)))
    out.sort()
    return scale, [(lay, mod, ka, kb) for lay, mod, ka, kb, _ in out]


def select_layers(spec: str, layers: list[int]) -> set[int]:
    if spec == "all":
        return set(layers)
    if spec.startswith("every:"):
        k = int(spec.split(":")[1])
        return {x for x in layers if x % k == 0} | {max(layers)}
    return {int(x) for x in spec.split(",")}


def _step(scheme: str, W: torch.Tensor, base_q: torch.Tensor) -> torch.Tensor | None:
    """Grid spacing that the update is rounded with, where the format has a simple local step."""
    if scheme in _STEP_FMT:
        return quant.local_step(W, _STEP_FMT[scheme])
    fmt, _, mode = scheme.rpartition("_")
    if fmt in ("fp8", "int8", "int4g128") and (mode in ("shared", "frozen", "srfrozen") or mode.startswith("jitter")):
        return quant.local_step(W if mode not in ("frozen", "srfrozen") else base_q, fmt)
    if fmt.startswith("e8m") and mode == "frozen":                     # reduced-mantissa float grid, on-grid
        return quant.local_step(base_q, fmt)
    return None


def predict(dW: torch.Tensor, s: torch.Tensor, on_grid_nearest: bool) -> dict:
    r = dW.abs() / s
    d2 = dW.pow(2).sum()
    if on_grid_nearest:
        d = s * torch.round(dW / s)
        return {"pred_gain": float((d * dW).sum() / d2), "pred_changed": float((d != 0).float().mean()),
                "pred_noise": float((d - dW).norm() / dW.norm())}
    p = r - torch.floor(r)
    var = s.pow(2) * p * (1 - p)
    return {"pred_gain": 1.0, "pred_changed": float(torch.where(r >= 1, torch.ones_like(r), r).mean()),
            "pred_noise": float((var.sum() / d2).sqrt())}


def measure(W: torch.Tensor, dW: torch.Tensor, scheme: str, x: torch.Tensor, seed: int) -> dict:
    if scheme == "int4g128_gptq":                         # calibrated on the activations x (only with --acts)
        H = gptq.hessian(x)
        base_q, rl_q = gptq.gptq(W, H)[0], gptq.gptq(W + dW, H)[0]
    else:
        base_q, rl_q = merge.apply_scheme(W, dW, scheme, seed=seed)
    d = rl_q - base_q
    d2 = dW.pow(2).sum()
    yt, yd = x @ dW.T, x @ d.T
    row = {"scheme": scheme,
           "gain": float((d * dW).sum() / d2),
           "retained": float(d.norm() / dW.norm()),
           "cos": float(torch.nn.functional.cosine_similarity(d.flatten(), dW.flatten(), dim=0)),
           "noise": float((d - dW).norm() / dW.norm()),
           "changed": float((d != 0).float().mean()),
           "out_gain": float((yd * yt).sum() / yt.pow(2).sum()),
           "out_snr": float(yt.pow(2).sum() / (yd - yt).pow(2).sum().clamp_min(1e-30))}
    fmt, _, mode = scheme.rpartition("_")
    if mode == "fresh" and fmt in ("fp8", "int8", "int4g128"):
        b2, r2 = merge.apply_scheme(W, dW, f"{fmt}_shared", seed=seed)
        row["drift"] = float((d - (r2 - b2)).norm() / dW.norm())       # extra noise of independent grids
    s = _step(scheme, W, base_q)
    if s is not None:
        r = (dW.abs() / s).flatten()
        idx = torch.randint(0, r.numel(), (min(r.numel(), 1_000_000),),
                            generator=torch.Generator(device=r.device).manual_seed(0), device=r.device)
        q = torch.quantile(r[idx], torch.tensor([0.1, 0.5, 0.9], device=r.device))
        row.update({"r_p10": float(q[0]), "r_p50": float(q[1]), "r_p90": float(q[2]),
                    "below_half": float((r < 0.5).float().mean())})
        on_grid_nearest = merge.is_on_grid(scheme) and scheme != "bf16_sr_merge" and not scheme.endswith("_srfrozen")
        row.update(predict(dW, s, on_grid_nearest))
    # per input channel (lowbit T2-CHANNELS): energy sig2_j = mean_t x_tj^2 and the fraction of its update that channel j
    # keeps, f_j = sum_i d_ij dW_ij / sum_i dW_ij^2. With p_j = m_j / sum m (m_j = sum_i dW_ij^2), the diagonal output
    # gain sum sig2 d dW / sum sig2 dW^2 minus the weight gain equals Cov_p(sig2, f) / E_p[sig2] (Prop. 2(a), (c)).
    sig2 = x.float().pow(2).mean(0)
    mj = dW.float().pow(2).sum(0)
    dj = (d.float() * dW.float()).sum(0)
    ok = mj > 0
    if int(ok.sum()) > 2:
        f = dj[ok] / mj[ok]
        rank = lambda v: torch.argsort(torch.argsort(v)).float()  # noqa: E731
        rf, rs = rank(f), rank(sig2[ok])
        rf, rs = rf - rf.mean(), rs - rs.mean()
        row["chan_spearman"] = float((rf * rs).sum() / (rf.norm() * rs.norm()).clamp_min(1e-30))
    row["_sd"] = float((sig2 * dj).sum())
    row["_s2"] = float((sig2 * mj).sum())
    # sums for exact aggregation across modules
    row["_n"] = d.numel()
    row["_d2"] = float(d2)
    row["_dd"] = float((d * dW).sum())
    row["_e2"] = float(d.pow(2).sum())
    row["_n2"] = float((d - dW).pow(2).sum())
    row["_chg"] = float((d != 0).sum())
    row["_yt2"] = float(yt.pow(2).sum())
    row["_ytd"] = float((yd * yt).sum())
    row["_yn2"] = float((yd - yt).pow(2).sum())
    return row


def aggregate(rows: list[dict]) -> dict:
    t = {k: sum(r[k] for r in rows) for k in ("_n", "_d2", "_dd", "_e2", "_n2", "_chg", "_yt2", "_ytd", "_yn2")}
    out = {"modules": len(rows),
           "gain": t["_dd"] / t["_d2"], "retained": (t["_e2"] / t["_d2"]) ** 0.5,
           "cos": t["_dd"] / (t["_e2"] * t["_d2"]) ** 0.5 if t["_e2"] else 0.0,
           "noise": (t["_n2"] / t["_d2"]) ** 0.5, "changed": t["_chg"] / t["_n"],
           "out_gain": t["_ytd"] / t["_yt2"], "out_snr": t["_yt2"] / max(t["_yn2"], 1e-30)}
    if all("_sd" in r for r in rows):              # T2-CHANNELS: the diagonal output gain and the per-channel ordering
        out["out_gain_diag"] = sum(r["_sd"] for r in rows) / sum(r["_s2"] for r in rows)
        sp = sorted(r["chan_spearman"] for r in rows if "chan_spearman" in r)
        if sp:
            out["chan_spearman_median"] = sp[len(sp) // 2] if len(sp) % 2 else (sp[len(sp) // 2 - 1] + sp[len(sp) // 2]) / 2
            out["chan_spearman_pos"] = sum(v > 0 for v in sp) / len(sp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base model dir (sharded safetensors)")
    ap.add_argument("--adapter", nargs="+", default=[], help="PEFT adapter dir(s)")
    ap.add_argument("--full-other", nargs="+", default=[],
                    help="full-parameter fine-tunes of --base (dW = W_other - W_base per projection), instead of adapters")
    ap.add_argument("--schemes", default=DEFAULT_SCHEMES)
    ap.add_argument("--layers", default="every:4", help="all | every:K | comma list")
    ap.add_argument("--modules", default="", help="regex on module names (default: every LoRA module)")
    ap.add_argument("--x-rows", type=int, default=64, help="random input rows for the output metrics")
    ap.add_argument("--acts", default="", help="mlsys.acts file: use real input activations for the output metrics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="JSONL: one row per (adapter, module, scheme) + summaries")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    acts = torch.load(args.acts)["acts"] if args.acts else {}
    base = Base(args.base)
    schemes = [s for s in args.schemes.split(",") if s]
    if args.acts and "int4g128_gptq" not in schemes:
        schemes.append("int4g128_gptq")                      # GPTQ needs real calibration inputs
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w") as fo:
        jobs = [("adapter", Path(a)) for a in args.adapter] + [("full", Path(a)) for a in args.full_other]
        for kind, ad in jobs:
            if kind == "full":
                other = Base(ad)
                names = other.names()
                mods = sorted((int(m.group(1)), m.group(2), n, n) for n in names
                              for m in [re.match(r"model\.layers\.(\d+)\.((?:self_attn\.[qkvo]_proj)|(?:mlp\.(?:gate|up|down)_proj))\.weight$", n)]
                              if m)
                scale = 1.0
            else:
                scale, mods = adapter_modules(ad)
            keep = select_layers(args.layers, sorted({m[0] for m in mods}))
            rows: dict[str, list[dict]] = {s: [] for s in schemes}
            fa_path = ad / "adapter_model.safetensors"
            with (safe_open(str(fa_path), framework="pt") if kind == "adapter" else _Null()) as fa:
                for layer, mod, ka, kb in mods:
                    if layer not in keep or (args.modules and not re.search(args.modules, mod)):
                        continue
                    if kind == "full":
                        name = ka
                        W = base.get(name).to(args.device).float()
                        dW = other.get(name).to(args.device).float() - W
                    else:
                        name = ka.removeprefix("base_model.model.").removesuffix(".lora_A.weight") + ".weight"
                        W = base.get(name).to(args.device).float()
                        dW = merge.lora_delta(fa.get_tensor(ka).to(args.device), fa.get_tensor(kb).to(args.device),
                                              scale)
                    akey = name.removesuffix(".weight")
                    if akey in acts:                                  # real activations (mlsys.acts)
                        x = acts[akey].to(args.device)
                    else:
                        if acts:
                            continue                                      # --acts given: only modules it covers
                        g = torch.Generator(device=args.device).manual_seed(args.seed + layer)
                        x = torch.randn(args.x_rows, W.shape[1], generator=g, device=args.device)
                    for s in schemes:
                        r = measure(W, dW, s, x, args.seed)
                        r.update({"adapter": str(ad), "layer": layer, "module": mod,
                                  "dW_rms": float(dW.pow(2).mean().sqrt()), "W_rms": float(W.pow(2).mean().sqrt())})
                        rows[s].append(r)
                        fo.write(json.dumps(r) + "\n")
            for s in schemes:
                if not rows[s]:
                    continue
                summ = {"summary": True, "adapter": str(ad), "scheme": s, **aggregate(rows[s])}
                for kind, pat in (("attn", r"attn"), ("mlp", r"mlp")):
                    sub = [r for r in rows[s] if re.search(pat, r["module"])]
                    if sub:
                        summ[kind] = aggregate(sub)
                preds = [r for r in rows[s] if "pred_gain" in r]
                if preds:
                    w = [r["_d2"] for r in preds]
                    summ["pred_gain"] = sum(r["pred_gain"] * wi for r, wi in zip(preds, w)) / sum(w)
                    summ["pred_changed"] = sum(r["pred_changed"] * r["_n"] for r in preds) / sum(r["_n"] for r in preds)
                    summ["pred_noise"] = (sum(r["pred_noise"] ** 2 * wi for r, wi in zip(preds, w)) / sum(w)) ** 0.5
                    summ["below_half"] = sum(r["below_half"] * r["_n"] for r in preds) / sum(r["_n"] for r in preds)
                fo.write(json.dumps(summ) + "\n")
                print("SUMMARY " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                               for k, v in summ.items() if not isinstance(v, dict)}), flush=True)
    print(f"done in {time.time() - t0:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
