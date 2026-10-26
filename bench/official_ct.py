"""Official compressed-tensors releases (GRANITE-RL: IBM's granite-4.2-8b-{nvfp4,fp8,mxfp4}): read the release's weights as
bf16 values of its own grid and optionally merge an adapter onto that grid, writing a bf16 checkpoint any GPU serves (as
bench/official_nvfp4.py does for ModelOpt releases). Formats, per module (read from the tensors present):
  nvfp4   weight_packed (uint8, two E2M1 codes per byte), weight_scale (E4M3 per 16 entries), weight_global_scale (fp32):
          w = code x weight_scale / weight_global_scale (compressed-tensors' tensor_group convention)
  mxfp4   weight_packed, weight_scale (uint8 E8M0 exponent per 32 entries): w = code x 2^(e - 127)
  fp8     weight (E4M3) and weight_scale (fp32 per output channel): w = code x scale
--merge none | frozen (the update rounded onto the release's grid, its scales kept; nearest) | srfrozen (stochastic, one
fixed seed) | gptq (error-compensated onto the same grid; H from --calib as bench/official_nvfp4_2.py). Modules the
release keeps in bf16 (lm_head here) get a bf16 merge. `check` decodes a few modules and reports their relative error
against the bf16 original (a check of the decoding, run before any build is used).
    python bench/official_ct.py check --release <dir> --orig <bf16 dir>
    python bench/official_ct.py build --release <dir> --orig <bf16 dir> --out <dir> --merge M [--adapter A --calib C]"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_nvfp4_2 as on  # noqa: E402
from official_nvfp4_2 import quant  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
LV4 = torch.cat([-quant._E2M1_POS.flip(0)[:-1], quant._E2M1_POS])


def unpack(packed: torch.Tensor, low_first: bool = True) -> torch.Tensor:
    lo, hi = packed & 0x0F, packed >> 4
    pair = (lo, hi) if low_first else (hi, lo)
    nib = torch.stack(pair, dim=-1).reshape(packed.shape[0], -1).long()
    mag = E2M1.to(packed.device)[nib & 0x7]
    return torch.where((nib & 0x8) != 0, -mag, mag)


def fmt_of(sh: on.Shards, mod: str) -> str | None:
    ks = sh.keys()
    if f"{mod}.weight_packed" in ks:
        return "nvfp4" if f"{mod}.weight_global_scale" in ks else "mxfp4"
    if f"{mod}.weight_scale" in ks and f"{mod}.weight" in ks:
        return "fp8"
    return None


def grid(sh: on.Shards, mod: str, f: str, dev: str):
    """(codes, per-entry scale S, block size, levels) with w = codes x S on the release's grid."""
    if f == "fp8":
        codes = sh.get(f"{mod}.weight").to(dev).float()
        s = sh.get(f"{mod}.weight_scale").to(dev).float().reshape(-1, 1)
        return codes, s.expand_as(codes), on.FBLK, quant._e4m3_levels()
    codes = unpack(sh.get(f"{mod}.weight_packed").to(dev))
    if f == "nvfp4":
        bs = sh.get(f"{mod}.weight_scale").to(dev).float()
        gs = sh.get(f"{mod}.weight_global_scale").to(dev).float().reshape(())
        return codes, (bs / gs).repeat_interleave(16, dim=1), 16, LV4
    e = sh.get(f"{mod}.weight_scale").to(dev).to(torch.int32)
    return codes, torch.exp2((e - 127).float()).repeat_interleave(32, dim=1), 32, LV4


def round_grid(y: torch.Tensor, f: str, stochastic: bool, gen) -> torch.Tensor:
    """y in units of the grid's scale -> the grid's code values (nearest or stochastic)."""
    if f == "fp8":
        return on.e4m3(y)
    y = y.clamp(-quant.E2M1_MAX, quant.E2M1_MAX)
    return quant._e2m1(y, "stochastic", gen) if stochastic else quant._e2m1(y, "nearest", None)


def check(a) -> int:
    rel, orig = on.Shards(Path(a.release)), on.Shards(Path(a.orig))
    mods = sorted({k.rsplit(".", 1)[0] for k in rel.keys() if k.endswith((".weight_packed", ".weight_scale"))})
    pick = [m for m in mods if ".layers.0." in m or ".layers.20." in m][:8]
    out = {}
    for m in pick:
        f = fmt_of(rel, m)
        W = orig.get(f"{m}.weight").float()
        res = {}
        for low in ((True, False) if f != "fp8" else (True,)):
            if f != "fp8":
                codes = unpack(rel.get(f"{m}.weight_packed"), low)
                _, S, _, _ = grid(rel, m, f, "cpu")
                w = codes * S
            else:
                c, S, _, _ = grid(rel, m, f, "cpu")
                w = c * S
            res["low_first" if low else "high_first"] = float((w - W).norm() / W.norm())
        out[m] = {"format": f, **res}
        print(m, json.dumps(out[m]), flush=True)
    return 0


def build(a) -> int:
    from safetensors.torch import save_file
    rel_dir, orig_dir, out = Path(a.release), Path(a.orig), Path(a.out)
    rel, orig = on.Shards(rel_dir), on.Shards(orig_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out.mkdir(parents=True, exist_ok=True)
    for f in orig_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    upd = on.adapter_updates(Path(a.adapter)) if a.merge != "none" else {}
    # GPTQ: Hessians a chunk of decoder layers at a time, in the order of the loop below (fits a 48-GB GPU)
    Hs = (on.ChunkedHessians(orig_dir, Path(a.adapter), a.calib, a.n_cal, dev,
                             order=[k[:-len(".weight")] for k in sorted(orig.keys()) if k.endswith(".weight") and k[:-len(".weight")] in upd])
          if a.merge == "gptq" else {})
    gen = torch.Generator(device=dev).manual_seed(0)
    stats = {"nvfp4": 0, "mxfp4": 0, "fp8": 0, "merged": 0, "bf16_merged": 0, "num": 0.0, "den": 0.0, "moved": 0, "n": 0}
    shard, size, weight_map, part = {}, 0, {}, 0

    def flush():
        nonlocal shard, size, part
        if shard:
            part += 1
            name = f"model-{part:05d}.safetensors"
            save_file(shard, str(out / name), metadata={"format": "pt"})
            weight_map.update({k: name for k in shard})
            shard, size = {}, 0

    for k in sorted(orig.keys()):
        mod = k[: -len(".weight")] if k.endswith(".weight") else None
        f = fmt_of(rel, mod) if mod else None
        if f:
            stats[f] += 1
            codes, S, blk, lv = grid(rel, mod, f, dev)
            w = codes * S
            if mod in upd:
                A, B, sc = upd[mod]
                dW = sc * (B.to(dev).float() @ A.to(dev).float())
                if a.merge == "gptq":
                    from mlsys import gptq_merge
                    q = gptq_merge.gptq_on_grid(w + dW, Hs[mod], S[:, ::blk].clamp_min(1e-30), blk, lv)
                    new = torch.where(S > 0, q, w)
                else:
                    y = (w + dW) / S.clamp_min(1e-30)
                    new = torch.where(S > 0, round_grid(y, f, a.merge == "srfrozen", gen) * S, w)
                D = new - w
                stats["merged"] += 1
                stats["num"] += float((D.double() * dW.double()).sum())
                stats["den"] += float((dW.double() ** 2).sum())
                stats["moved"] += int((D != 0).sum())
                stats["n"] += D.numel()
                w = new
            t = w.to(torch.bfloat16).cpu()
        else:
            t = rel.get(k) if k in rel.keys() else orig.get(k)
            if t.dtype != torch.bfloat16 and t.is_floating_point():
                t = t.to(torch.bfloat16)
            if mod in upd:
                A, B, sc = upd[mod]
                t = (t.float() + sc * (B.float() @ A.float())).to(torch.bfloat16)
                stats["bf16_merged"] += 1
        shard[k] = t.contiguous()
        size += t.numel() * t.element_size()
        if size >= 4 << 30:
            flush()
    flush()
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    res = {"out": str(out), "merge": a.merge, "adapter": a.adapter, **{x: stats[x] for x in ("nvfp4", "mxfp4", "fp8", "merged", "bf16_merged")},
           "tensors": len(weight_map)}
    if stats["den"]:
        res |= {"g_W": stats["num"] / stats["den"], "moved_share": stats["moved"] / stats["n"]}
    print(json.dumps(res))
    if a.merge != "none" and stats["merged"] == 0:
        raise SystemExit("no adapter module matched a quantized module of the release")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    pc = sp.add_parser("check")
    pc.add_argument("--release", required=True)
    pc.add_argument("--orig", required=True)
    pb = sp.add_parser("build")
    pb.add_argument("--release", required=True)
    pb.add_argument("--orig", required=True)
    pb.add_argument("--out", required=True)
    pb.add_argument("--merge", default="none", choices=("none", "frozen", "srfrozen", "gptq"))
    pb.add_argument("--adapter", default="")
    pb.add_argument("--calib", default="gsm8k")
    pb.add_argument("--n-cal", type=int, default=128)
    a = ap.parse_args()
    return check(a) if a.cmd == "check" else build(a)


if __name__ == "__main__":
    raise SystemExit(main())
