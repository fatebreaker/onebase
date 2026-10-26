"""bench/official_nvfp4.py's build for NVIDIA's mixed-precision ModelOpt releases of multimodal Qwen3.5-architecture models
(Q27-RL: nvidia/Qwen3.8-27B-NVFP4, quant_algo MIXED_PRECISION: MLP gate/up/down and lm_head in NVFP4, the full-attention
q/k/v/o and the linear-attention projections in per-tensor FP8 (E4M3 codes x one weight_scale)). The release's tensors
are read under the text model's names (model.language_model.* -> model.*; vision and MTP tensors ignored) and each
module's format is read from its tensors (weight_scale_2: NVFP4; a float8 weight with one weight_scale: FP8). Output: a
bf16 checkpoint of the release dequantized, with --merge
  none      the release itself
  frozen    the adapter's update rounded onto the release's grid, its scales kept (nearest; an all-zero NVFP4 block or a
            zero tensor scale cannot move)
  srfrozen  as frozen with stochastic rounding on the NVFP4 layers (one fixed seed; FP8 layers nearest)
  gptq      error-compensated onto the same grid (mlsys gptq_merge.gptq_on_grid; H from --calib, as official_nvfp4_2.py)
  absmax    the merged weights re-quantized with recomputed scales (NVFP4: our absmax rule; FP8: amax / 448 per tensor)
Modules the release keeps in bf16 get a bf16 merge.
    python bench/official_mixed.py build --release <dir> --orig <text model> --out <dir> --merge M [--adapter A --calib C]"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_nvfp4_2 as on  # noqa: E402
from official_nvfp4_2 import BLK, FP8_MAX, quant  # noqa: E402


class MMShards(on.Shards):
    """The release's tensors under the text model's names."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.alias = {}
        for k in list(self.where):
            if k.startswith("model.language_model."):
                self.alias["model." + k[len("model.language_model."):]] = k
            elif not k.startswith(("model.visual.", "mtp.")):
                self.alias[k] = k

    def keys(self):
        return self.alias.keys()

    def get(self, k: str) -> torch.Tensor:
        return super().get(self.alias[k])


def fmt_of(rel: MMShards, mod: str) -> str | None:
    ks = rel.keys()
    if f"{mod}.weight_scale_2" in ks:
        return "nvfp4"
    if f"{mod}.weight_scale" in ks and f"{mod}.weight" in ks and rel.get(f"{mod}.weight").dtype == torch.float8_e4m3fn:
        return "fp8"
    return None


def build(a) -> int:
    from safetensors.torch import save_file
    rel_dir, orig_dir, out = Path(a.release), Path(a.orig), Path(a.out)
    rel, orig = MMShards(rel_dir), on.Shards(orig_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out.mkdir(parents=True, exist_ok=True)
    for f in orig_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    upd = on.adapter_updates(Path(a.adapter)) if a.merge != "none" else {}
    # GPTQ: Hessians a chunk of decoder layers at a time, in the order of the loop below (a 27B's do not fit at once)
    Hs = (on.ChunkedHessians(orig_dir, Path(a.adapter), a.calib, a.n_cal, dev,
                             order=[k[:-len(".weight")] for k in sorted(orig.keys()) if k[:-len(".weight")] in upd])
          if a.merge == "gptq" else {})
    gen = torch.Generator(device=dev).manual_seed(0)
    lv4 = torch.cat([-quant._E2M1_POS.flip(0)[:-1], quant._E2M1_POS])
    stats = {"nvfp4": 0, "fp8": 0, "merged": 0, "bf16_merged": 0, "num": 0.0, "den": 0.0, "moved": 0, "n": 0}
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
        if f == "nvfp4":
            codes, bs, s2 = on.release_layer(rel, mod, dev)
            S = (bs * s2).repeat_interleave(BLK, dim=1)
            w = codes * S
        elif f == "fp8":
            s = rel.get(f"{mod}.weight_scale").to(dev).float().reshape(())
            w = rel.get(f"{mod}.weight").to(dev).float() * s
        if f:
            stats[f] += 1
            if mod in upd:
                A, B, sc = upd[mod]
                dW = sc * (B.to(dev).float() @ A.to(dev).float())
                if a.merge == "absmax":
                    if f == "nvfp4":
                        new = quant.nvfp4_quant(w + dW)
                    else:
                        s_new = ((w + dW).abs().max() / FP8_MAX).clamp_min(1e-30)
                        new = on.e4m3((w + dW) / s_new) * s_new
                elif a.merge == "gptq":
                    from mlsys import gptq_merge
                    if f == "nvfp4":
                        Sb = bs * s2
                        q = gptq_merge.gptq_on_grid(w + dW, Hs[mod], Sb.clamp_min(1e-30), BLK, lv4)
                        new = torch.where(S > 0, q, w)
                    else:
                        Sb = torch.full((w.shape[0], -(-w.shape[1] // on.FBLK)), float(s), device=dev)
                        new = gptq_merge.gptq_on_grid(w + dW, Hs[mod], Sb.clamp_min(1e-30), on.FBLK, quant._e4m3_levels())
                elif f == "nvfp4":                                   # frozen / srfrozen on the NVFP4 grid
                    y = ((w + dW) / S.clamp_min(1e-30)).clamp(-quant.E2M1_MAX, quant.E2M1_MAX)
                    q = quant._e2m1(y, "stochastic", gen) if a.merge == "srfrozen" else quant._e2m1(y, "nearest", None)
                    new = torch.where(S > 0, q * S, w)
                else:                                                # frozen / srfrozen on the FP8 grid: nearest
                    new = on.e4m3((w + dW) / s.clamp_min(1e-30)) * s if float(s) > 0 else w
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
    res = {"out": str(out), "merge": a.merge, "adapter": a.adapter, "nvfp4_modules": stats["nvfp4"], "fp8_modules": stats["fp8"],
           "merged_modules": stats["merged"], "bf16_merged_modules": stats["bf16_merged"], "tensors": len(weight_map)}
    if stats["den"]:
        res |= {"g_W": stats["num"] / stats["den"], "moved_share": stats["moved"] / stats["n"]}
    print(json.dumps(res))
    if a.merge != "none" and stats["merged"] == 0:
        raise SystemExit("no adapter module matched a quantized module of the release")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    pb = sp.add_parser("build")
    pb.add_argument("--release", required=True)
    pb.add_argument("--orig", required=True)
    pb.add_argument("--out", required=True)
    pb.add_argument("--merge", default="none", choices=("none", "frozen", "srfrozen", "gptq", "absmax"))
    pb.add_argument("--adapter", default="")
    pb.add_argument("--calib", default="gsm8k")
    pb.add_argument("--n-cal", type=int, default=128)
    a = ap.parse_args()
    return build(a)


if __name__ == "__main__":
    raise SystemExit(main())
