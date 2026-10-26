"""STORED-VS-RANK (docs/REGISTRY.md): the equal-bytes alternative to NVFP4 adapter storage. A PEFT LoRA adapter truncated to
rank r' by the SVD of each update B A (exact in fp32 through QR of the factors: B A = Q_B (R_B R_A^T) Q_A^T), keeping the r'
largest singular directions; r' = 18 stores a rank-64 adapter's bf16 bytes in NVFP4's (64 x 4.5 bits = 18 x 16 bits: 4-bit
codes and an 8-bit scale per 16 values). lora_alpha is scaled so that the update's scale (alpha / r, or alpha / sqrt(r) with
rsLoRA) is unchanged, so B' A' is the best rank-r' approximation of B A. Writes the adapter and svd_stats.json (the fraction of
each update's squared Frobenius norm kept).
    python bench/svd_adapter.py --src <adapter dir> --rank 18 --out <dir> [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    import torch
    from safetensors.torch import load_file, save_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "adapter_config.json").read_text())
    r, alpha, rs = cfg["r"], cfg["lora_alpha"], bool(cfg.get("use_rslora", False))
    assert not cfg.get("rank_pattern") and not cfg.get("alpha_pattern"), "per-module ranks are not handled"
    k = a.rank
    assert 1 <= k <= r
    sd = load_file(str(src / "adapter_model.safetensors"))
    out, kept = {}, []
    for ka in [x for x in sd if ".lora_A." in x]:
        kb = ka.replace(".lora_A.", ".lora_B.")
        A = sd[ka].to(a.device, torch.float32)                    # [r, K]
        B = sd[kb].to(a.device, torch.float32)                    # [N, r]
        qb, rb = torch.linalg.qr(B)                               # B = qb rb
        qa, ra = torch.linalg.qr(A.t())                           # A^T = qa ra
        u, sv, vh = torch.linalg.svd(rb @ ra.t())                 # B A = qb (rb ra^T) qa^T = (qb u) diag(sv) (qa vh^T)^T
        sq = sv[:k].sqrt()
        out[kb] = ((qb @ u[:, :k]) * sq[None, :]).to(sd[kb].dtype).cpu()          # [N, k]
        out[ka] = ((sq[:, None] * vh[:k]) @ qa.t()).to(sd[ka].dtype).cpu()       # [k, K]
        tot = float((sv ** 2).sum())
        kept.append(float((sv[:k] ** 2).sum()) / tot if tot > 0 else 1.0)
    for key, w in sd.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            out[key] = w
    save_file(out, str(dst / "adapter_model.safetensors"))
    for f in src.iterdir():
        if f.is_file() and f.name not in ("adapter_model.safetensors", "adapter_config.json"):
            shutil.copy2(f, dst / f.name)
    cfg["r"] = k
    cfg["lora_alpha"] = alpha * ((k / r) ** 0.5 if rs else k / r)
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    stats = {"src": str(src), "rank": k, "from_rank": r, "pairs": len(kept), "kept_frac_mean": sum(kept) / len(kept),
             "kept_frac_min": min(kept), "kept_frac_max": max(kept)}
    (dst / "svd_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
