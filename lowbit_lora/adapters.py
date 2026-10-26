"""PEFT LoRA adapters for the Phase-0 measurements (docs/REGISTRY.md).

random:   an adapter with random weights on the seven projections of a dense Llama/Qwen3 model. For speed only: the
          cost of a LoRA path depends on the rank and on how many distinct adapters a batch touches, not on the values.
quantize: a copy of an adapter whose lora_A / lora_B are rounded to a low-precision format and stored dequantized in
          the adapter's own dtype, so a served bf16 adapter path sees exactly the rounded values (P0-NUMERICS).
          Formats come from the mlsys simulators (mlsys/quant.py):
            fp8    E4M3, one float32 scale per row (A: per rank vector; B: per output channel)
            nvfp4  E2M1 in blocks of 16 along each row (the reduction dim of x A^T and h B^T), E4M3 block scales,
                   one float32 tensor scale
            mxfp4  E2M1 in blocks of 32 along each row, power-of-two (E8M0) scales (OCP rule: floor; may clip)
            mxfp4c the same with the scale rounded up (no element saturates)

    python -m lowbit_lora.adapters random --model <dir> --rank 64 --out <dir>
    python -m lowbit_lora.adapters quantize --src <dir> --fmt nvfp4 --out <dir> [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def projection_shapes(cfg: dict) -> dict[str, tuple[int, int]]:
    """(out, in) of each target projection of a dense Llama/Qwen3 decoder layer."""
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    hd = cfg.get("head_dim") or h // cfg["num_attention_heads"]
    q, kv = cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd
    return {"q_proj": (q, h), "k_proj": (kv, h), "v_proj": (kv, h), "o_proj": (h, q),
            "gate_proj": (inter, h), "up_proj": (inter, h), "down_proj": (h, inter)}


def make_random(model_dir: str, out_dir: str, rank: int, seed: int = 0, b_std: float = 1e-3) -> Path:
    from safetensors.torch import save_file
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    shapes = projection_shapes(cfg)
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for layer in range(cfg["num_hidden_layers"]):
        for name, (n_out, n_in) in shapes.items():
            mod = "self_attn" if name in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            key = f"base_model.model.model.layers.{layer}.{mod}.{name}"
            sd[f"{key}.lora_A.weight"] = (torch.randn(rank, n_in, generator=g) / n_in ** 0.5).to(torch.bfloat16)
            sd[f"{key}.lora_B.weight"] = (torch.randn(n_out, rank, generator=g) * b_std).to(torch.bfloat16)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": str(model_dir), "r": rank,
        "lora_alpha": rank, "lora_dropout": 0.0, "bias": "none", "target_modules": list(TARGETS),
        "fan_in_fan_out": False, "use_rslora": False, "use_dora": False, "modules_to_save": None}, indent=1))
    return out


def round_matrix(w: torch.Tensor, fmt: str) -> torch.Tensor:
    """Round a 2-D weight [rows, cols] to fmt with blocks/scales along each row; float32 dequantized result."""
    from mlsys import quant
    x = w.float()
    if fmt == "fp8":
        return quant.fp8_quant(x)[0]
    if fmt == "nvfp4":
        return quant.nvfp4_quant(x)
    if fmt == "mxfp4":
        return quant.mxfp4_quant(x)
    if fmt == "mxfp4c":   # MXFP4 with the power-of-two scale rounded UP: no element saturates (registry addendum 3)
        g, pad = quant._grouped(x, 32)
        amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        scale = torch.exp2(torch.ceil(torch.log2(amax / quant.E2M1_MAX)))
        return quant._ungrouped(quant._e2m1(g / scale, "nearest", None) * scale, pad)
    raise ValueError(f"unknown format {fmt}")


def quantize_dir(src: str, dst: str, fmt: str, device: str = "cpu") -> dict:
    """Copy a PEFT adapter dir with every lora_A/lora_B rounded to fmt. Returns relative-error statistics."""
    from safetensors.torch import load_file, save_file
    s, d = Path(src), Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(s / "adapter_model.safetensors"))
    num = {"A": 0.0, "B": 0.0}
    den = {"A": 0.0, "B": 0.0}
    out = {}
    for k, w in sd.items():
        if ".lora_A." in k or ".lora_B." in k:
            q = round_matrix(w.to(device), fmt)
            part = "A" if ".lora_A." in k else "B"
            num[part] += float(((q - w.to(device).float()) ** 2).sum())
            den[part] += float((w.to(device).float() ** 2).sum())
            out[k] = q.to(w.dtype).cpu()
        else:
            out[k] = w
    save_file(out, str(d / "adapter_model.safetensors"))
    for f in s.iterdir():
        if f.name != "adapter_model.safetensors" and f.is_file():
            shutil.copy2(f, d / f.name)
    stats = {"fmt": fmt, "src": str(s), "rel_rms_err_A": (num["A"] / max(den["A"], 1e-30)) ** 0.5,
             "rel_rms_err_B": (num["B"] / max(den["B"], 1e-30)) ** 0.5}
    (d / "lowbit_quant.json").write_text(json.dumps(stats, indent=1))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("random")
    r.add_argument("--model", required=True)
    r.add_argument("--rank", type=int, required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--src", required=True)
    q.add_argument("--fmt", required=True, choices=("fp8", "nvfp4", "mxfp4", "mxfp4c"))
    q.add_argument("--out", required=True)
    q.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.cmd == "random":
        print(make_random(a.model, a.out, a.rank, a.seed))
    else:
        print(json.dumps(quantize_dir(a.src, a.out, a.fmt, a.device)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
