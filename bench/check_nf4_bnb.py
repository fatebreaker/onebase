"""QLORA-MERGE emulation check: does mlsys' NF4 (quant.nf4_quant, merge scheme nf4_requant) reproduce bitsandbytes'
codes and block scales, for the base and for PEFT's merge into Linear4bit? Real Mistral-7B-Instruct projections (the
first 512 rows of q, v and down in layers 0 and 16) and the STD-BENCH-RL seed-0 GRPO adapter's update. CPU only.
bitsandbytes is not in the agent_harness venv; it was installed with --no-deps --target into a scratch directory (0.47.0):
    PYTHONPATH=<that dir> venv_vllm/bin/python bench/check_nf4_bnb.py > results/std_bench/nf4_bnb_check.jsonl"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, "$ROUNDING")
sys.path[:0] = [p for p in __import__("os").environ.get("PYTHONPATH", "").split(":") if p]
import bitsandbytes.functional as F  # noqa: E402
from mlsys import quant  # noqa: E402

M = Path("$SCRATCH/models/Mistral-7B-Instruct-v0.3")
AD = Path("$SCRATCH/std_bench/adapters/mistral_gsm8k_rl")
idx = json.loads((M / "model.safetensors.index.json").read_text())["weight_map"]
cfg = json.loads((AD / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]
fa = safe_open(str(AD / "adapter_model.safetensors"), framework="pt")
akeys = list(fa.keys())


def weight(name):
    with safe_open(str(M / idx[name]), framework="pt") as f:
        return f.get_tensor(name)


ROWS = 512                                                 # a row slice: NF4 blocks run along rows


def update(name):
    mod = name[: -len(".weight")]
    a = [k for k in akeys if k.endswith(mod + ".lora_A.weight")]
    b = [k for k in akeys if k.endswith(mod + ".lora_B.weight")]
    return (fa.get_tensor(b[0])[:ROWS].float() @ fa.get_tensor(a[0]).float()) * scale


LV = quant.NF4_LEVELS.float()


def bnb_codes(w):
    """bitsandbytes' 4-bit codes and block absmax of w (float32 input: the default backend's dequantization casts the
    levels to the input dtype first, unlike the CUDA kernels, so codes and scales are what we compare)."""
    q, st = F.quantize_4bit(w.float().contiguous(), blocksize=64, quant_type="nf4", compress_statistics=False)
    q = q.view(torch.uint8).reshape(-1)
    codes = torch.empty(q.numel() * 2, dtype=torch.long)
    codes[::2], codes[1::2] = (q >> 4).long(), (q & 0xF).long()
    return codes[: w.numel()], st.absmax.float()


def our_codes(v):
    """codes and absmax of mlsys' dequantized NF4 values v (each entry is level * block absmax in float32)."""
    g = v.float().reshape(-1, 64)
    amax = g.abs().amax(-1)
    return torch.argmin((g / amax[:, None]).reshape(-1, 1).sub(LV).abs(), dim=-1), amax


out = []
for layer in (0, 16):
    for p in ("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"):
        name = f"model.layers.{layer}.{p}.weight"
        W = weight(name)[:ROWS].contiguous()               # bf16, as the checkpoint stores it
        d = update(name)
        cb, ab = bnb_codes(W)
        co, ao = our_codes(quant.nf4_quant(W.float()))
        # PEFT's merge into Linear4bit on CUDA: dequantize (float32 level x absmax, cast to bf16), add the update in
        # float32 (LoRA weights in float32), quantize the sum afresh
        base_dq = (LV[cb].reshape(-1, 64) * ab[:, None]).reshape(W.shape).to(torch.bfloat16)
        cbm, abm = bnb_codes(base_dq.float() + d)
        com, aom = our_codes(quant.nf4_quant(quant.nf4_quant(W.float()) + d))
        merged = (LV[cbm].reshape(-1, 64) * abm[:, None]).reshape(W.shape)
        D = merged - base_dq.float()
        # QLoRA's double quantization (compress_statistics=True: block absmax itself quantized to 8 bits in groups of
        # 256): PEFT's merge on such a base, all in float32 (no emulation of ours to compare; descriptive)
        q2, st2 = F.quantize_4bit(W.float().contiguous(), blocksize=64, quant_type="nf4", compress_statistics=True)
        base2 = F.dequantize_4bit(q2, st2)
        q2m, st2m = F.quantize_4bit((base2 + d).contiguous(), blocksize=64, quant_type="nf4", compress_statistics=True)
        D2 = F.dequantize_4bit(q2m, st2m) - base2
        dq_gain = round(float((D2 * d).sum() / (d * d).sum()), 5)
        dq_codes_changed = float((q2m.view(torch.uint8) != q2.view(torch.uint8)).float().mean())
        dq_entries_changed = float((D2 != 0).float().mean())
        out.append({"module": name, "base_codes_equal": float((cb == co).float().mean()),
                    "base_absmax_equal": float((ab == ao).float().mean()),
                    "merge_codes_equal": float((cbm == com).float().mean()),
                    "merge_absmax_rel_diff_max": float(((abm - aom).abs() / aom).max()),
                    "codes_changed_by_merge_bnb": float((cbm != cb).float().mean()),
                    "merge_weight_gain_bnb": round(float((D * d).sum() / (d * d).sum()), 5),
                    "dq_merge_weight_gain": dq_gain, "dq_packed_bytes_changed": dq_codes_changed,
                    "dq_entries_changed": dq_entries_changed})
        print(json.dumps(out[-1]), flush=True)
