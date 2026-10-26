"""TRAINER-BENCH (descriptive; docs/REGISTRY.md): the colocated-RL trainer's step on one GPU, as train/grpo_q3.py runs it --
Qwen3.5-9B (Hugging Face), LoRA rank 64 (alpha 128) on q/k/v/o/gate/up/down with fp32 adapter weights, gradient
checkpointing, bf16 autocast, the fp32 chunked token log-probabilities (train/token_lp.py), the GRPO loss, AdamW -- on
synthetic sequences (N sequences of P prompt + C completion tokens, micro-batches of M), for one base mode per process:
  bf16      the bf16 trainer (the S0 arms)
  fp8w      the weight-matched trainer as the RL runs used it (mlsys.matched_fp8 parts="w": FP8 codes dequantized per call)
  fp8       W8A8 forward on the engine's codes (torch._scaled_mm), bf16 backward on the dequantized weight (parts="both")
  fp8bwd    W8A8 forward and FP8 backward (lowbit_lora.fp8_train; --grad-fmt e4m3 or e5m2)
  fp8fused  fp8bwd for the projections without an adapter; the adapter projections through our fused serving kernels in the
            forward and the backward (lowbit_lora.train_lora: K1/K2, bf16 LoRA path as the engine's)
Whether Qwen3.5's gated-delta-rule layers run flash-linear-attention's kernels or Hugging Face's PyTorch fallback is set by
PYTHONPATH (lowbit-lora/pylib_fla) before transformers is imported; the script reports which. One warm-up step, then
--steps timed steps. With --parity (fp8bwd only) one micro-batch's LoRA gradients are compared against the same forward
with the bf16 backward (cosine and relative error over all LoRA parameters). One JSON row per run to --out.
    python bench/trainer_bench.py --model <dir> --mode fp8bwd --out results/trainer_bench.jsonl [--parity]"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "train"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=("bf16", "fp8w", "fp8", "fp8bwd", "fp8fused", "fp8hybrid", "fp8w16", "fp8w16fused"))
    ap.add_argument("--grad-fmt", default="e4m3", choices=("e4m3", "e5m2"))
    ap.add_argument("--n-seqs", type=int, default=16)
    ap.add_argument("--prompt", type=int, default=160)
    ap.add_argument("--completion", type=int, default=1400)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--no-ckpt", action="store_true", help="no gradient checkpointing (memory permitting)")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fla = importlib.util.find_spec("fla") is not None
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from token_lp import token_logprobs
    torch.manual_seed(0)
    dev = "cuda:0"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    info = {}
    if a.mode != "bf16":
        from mlsys.matched_fp8 import patch_model
        info = patch_model(model, parts="w" if a.mode in ("fp8w", "fp8w16", "fp8w16fused") else "both", use_scaled_mm=True)
        if a.mode in ("fp8bwd", "fp8fused", "fp8hybrid", "fp8w16", "fp8w16fused"):
            from lowbit_lora import fp8_train
            info.update(fp8_train.fp8_backward(model, fmt=a.grad_fmt, kind="w8a16" if a.mode.startswith("fp8w16") else "w8a8"))
    if not a.no_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    if a.mode in ("fp8fused", "fp8hybrid", "fp8w16fused"):
        from lowbit_lora import train_lora
        info.update(train_lora.install(model, kind={"fp8fused": "fused", "fp8hybrid": "hybrid", "fp8w16fused": "w16"}[a.mode]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=1e-6, betas=(0.9, 0.99), weight_decay=0.0)
    t_load = time.time() - t0
    L = a.prompt + a.completion
    g = torch.Generator().manual_seed(0)
    ids_all = torch.randint(100, 100000, (a.n_seqs, L), generator=g)
    adv = torch.randn(a.n_seqs, generator=g)
    n_tok = a.n_seqs * a.completion

    def micro_loss(b):
        ids = ids_all[b:b + a.micro].to(dev)
        att = torch.ones_like(ids)
        w = torch.zeros((ids.shape[0], L - 1), device=dev)
        w[:, a.prompt - 1:] = adv[b:b + a.micro].to(dev)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=att).logits
        lp = token_logprobs(logits, ids)
        return -(w * lp).sum() / n_tok

    def step():
        torch.cuda.synchronize()
        ts = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        for b in range(0, a.n_seqs, a.micro):
            micro_loss(b).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        torch.cuda.synchronize()
        return time.time() - ts

    torch.cuda.reset_peak_memory_stats()
    t_warm = step()
    ts = [step() for _ in range(a.steps)]
    row = {"label": a.label or a.mode, "mode": a.mode, "grad_fmt": a.grad_fmt if a.mode == "fp8bwd" else None, "fla": fla,
           "ckpt": not a.no_ckpt, "gpu": torch.cuda.get_device_name(0), "n_seqs": a.n_seqs, "prompt": a.prompt,
           "completion": a.completion, "micro": a.micro, "t_load": round(t_load, 1), "t_warm": round(t_warm, 2),
           "t_step": [round(t, 2) for t in ts], "tok_per_s": round(n_tok * len(ts) / sum(ts), 1),
           "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "patch": info}
    if a.mode in ("fp8bwd", "fp8fused", "fp8hybrid", "fp8w16", "fp8w16fused"):
        from lowbit_lora import fp8_train
        row["fp8_bwd_stats"] = fp8_train.stats()
    def grads():
        opt.zero_grad(set_to_none=True)
        micro_loss(0).backward()
        return torch.cat([p.grad.detach().flatten().float() for p in params])
    if a.parity and a.mode == "fp8bwd":            # the same forward, bf16 backward (matched_fp8's) as the reference
        from mlsys import matched_fp8 as mf
        g8 = grads()
        fwd_ref = mf._make_forward(True, "both")
        saved = {}
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and hasattr(m, "w_codes"):
                saved[name] = m.forward
                m.forward = fwd_ref.__get__(m, type(m))
        g16 = grads()
        for name, m in model.named_modules():
            if name in saved:
                m.forward = saved[name]
        row["parity"] = {"cos": float(torch.nn.functional.cosine_similarity(g8, g16, dim=0)),
                         "rel_err": float((g8 - g16).norm() / g16.norm()), "n": int(g8.numel())}
    if a.parity and a.mode in ("fp8fused", "fp8hybrid", "fp8w16fused"):          # reference: PEFT's LoRA layers (fp32 adapters) on the same FP8 base
        from peft.tuners.lora import Linear as LoraLinear
        gf = grads()
        fused = [m for m in model.modules() if isinstance(m, LoraLinear) and "forward" in m.__dict__]
        saved = {id(m): m.__dict__.pop("forward") for m in fused}
        gp = grads()
        for m in fused:
            m.forward = saved[id(m)]
        row["parity"] = {"ref": "peft_fp32_lora_on_fp8bwd", "cos": float(torch.nn.functional.cosine_similarity(gf, gp, dim=0)),
                         "rel_err": float((gf - gp).norm() / gp.norm()), "n": int(gf.numel()), "fused_layers": len(fused)}
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
