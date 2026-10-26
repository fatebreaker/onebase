"""v2 (2026-10-24; mock review R2 v25, missing experiment 3): adds run own_exact, the trainer's own copy built from the engine's
exact codes (the mapped tensors cloned into its own memory), so that own_exact against shared isolates the one-copy mapping from
the 0.05% of codes the trainer's quantizer rounds differently (own against own_exact). v1's doc follows.
RL-SHARED diagnostic (not registered data; an engineering check): does the trainer's backward agree when it computes on the
engine's mapped FP8 base (one copy, lowbit_lora.shared_base.map_info) and on its own FP8 copy (two copies, mlsys.matched_fp8.
patch_model), as train/grpo_q14.py sets each up (CPU load, then the weight-matched forward, fp8_train.fp8_backward kind w8a16,
gradient checkpointing, PEFT LoRA rank 64 on the seven projections, train_lora.install kind w16)? On one fixed batch (token
sequences from the DAPO-Math prompts, teacher forced), the loss is the mean log-probability of the next tokens; lora_B is set to
small seeded values so every LoRA parameter gets a gradient. The own-copy model runs twice (the kernels' run-to-run noise floor),
the shared model once. Prints, per comparison, the loss difference, the overall relative L2 error and cosine of all LoRA
gradients, and the worst modules.
    PYTHONPATH=<lowbit>/pylib_fla:<lowbit>:<mlsys>:<mlsys>/pylib LOWBIT_FUSED=1 python bench/shared_grad_check.py --model <dir>"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def batch(tok, n: int, length: int) -> torch.Tensor:
    text = ("Let $x$ and $y$ be real numbers such that $x^2 + y^2 = 1$. Find the largest possible value of $3x + 4y$. "
            "We use the Cauchy-Schwarz inequality: $(3x+4y)^2 \\le (3^2+4^2)(x^2+y^2) = 25$, so $3x+4y \\le 5$, with "
            "equality at $x = 3/5$, $y = 4/5$. Therefore the answer is $\\boxed{5}$. ") * 64
    ids = tok(text, return_tensors="pt").input_ids[0]
    return torch.stack([ids[i * 7: i * 7 + length] for i in range(n)])


def build(model_dir: str, mode: str, info: dict | None, seed: int = 0):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": "cpu"})
    if mode in ("shared", "own_exact"):
        from lowbit_lora.shared_base import map_info
        stats = map_info(model, info, check_codes=True)
        if mode == "own_exact":          # the engine's exact codes, cloned: the trainer's own copy, the same values as one copy's
            n_cl = 0
            for m in model.modules():
                if hasattr(m, "w_codes"):
                    m.w_codes = m.w_codes.clone()
                    n_cl += 1
            stats = {**stats, "cloned_modules": n_cl}
    else:
        from mlsys.matched_fp8 import patch_model
        stats = patch_model(model, parts="w", use_scaled_mm=torch.cuda.get_device_capability(0) >= (8, 9))
    model.to("cuda:0")
    from lowbit_lora import fp8_train
    fb = fp8_train.fp8_backward(model, kind="w8a16")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=TARGETS)
    torch.manual_seed(seed)
    model = get_peft_model(model, cfg)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_((torch.randn(p.shape, generator=g) * 1e-3).to(p.dtype))
    from lowbit_lora import train_lora
    tl = train_lora.install(model, kind="w16")
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model, {"map": stats, "fp8_backward": fb, "train_lora": tl}


def grads(model, ids: torch.Tensor) -> tuple[float, dict]:
    model.zero_grad(set_to_none=True)
    x = ids.to("cuda:0")
    logits = model(input_ids=x).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, x[:, 1:, None]).squeeze(-1)
    loss = -lp.mean()
    loss.backward()
    out = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    return float(loss), out


def compare(a: dict, b: dict) -> dict:
    num = den = dot = nb = 0.0
    per = []
    for n in a:
        ga, gb = a[n].flatten(), b[n].flatten()
        d = float(((ga - gb) ** 2).sum())
        s = float((ga ** 2).sum())
        num, den, dot, nb = num + d, den + s, dot + float((ga * gb).sum()), nb + float((gb ** 2).sum())
        per.append((d / max(s, 1e-30)) ** 0.5)
    names = list(a)
    worst = sorted(zip(per, names), reverse=True)[:5]
    return {"rel_l2": (num / max(den, 1e-30)) ** 0.5, "cosine": dot / max((den * nb) ** 0.5, 1e-30),
            "median_module_rel_l2": sorted(per)[len(per) // 2], "worst": [(n, round(e, 4)) for e, n in worst]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--util", type=float, default=0.30)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM
    t0 = time.time()
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=64, max_loras=1, max_model_len=2048, gpu_memory_utilization=a.util,
              enforce_eager=True, quantization="fp8", worker_extension_cls="mlsys.ipc_share.SharedBaseWorker", seed=0)
    info = llm.collective_rpc("export_base_ipc")[0]
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = batch(tok, a.n, a.length)
    res: dict = {"model": Path(a.model).name, "n": a.n, "length": a.length, "gpu": torch.cuda.get_device_name(0)}
    g = {}
    for tag, mode in (("own1", "own"), ("own2", "own"), ("shared", "shared"), ("own_exact", "own_exact")):
        m, setup = build(a.model, mode, info if mode in ("shared", "own_exact") else None)
        loss, g[tag] = grads(m, ids)
        res[f"loss_{tag}"] = loss
        res[f"setup_{tag}"] = {k: (v if k != "map" else {kk: vv for kk, vv in v.items() if kk != "examples"}) for k, v in setup.items()}
        print(json.dumps({"run": tag, "loss": loss, "setup": res[f"setup_{tag}"]}, default=str), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()
    res["own1_vs_own2"] = compare(g["own1"], g["own2"])
    res["own1_vs_shared"] = compare(g["own1"], g["shared"])
    res["own_exact_vs_shared"] = compare(g["own_exact"], g["shared"])
    res["own1_vs_own_exact"] = compare(g["own1"], g["own_exact"])
    res["seconds"] = round(time.time() - t0)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("setup")}, indent=1, default=str), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
