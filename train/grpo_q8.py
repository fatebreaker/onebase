"""train/grpo_q8.py = train/grpo_q7.py with --offload-base extended to the weight-matched FP8 trainer (RL-27B-L40S-FIT: a base too
large for the GPU in bf16): with --offload-base the engine falls asleep while the trainer is built (so the trainer's base and the
engine's never share the GPU), the trainer's frozen parameters and its large buffers (the FP8 codes, w_codes) live in pinned
CPU memory and reach the GPU only for the trainer's pass. Off by default: then this is grpo_q7.py exactly.
grpo_q7.py's doc follows.
train/grpo_q7.py = train/grpo_q4.py plus --offload-base (the MLSys v3 mock review, R1 and R2: a one-GPU bf16 baseline with
verl-style colocation). Off by default: then this is grpo_q4.py exactly.
  --offload-base  with --sleep: the trainer's frozen base lives in pinned CPU memory; it is copied to the GPU after the
                  engine falls asleep, used by the trainer's pass (forward, backward, optimizer step), and dropped from the
                  GPU before the engine wakes. The adapter, its optimizer state and the model's buffers stay on the GPU.
                  So at any time the GPU holds either the engine (weights + KV cache) or the trainer (base + activations),
                  as colocated frameworks do with parameter offload and the engine's sleep mode. Logged per step: t_load,
                  t_offload (seconds), the trainer's peak allocated memory (GiB).
grpo_q4.py's doc follows.
train/grpo_q4.py = train/grpo_q3.py plus the RL-KERNELS trainer options (docs/PLAN_MLSYS_V2.md), all off by default (then
this is grpo_q3.py exactly):
  --fp8-train bwd    with --match fp8w: the trainer's base GEMMs on the engine's FP8 codes in the forward (W8A8 through
                     torch._scaled_mm, matched_fp8 parts="both") and the backward (lowbit_lora.fp8_train), instead of
                     parts="w"'s per-call dequantization to bf16
  --fp8-train fused  as bwd, and the adapter projections through our fused serving kernels in the forward and the backward
                     (lowbit_lora.train_lora)
  --fp8-train w16    the weight-matched arithmetic of parts="w" (the engine's FP8 weights, bf16 activations: the lowest
                     trainer-engine mismatch, MISMATCH-BENCH) through our mixed-input GEMM and the fused adapter path
  --no-ckpt          no gradient checkpointing (where the memory allows it)
The gated-delta-rule layers use flash-linear-attention when lowbit-lora/pylib_fla is on PYTHONPATH (logged at start).
grpo_q3.py's doc follows.
train/grpo_q2.py plus --tis C (truncated importance sampling, as train/grpo_q_tis.py) and --cpu-load (RL-27B-FIT: load the
trainer's bf16 model on the CPU and, with --match fp8w, convert it to the engine's FP8 codes there before moving it to the
GPU, so the trainer never holds the bf16 weights on the GPU). Both default off: then this is grpo_q2.py exactly.
grpo_q2.py's doc follows.

grpo_q2.py = grpo_q.py plus --sleep (RL-FP8-QMATH-TUNED; grpo_q.py is left unchanged for the runs using it). LoRA GRPO for strong models (user, 10-15: Qwen and other strong models, not Llama): train/grpo_lora9.py plus
  --task dapo    DAPO-Math-17k (open-r1's deduplicated English set of BytedTsinghua-SIA/DAPO-Math-17k, Apache-2.0;
                 14,116 competition problems with integer answers), the question with std_bench's boxed-answer
                 instruction, reward by math-verify against the integer answer
  --task tools   ToolACE in BFCL's prompting format (as train/grpo_lora6.py; reward bench/toolcall.py)
  --no-think     Qwen3's non-thinking chat mode (chat_template_kwargs enable_thinking=False) for rollouts; the trainer
                 trains on the engine's own prompt token ids, so both see the same template
  --graphs       the engine captures CUDA graphs (eager otherwise)
  --max-model-len the engine's context (default 4096)
  --trainer-gpu K the trainer on visible GPU K while the engine keeps GPU 0 (models too large to share one GPU, e.g. the
                 27B: run with CUDA_VISIBLE_DEVICES=a,b and --vllm-util up to ~0.9)
Otherwise grpo_lora9.py's doc:

train/grpo_lora4.py (kept unchanged for the streams that run it) plus --task math_hard (MATH-RL-HARD): GRPO on the
MATH training split's Level 4 and 5 problems (EleutherAI/hendrycks_math, MIT; 3,994 problems; MATH-500 is drawn from the
test split), the reference the last boxed answer of the problem's solution, the reward std_bench.grade_math (math-verify,
as MATH-500 is scored). Otherwise grpo_lora4.py's doc:

train/grpo_lora3.py (kept unchanged for the streams that run it) plus the MLSys RL arms on popular tasks (RL-FP8-MATH):
--engine fp8 serves the rollouts from vLLM's online FP8 engine (W8A8, per-tensor weight scales; with LOWBIT_FUSED=1 in
the environment the exact adapter runs through our fused kernels, else through vLLM's LoRA path) and --match fp8w makes
the trainer compute on that engine's FP8 weight codes and scales with bf16 activations (mlsys matched_fp8.patch_model,
parts "w": the weight-matched trainer of the RL study). With --engine fp8 every step logs the trainer-engine mismatch.

train/grpo_lora2.py (kept unchanged for the streams that run it) plus --task math: GRPO on MATH-style problems, the
prompts MetaMathQA's MATH_AnsAug entries (rephrased-answer augmentations of the MATH training set, so disjoint from
MATH-500, which comes from MATH's test set; one per distinct question), the reference the text after "The answer is:",
the reward std_bench.grade_math (math-verify, as MATH-500 is scored). --task gsm8k (default) is grpo_lora2.py's.

train/grpo_lora.py (kept unchanged for the streams that run it) plus --base fp8 and --sync fp8merge (SYNC-FP8): the
engine's base in weights-only FP8 with per-row scales (scheme fp8_fresh; row-local, so the trainer's values equal the
engine's), and the policy synced as a nearest merge onto that frozen FP8 grid (scheme fp8_frozen, role rl). Otherwise
grpo_lora.py's doc:

LoRA GRPO on GSM8K for STD-BENCH-RL: a minimal on-policy GRPO loop whose rollouts use the exact adapter.

Each step samples P training questions, generates G completions per question with vLLM serving the base model and the
current LoRA unmerged (saved and loaded as a new LoRA request every step: the exact sync of our RL arm S0), rewards a
completion 1 if its final answer equals the reference (bench/std_bench.py's GSM8K grader), normalizes rewards within
each question's group (GRPO), and takes one optimizer step on the token-level policy-gradient loss
    L = - sum_{i,t} A_i log pi(c_{i,t} | x_i, c_{i,<t}) / sum_i |c_i|
(one update per batch, so the importance ratio is 1 and there is no clipping; no KL term). The policy trains in bf16
with fp32 LoRA weights and gradient checkpointing; vLLM shares the GPU (--vllm-util of its memory, eager mode).

    venv_vllm/bin/python train/grpo_lora.py --model <dir> --out <dir> [--steps 200 --prompts 16 --group 8 --lr 1e-5]
Writes <out>/log.jsonl (one line per step), <out>/step_<k>/ every --save-every steps, and the final adapter in <out>.
A restarted run resumes from the newest step_<k> (optimizer moments restart; recorded in grpo_meta.json).
--sync bf16merge (RL-SYNC-GSM8K) serves the policy as common LoRA RL pipelines do instead: every step the current
adapter is merged into the engine's pristine bf16 weights with nearest rounding (mlsys VariantWorker scheme bf16_merge,
role rl; needs <mlsys> on PYTHONPATH) and rollouts run on the merged weights; the delivered share of the update is
logged per step ("sync_gain").
--base nf4 (QLORA-GRPO) trains QLoRA-style: the engine serves the bitsandbytes-NF4 base (mlsys VariantWorker scheme
nf4_fresh, role base: blocks of 64, absmax in float32) with the exact adapter, and the trainer computes on the same
NF4 values (mlsys merge.apply_scheme on each projection weight, cast to bf16 as the engine loads them), so the adapter
learns on the quantized base. With --base nf4 each step also logs the mean |trainer - engine| log-probability of the
sampled completion tokens ("mismatch"), a check that the two bases agree.
--base nvfp4 (SYNC-NVFP4) does the same on the weights-only NVFP4 base (scheme nvfp4_fresh), and --sync nvfp4merge (with
--base nvfp4) syncs the policy as a merge onto that checkpoint's frozen NVFP4 grid at every step (scheme nvfp4_frozen,
role rl, from the pristine weights: Q(Q(W) + dW) on Q(W)'s block scales), logging the delivered share ("sync_gain")
and the trainer-engine mismatch, which then includes what the merge failed to deliver. --packed-scales (with --base
nvfp4; SYNC-NVFP4-M) gives the trainer the engine's own NVFP4 values (tensor scales per 4,096-row chunk of the packed
q/k/v and gate/up weights, as the engine's variant worker computes them); without it each projection has its own.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
DL = Path("$SCRATCH/data/std_bench_dl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--vllm-util", type=float, default=0.35)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="gsm8k", choices=("gsm8k", "math", "math_hard", "dapo", "tools"))
    ap.add_argument("--no-think", action="store_true", help="Qwen3 non-thinking chat mode")
    ap.add_argument("--graphs", action="store_true", help="the engine captures CUDA graphs (default: eager)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--trainer-gpu", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="engine batch cap; with CUDA graphs on hybrid (GDN) models it must fit the state-cache blocks")
    ap.add_argument("--engine", default="bf16", choices=("bf16", "fp8"))
    ap.add_argument("--match", default="none", choices=("none", "fp8w"))
    ap.add_argument("--tis", type=float, default=0.0, help="truncated importance sampling cap C (0: off)")
    ap.add_argument("--cpu-load", action="store_true", help="load (and FP8-match) the trainer's model on the CPU first")
    ap.add_argument("--fp8-train", default="none", choices=("none", "bwd", "fused", "w16"))
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--sync", default="exact", choices=("exact", "bf16merge", "nvfp4merge", "fp8merge"))
    ap.add_argument("--base", default="bf16", choices=("bf16", "nf4", "nvfp4", "fp8"))
    ap.add_argument("--offload-base", action="store_true",
                    help="with --sleep: the trainer's frozen base in pinned CPU memory except during the trainer's pass")
    ap.add_argument("--sleep", action="store_true",
                    help="RL-FP8-QMATH-TUNED: put the engine to sleep while the trainer runs (vLLM sleep mode, level 1: weights "
                         "offloaded to the CPU, KV cache discarded) and wake it before each generation, as colocated frameworks "
                         "do (verl, OpenRLHF), so a bf16 engine can take the largest memory share the trainer leaves")
    ap.add_argument("--packed-scales", action="store_true",
                    help="--base nvfp4: compute the trainer's NVFP4 values as the engine does (SYNC-NVFP4-M)")
    a = ap.parse_args()
    lp_engine = True                                            # log the trainer-engine mismatch in every arm
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import pandas as pd
    import torch
    from token_lp import keep_old_log, token_logprobs
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import std_bench

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.task == "gsm8k":
        df = pd.read_parquet(DL / "gsm8k_train.parquet")
        data = [{"q": r.question + std_bench.SUFFIX, "a": r.answer.split("####")[-1].strip().replace(",", "")}
                for r in df.itertuples()]
    elif a.task == "dapo":                                       # DAPO-Math-17k (deduplicated, English)
        df = pd.read_parquet(DL.parent / "dapo_dl/en/train.parquet")
        data = [{"q": r.prompt + std_bench.SUFFIX, "a": str(r.solution)} for r in df.itertuples()]
    elif a.task == "tools":                                      # ToolACE in BFCL's prompting format
        import toolcall
        data = [{"msgs": r["messages"], "gold": r["gold"]} for r in
                map(json.loads, (DL.parent / "toolace_dl/toolace_rl.jsonl").read_text().splitlines())
                if sum(len(m["content"]) for m in r["messages"]) <= 8000]   # drops 77 of 4,850 (trainer memory)
    elif a.task == "math_hard":                                  # MATH train, Levels 4-5, reference = last boxed answer
        data = []
        for sub in sorted(p for p in (DL.parent / "hendrycks_math_dl").iterdir() if p.is_dir()):
            df = pd.read_parquet(sub / "train.parquet")
            for r in df.itertuples():
                ans = std_bench._last_boxed(r.solution)
                if r.level in ("Level 4", "Level 5") and ans:
                    data.append({"q": r.problem + std_bench.SUFFIX, "a": ans})
    else:                                                        # MATH: MetaMathQA's MATH_AnsAug, one per question
        seen, data = set(), []
        for r in json.loads((DL / "metamathqa_40k.json").read_text()):
            ans = r["response"].split("The answer is:")[-1].strip()
            if r["type"] == "MATH_AnsAug" and r["query"] not in seen and ans:
                seen.add(r["query"])
                data.append({"q": r["query"] + std_bench.SUFFIX, "a": ans})
    grade = std_bench.grade_gsm8k if a.task == "gsm8k" else std_bench.grade_math
    print(f"task {a.task}: {len(data)} training questions", flush=True)
    order = list(range(len(data)))
    random.Random(a.seed).shuffle(order)

    # vLLM first: it sizes its KV cache from the memory free at start
    if a.base == "nf4" and a.sync != "exact":
        raise SystemExit("--base nf4 needs --sync exact")
    if (a.sync == "nvfp4merge") != (a.base == "nvfp4" and a.sync != "exact"):
        raise SystemExit("--sync nvfp4merge needs --base nvfp4 (and --base nvfp4 takes --sync exact or nvfp4merge)")
    if (a.sync == "fp8merge") != (a.base == "fp8" and a.sync != "exact"):
        raise SystemExit("--sync fp8merge needs --base fp8 (and --base fp8 takes --sync exact or fp8merge)")
    if a.engine == "fp8" and (a.base != "bf16" or a.sync != "exact"):
        raise SystemExit("--engine fp8 takes --base bf16 --sync exact")
    if a.match == "fp8w" and a.engine != "fp8":
        raise SystemExit("--match fp8w needs --engine fp8")
    extra = {"worker_extension_cls": "mlsys.vllm_ext.VariantWorker"} if a.sync != "exact" or a.base != "bf16" else {}
    if a.engine == "fp8":
        extra["quantization"] = "fp8"                            # vLLM's online FP8 (W8A8) on the bf16 checkpoint
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=max(16, a.lora_r), max_loras=1, max_cpu_loras=2,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.vllm_util, enforce_eager=not a.graphs, max_num_seqs=a.max_num_seqs,
              seed=a.seed, **({"enable_sleep_mode": True} if a.sleep else {}), **extra)
    print(f"engine: {a.engine}, LOWBIT_FUSED={os.environ.get('LOWBIT_FUSED', '0')}, graphs {a.graphs}, no-think {a.no_think}, task {a.task}", flush=True)
    if a.offload_base:                                           # the trainer is built while the engine sleeps
        t1 = time.time()
        llm.sleep(level=1)
        print(f"engine asleep for the trainer's setup ({time.time() - t1:.1f} s)", flush=True)
    if a.base != "bf16" and a.sync == "exact":                 # the engine's base -> NF4 (QLoRA inference) or NVFP4
        info = llm.collective_rpc("apply_variant", args=(json.dumps({"scheme": f"{a.base}_fresh", "role": "base"}),))[0]
        llm.reset_prefix_cache()
        print("engine base:", json.dumps(info), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map={"": "cpu" if a.cpu_load or a.offload_base else a.trainer_gpu})
    torch.cuda.set_device(a.trainer_gpu)
    if a.base in ("nf4", "fp8") or (a.base == "nvfp4" and not a.packed_scales):   # the trainer's base -> NF4 / FP8 / NVFP4
        import re
        from mlsys import merge
        n_q = 0
        with torch.no_grad():
            for name, mod in model.named_modules():
                if re.search(r"layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj$", name):
                    w = mod.weight.data.float()
                    mod.weight.data.copy_(merge.apply_scheme(w, torch.zeros_like(w), f"{a.base}_fresh")[0].to(torch.bfloat16))
                    n_q += 1
        print(f"trainer base: {a.base} on {n_q} projection weights", flush=True)
    elif a.base == "nvfp4":     # --packed-scales: the engine's own values. NVFP4 has a per-tensor scale, which the engine's
        # variant worker computes per 4,096-row chunk of vLLM's packed q/k/v and gate/up weights (mlsys vllm_ext
        # chunked_scheme); quantizing each projection on its own differs in 31-47% of entries (SYNC-NVFP4 note)
        from mlsys.vllm_ext import chunked_scheme
        n_q = 0
        with torch.no_grad():
            for layer in model.model.layers:
                for pack in (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                             ("mlp.gate_proj", "mlp.up_proj"), ("self_attn.o_proj",), ("mlp.down_proj",)):
                    mods = [layer.get_submodule(m) for m in pack]
                    Ws = [m.weight.data.float() for m in mods]
                    q = chunked_scheme(torch.cat(Ws), None, "nvfp4_fresh", "base", seed=0)
                    for m, qm in zip(mods, torch.split(q, [W.shape[0] for W in Ws])):
                        m.weight.data.copy_(qm.to(torch.bfloat16))
                        n_q += 1
        print(f"trainer base: nvfp4 with the engine's packed tensor scales on {n_q} projection weights", flush=True)
    if a.match == "fp8w":                                        # the trainer on the engine's FP8 weights (bf16 acts)
        from mlsys.matched_fp8 import patch_model
        print("trainer base: weight-matched FP8:", patch_model(model, parts="w" if a.fp8_train in ("none", "w16") else "both",
                                                             use_scaled_mm=torch.cuda.get_device_capability(a.trainer_gpu) >= (8, 9)
                                                             if a.cpu_load or a.offload_base else None), flush=True)
    if a.cpu_load or a.offload_base:                             # the (FP8-coded) model to the trainer's GPU (engine asleep)
        model.to(f"cuda:{a.trainer_gpu}")
        print(f"trainer on cuda:{a.trainer_gpu}: {torch.cuda.memory_allocated(a.trainer_gpu) / 2**30:.1f} GiB allocated", flush=True)
    import importlib.util
    print(f"trainer kernels: gated delta rule {'flash-linear-attention' if importlib.util.find_spec('fla') else 'PyTorch fallback'}, "
          f"fp8-train {a.fp8_train}, checkpointing {'off' if a.no_ckpt else 'on'}", flush=True)
    if a.fp8_train != "none":
        assert a.match == "fp8w", "--fp8-train needs --match fp8w"
        from lowbit_lora import fp8_train
        print("trainer FP8 backward:", fp8_train.fp8_backward(model, kind="w8a16" if a.fp8_train == "w16" else "w8a8"), flush=True)
    if not a.no_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    start, resumed = 0, None
    ckpts = sorted(out.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if ckpts:
        resumed = ckpts[-1].name
        start = int(resumed.split("_")[1])
        model = PeftModel.from_pretrained(model, str(ckpts[-1]), is_trainable=True)
    else:
        keep_old_log(out)
        cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    if a.fp8_train in ("fused", "w16"):                          # after either branch: a resumed run is fused too
        from lowbit_lora import train_lora
        print("trainer fused LoRA:", train_lora.install(model, kind=a.fp8_train), flush=True)
    dev_t = torch.device(f"cuda:{a.trainer_gpu}")
    frozen, host = [], []
    if a.offload_base:
        if not a.sleep or a.cpu_load or a.base != "bf16" or a.sync != "exact":
            raise SystemExit("--offload-base: --sleep, --base bf16, --sync exact, not --cpu-load")
        # frozen parameters and large buffers (the FP8 codes) -> pinned host copies, one tensor at a time
        frozen = [("p", p, None) for p in model.parameters() if not p.requires_grad and p.numel() > 0]
        frozen += [("b", mod, k) for mod in model.modules() for k, buf in mod._buffers.items()
                   if buf is not None and buf.numel() * buf.element_size() >= 1 << 20]
        host = []
        for kind, obj, k in frozen:
            if kind == "p":
                obj.data = obj.data.to("cpu").pin_memory()
                host.append(obj.data)
            else:
                obj._buffers[k] = obj._buffers[k].to("cpu").pin_memory()
                host.append(obj._buffers[k])
        torch.cuda.empty_cache()
        gib = sum(h.numel() * h.element_size() for h in host) / 2**30
        print(f"trainer base offloaded: {len(frozen)} tensors, {gib:.1f} GiB pinned on the CPU; on cuda:{a.trainer_gpu}: "
              f"{torch.cuda.memory_allocated(dev_t) / 2**30:.2f} GiB", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    (out / "grpo_meta.json").write_text(json.dumps({**vars(a), "resumed_from": resumed, "train_questions": len(data)},
                                                    indent=1))
    log = open(out / "log.jsonl", "a")
    tmp = out / "_sync"
    asleep, t_wake, t_sleep = bool(a.offload_base), 0.0, 0.0
    for step in range(start + 1, a.steps + 1):
        t0 = time.time()
        if asleep:                                               # --sleep: give the engine its memory back before generating
            torch.cuda.empty_cache()
            llm.wake_up()
            asleep, t_wake = False, time.time() - t0
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(str(tmp))                          # the exact adapter, as the engine will serve it
        sync_gain = None
        if a.sync == "exact":
            req = LoRARequest(f"policy{step}", step, str(tmp))
        else:                                                    # merge it into the pristine weights instead
            torch.cuda.empty_cache()                             # leave the engine room for the merge (48 GB GPUs)
            info = llm.collective_rpc("apply_variant", args=(json.dumps(
                {"scheme": {"bf16merge": "bf16_merge", "nvfp4merge": "nvfp4_frozen", "fp8merge": "fp8_frozen"}[a.sync], "role": "rl",
                 "adapter": str(tmp)}),))[0]
            llm.reset_prefix_cache()
            sync_gain, req = info.get("gain"), None
        idx = [order[((step - 1) * a.prompts + k) % len(order)] for k in range(a.prompts)]
        msgs = [data[i]["msgs"] if "msgs" in data[i] else [{"role": "user", "content": data[i]["q"]}] for i in idx]
        sp = SamplingParams(n=a.group, temperature=a.temperature, top_p=1.0, max_tokens=a.max_new,
                            seed=a.seed * 1_000_003 + step, **({"logprobs": 0} if lp_engine else {}))
        outs = llm.chat(msgs, sp, lora_request=req, use_tqdm=False,
                        **({"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else {}))
        t_gen = time.time() - t0
        if a.sleep:                                              # free the engine's weights and KV cache for the trainer
            t1 = time.time()
            llm.sleep(level=1)
            asleep, t_sleep = True, time.time() - t1
        seqs, adv, rewards, zero, elp = [], [], [], 0, []
        for i, o in zip(idx, outs):
            r = ([toolcall.reward(c.text, data[i]["gold"]) for c in o.outputs] if a.task == "tools"
                 else [grade(c.text, data[i]["a"]) for c in o.outputs])
            rewards += r
            m = sum(r) / len(r)
            sd = (sum((x - m) ** 2 for x in r) / len(r)) ** 0.5
            if sd == 0:
                zero += 1
                continue
            for c, x in zip(o.outputs, r):
                seqs.append((list(o.prompt_token_ids), list(c.token_ids)))
                adv.append((x - m) / (sd + 1e-6))
                if lp_engine:                                    # the engine's log-prob of each sampled token
                    elp.append([d[t].logprob for d, t in zip(c.logprobs, c.token_ids)])
        n_tok = sum(len(c) for _, c in seqs)
        loss_sum, mm_sum, mm_n = 0.0, 0.0, 0
        tis_sum, tis_clip, tis_n = 0.0, 0, 0
        t_load = t_offload = 0.0
        if a.offload_base:                                       # the engine is asleep: the trainer's base to the GPU
            t1 = time.time()
            torch.cuda.reset_peak_memory_stats(dev_t)
            for (kind, obj, k), h in zip(frozen, host):
                if kind == "p":
                    obj.data = h.to(dev_t, non_blocking=True)
                else:
                    obj._buffers[k] = h.to(dev_t, non_blocking=True)
            torch.cuda.synchronize(dev_t)
            t_load = time.time() - t1
        model.train()
        opt.zero_grad(set_to_none=True)
        for b in range(0, len(seqs), a.micro):
            chunk, ach = seqs[b:b + a.micro], adv[b:b + a.micro]
            L = max(len(p) + len(c) for p, c in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            w = torch.zeros((len(chunk), L - 1), dtype=torch.float32)
            for j, ((p, c), A) in enumerate(zip(chunk, ach)):
                ids[j, :len(p) + len(c)] = torch.tensor(p + c)
                att[j, :len(p) + len(c)] = 1
                w[j, len(p) - 1:len(p) + len(c) - 1] = A        # positions whose next token is a completion token
            dev = f"cuda:{a.trainer_gpu}"
            ids, att, w = ids.to(dev), att.to(dev), w.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=att).logits
            lp = token_logprobs(logits, ids)                     # fp32, chunked (train/token_lp.py)
            if a.tis > 0:                                        # w_t = min(pi_trainer / pi_engine, C), detached
                with torch.no_grad():
                    corr = torch.ones_like(w)
                    for j, (p, c) in enumerate(chunk):
                        sl = slice(len(p) - 1, len(p) + len(c) - 1)
                        e = torch.tensor(elp[b + j], device=lp.device)
                        ratio = torch.exp(lp[j, sl].detach() - e)
                        corr[j, sl] = ratio.clamp(max=a.tis)
                        tis_sum += float(corr[j, sl].sum())
                        tis_clip += int((ratio > a.tis).sum())
                        tis_n += len(c)
                loss = -(w * corr * lp).sum() / max(n_tok, 1)
            else:
                loss = -(w * lp).sum() / max(n_tok, 1)
            if elp:
                with torch.no_grad():
                    for j, (p, c) in enumerate(chunk):
                        e = torch.tensor(elp[b + j], device=lp.device)
                        mm_sum += float((lp[j, len(p) - 1:len(p) + len(c) - 1].detach() - e).abs().sum())
                        mm_n += len(c)
            loss.backward()
            loss_sum += float(loss)
            del logits, lp, loss
        if seqs:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        peak_t = None
        if a.offload_base:                                       # drop the GPU copies before the engine wakes
            t1 = time.time()
            peak_t = torch.cuda.max_memory_allocated(dev_t) / 2**30
            for (kind, obj, k), h in zip(frozen, host):
                if kind == "p":
                    obj.data = h
                else:
                    obj._buffers[k] = h
            torch.cuda.empty_cache()
            t_offload = time.time() - t1
        rec = {"step": step, "reward": sum(rewards) / len(rewards), "zero_std_groups": zero, "trained": len(seqs),
               "tokens": n_tok, "loss": loss_sum, "sync_gain": sync_gain,
               **({"mismatch": round(mm_sum / mm_n, 5) if mm_n else None} if lp_engine else {}),
               **({"tis_mean": round(tis_sum / tis_n, 5) if tis_n else None,
                   "tis_clip": round(tis_clip / tis_n, 6) if tis_n else None} if a.tis > 0 else {}),
               "len": sum(len(c.token_ids) for o in outs for c in o.outputs) /
               max(len(rewards), 1), "t_gen": round(t_gen, 1), "t": round(time.time() - t0, 1),
               **({"t_wake": round(t_wake, 2), "t_sleep": round(t_sleep, 2)} if a.sleep else {}),
               **({"t_load": round(t_load, 2), "t_offload": round(t_offload, 2), "trainer_peak_gib": round(peak_t, 2)}
                  if a.offload_base else {})}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 and step < a.steps:
            model.save_pretrained(str(out / f"step_{step}"))
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
