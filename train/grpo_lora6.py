"""train/grpo_lora5.py (kept unchanged for the streams that run it) plus --task tools (TOOL-RL): GRPO on function calling,
the prompts ToolACE's (Team-ACE/ToolACE, Apache-2.0) single-call-turn examples in BFCL's prompting format (bench/
bfcl_tools.py prep-toolace: BFCL's system prompt listing the example's functions with BFCL's Python hint, then the user's
request; 4,850 examples whose reference calls BFCL's decoder reads), the reward 1 when the response's calls equal the
reference calls (bench/toolcall.py: BFCL's decoding, then names, keyword sets and values equal as a multiset), else 0;
and --graphs (the engine captures CUDA graphs instead of running eager, as grpo_lora7.py).

train/grpo_lora4.py (kept unchanged for the streams that run it) plus --task mbpp (CODE-RL): GRPO on MBPP's train and
validation splits (464 problems, task ids 511-974; disjoint from the test split that the sanitized MBPP test is drawn
from), each prompt in std_bench.py's MBPP format (the problem, its assert tests, "Give the function in a ```python code
block."), the reward 1 if the last code block passes all the problem's tests (std_bench.run_program: subprocess, CPU and
memory limits, 20 s timeout), run 16 at a time. The data are the dataset's parquet files (data/mbpp_full_dl/full).

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
    ap.add_argument("--task", default="gsm8k", choices=("gsm8k", "math", "mbpp", "tools"))
    ap.add_argument("--graphs", action="store_true", help="the engine captures CUDA graphs (default: eager)")
    ap.add_argument("--engine", default="bf16", choices=("bf16", "fp8"))
    ap.add_argument("--match", default="none", choices=("none", "fp8w"))
    ap.add_argument("--sync", default="exact", choices=("exact", "bf16merge", "nvfp4merge", "fp8merge"))
    ap.add_argument("--base", default="bf16", choices=("bf16", "nf4", "nvfp4", "fp8"))
    ap.add_argument("--packed-scales", action="store_true",
                    help="--base nvfp4: compute the trainer's NVFP4 values as the engine does (SYNC-NVFP4-M)")
    a = ap.parse_args()
    lp_engine = True                                            # log the trainer-engine mismatch in every arm
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import pandas as pd
    import torch
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
    elif a.task == "tools":                                      # ToolACE in BFCL's prompting format
        import toolcall
        data = [{"msgs": r["messages"], "gold": r["gold"]} for r in
                map(json.loads, (DL.parent / "toolace_dl/toolace_rl.jsonl").read_text().splitlines())
                if sum(len(m["content"]) for m in r["messages"]) <= 8000]   # drops 77 of 4,850 (trainer memory)
    elif a.task == "mbpp":                                       # MBPP train + validation, std_bench's prompt format
        data = []
        for split in ("train", "validation"):
            df = pd.read_parquet(DL.parent / "mbpp_full_dl/full" / f"{split}-00000-of-00001.parquet")
            for r in df.itertuples():
                tests = list(r.test_list)
                data.append({"q": f"{r.text}\nYour code should pass these tests:\n" + "\n".join(tests)
                             + "\nGive the function in a ```python code block.",
                             "imports": [r.test_setup_code] if r.test_setup_code else [], "tests": tests})
    else:                                                        # MATH: MetaMathQA's MATH_AnsAug, one per question
        seen, data = set(), []
        for r in json.loads((DL / "metamathqa_40k.json").read_text()):
            ans = r["response"].split("The answer is:")[-1].strip()
            if r["type"] == "MATH_AnsAug" and r["query"] not in seen and ans:
                seen.add(r["query"])
                data.append({"q": r["query"] + std_bench.SUFFIX, "a": ans})
    grade = std_bench.grade_gsm8k if a.task == "gsm8k" else std_bench.grade_math
    if a.task == "mbpp":
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=16)
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
              max_model_len=4096 if a.task == "tools" else 2048, gpu_memory_utilization=a.vllm_util,
              enforce_eager=not a.graphs, seed=a.seed, **extra)
    print(f"engine: {a.engine}, LOWBIT_FUSED={os.environ.get('LOWBIT_FUSED', '0')}", flush=True)
    if a.base != "bf16" and a.sync == "exact":                 # the engine's base -> NF4 (QLoRA inference) or NVFP4
        info = llm.collective_rpc("apply_variant", args=(json.dumps({"scheme": f"{a.base}_fresh", "role": "base"}),))[0]
        llm.reset_prefix_cache()
        print("engine base:", json.dumps(info), flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
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
        print("trainer base: weight-matched FP8:", patch_model(model, parts="w"), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    start, resumed = 0, None
    ckpts = sorted(out.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if ckpts:
        resumed = ckpts[-1].name
        start = int(resumed.split("_")[1])
        model = PeftModel.from_pretrained(model, str(ckpts[-1]), is_trainable=True)
    else:
        cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    (out / "grpo_meta.json").write_text(json.dumps({**vars(a), "resumed_from": resumed, "train_questions": len(data)},
                                                    indent=1))
    log = open(out / "log.jsonl", "a")
    tmp = out / "_sync"
    for step in range(start + 1, a.steps + 1):
        t0 = time.time()
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(str(tmp))                          # the exact adapter, as the engine will serve it
        sync_gain = None
        if a.sync == "exact":
            req = LoRARequest(f"policy{step}", step, str(tmp))
        else:                                                    # merge it into the pristine weights instead
            info = llm.collective_rpc("apply_variant", args=(json.dumps(
                {"scheme": {"bf16merge": "bf16_merge", "nvfp4merge": "nvfp4_frozen", "fp8merge": "fp8_frozen"}[a.sync], "role": "rl",
                 "adapter": str(tmp)}),))[0]
            llm.reset_prefix_cache()
            sync_gain, req = info.get("gain"), None
        idx = [order[((step - 1) * a.prompts + k) % len(order)] for k in range(a.prompts)]
        msgs = [data[i]["msgs"] if "msgs" in data[i] else [{"role": "user", "content": data[i]["q"]}] for i in idx]
        sp = SamplingParams(n=a.group, temperature=a.temperature, top_p=1.0, max_tokens=a.max_new,
                            seed=a.seed * 1_000_003 + step, **({"logprobs": 0} if lp_engine else {}))
        outs = llm.chat(msgs, sp, lora_request=req, use_tqdm=False)
        t_gen = time.time() - t0
        seqs, adv, rewards, zero, elp = [], [], [], 0, []
        for i, o in zip(idx, outs):
            if a.task == "mbpp":
                r = list(pool.map(lambda c: std_bench.run_program(std_bench.program("mbpp", data[i], c.text)), o.outputs))
            elif a.task == "tools":
                r = [toolcall.reward(c.text, data[i]["gold"]) for c in o.outputs]
            else:
                r = [grade(c.text, data[i]["a"]) for c in o.outputs]
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
            ids, att, w = ids.cuda(), att.cuda(), w.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
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
        rec = {"step": step, "reward": sum(rewards) / len(rewards), "zero_std_groups": zero, "trained": len(seqs),
               "tokens": n_tok, "loss": loss_sum, "sync_gain": sync_gain,
               **({"mismatch": round(mm_sum / mm_n, 5) if mm_n else None} if lp_engine else {}),
               "len": sum(len(c.token_ids) for o in outs for c in o.outputs) /
               max(len(rewards), 1), "t_gen": round(t_gen, 1), "t": round(time.time() - t0, 1)}
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
