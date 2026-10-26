"""train/grpo_multi.py (kept unchanged for the streams that run it) for strong models (RL-FP8-QMATH): --task dapo
(DAPO-Math-17k, as train/grpo_q.py), --no-think (Qwen3's non-thinking chat mode), --graphs (the engine captures CUDA
graphs) and --max-model-len. Otherwise grpo_multi.py's doc:

K LoRA GRPO jobs on one rollout engine and one GPU (RL-MULTI-MATH): the multi-job arm of the MLSys RL study on MATH.

train/grpo_lora4.py's loop (MATH or GSM8K, exact adapters, one on-policy update per step, no clipping or KL; see its doc)
run for K independent jobs at once in one process: one vLLM engine (with --engine fp8 vLLM's online FP8, and with
LOWBIT_FUSED=1 our fused kernels) serves every job's exact adapter, and one trainer base, weight-matched to that engine
with --match fp8w (mlsys matched_fp8.patch_model, parts "w": the engine's FP8 codes and scales), carries K PEFT
adapters, each job with its own adapter, AdamW state, seed, question order and sampling seed (job k: --seeds[k]).

--pipeline overlaps the engine with the trainer while every job stays on-policy: jobs advance round-robin, and while the
trainer updates job k at step s the engine generates job k+1's rollouts for step s from job k+1's adapter after step
s-1, which is final by then (with K = 1 nothing can overlap and the loop is sequential). Without --pipeline each step
generates all jobs' rollouts in one batch and then trains the jobs in turn.

Writes <out>/job<k>/log.jsonl (one line per step: grpo_lora4's fields plus t_train, the wall time "wall" since the run
started, and "t" = this job's share of the step: for GPU-hours per run sum t), <out>/job<k>/step_<s>/ every
--save-every steps and the final adapter in <out>/job<k>. A restarted run resumes all jobs from their newest common
step_<s> (optimizer moments restart; recorded in grpo_meta.json).
    venv_vllm/bin/python train/grpo_multi.py --model <dir> --out <dir> --seeds 0 1 2 --task math --engine fp8 \
        --match fp8w --pipeline [--steps 200 --prompts 16 --group 8 --lr 2e-5 --lora-r 64 --max-new 1024]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import shutil
import sys
import threading
import time
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
DL = Path("$SCRATCH/data/std_bench_dl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True, help="one per job")
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
    ap.add_argument("--task", default="math", choices=("gsm8k", "math", "dapo"))
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="engine batch cap; with CUDA graphs on hybrid (GDN) models it must fit the state-cache blocks")
    ap.add_argument("--engine", default="fp8", choices=("bf16", "fp8"))
    ap.add_argument("--match", default="none", choices=("none", "fp8w"))
    ap.add_argument("--pipeline", action="store_true")
    a = ap.parse_args()
    K = len(a.seeds)
    if a.match == "fp8w" and a.engine != "fp8":
        raise SystemExit("--match fp8w needs --engine fp8")
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
    jd = [out / f"job{k}" for k in range(K)]
    for d in jd:
        d.mkdir(parents=True, exist_ok=True)
    if a.task == "gsm8k":
        df = pd.read_parquet(DL / "gsm8k_train.parquet")
        data = [{"q": r.question + std_bench.SUFFIX, "a": r.answer.split("####")[-1].strip().replace(",", "")}
                for r in df.itertuples()]
    elif a.task == "dapo":                                       # DAPO-Math-17k (deduplicated, English)
        df = pd.read_parquet(DL.parent / "dapo_dl/en/train.parquet")
        data = [{"q": r.prompt + std_bench.SUFFIX, "a": str(r.solution)} for r in df.itertuples()]
    else:                                                        # MATH: MetaMathQA's MATH_AnsAug, one per question
        seen, data = set(), []
        for r in json.loads((DL / "metamathqa_40k.json").read_text()):
            ans = r["response"].split("The answer is:")[-1].strip()
            if r["type"] == "MATH_AnsAug" and r["query"] not in seen and ans:
                seen.add(r["query"])
                data.append({"q": r["query"] + std_bench.SUFFIX, "a": ans})
    grade = std_bench.grade_gsm8k if a.task == "gsm8k" else std_bench.grade_math
    orders = []
    for s in a.seeds:                                            # each job's question order, as grpo_lora4 at that seed
        o = list(range(len(data)))
        random.Random(s).shuffle(o)
        orders.append(o)

    # vLLM first: it sizes its KV cache from the memory free at start
    extra = {"quantization": "fp8"} if a.engine == "fp8" else {}
    llm = LLM(model=a.model, enable_lora=True, max_lora_rank=max(16, a.lora_r), max_loras=K, max_cpu_loras=2 * K + 2,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.vllm_util, enforce_eager=not a.graphs, max_num_seqs=a.max_num_seqs,
              seed=a.seeds[0], **extra)
    print(f"engine: {a.engine}, LOWBIT_FUSED={os.environ.get('LOWBIT_FUSED', '0')}, jobs {K}, pipeline {a.pipeline}",
          flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    if a.match == "fp8w":                                        # one trainer base on the engine's FP8 weights
        from mlsys.matched_fp8 import patch_model
        print("trainer base: weight-matched FP8:", patch_model(model, parts="w"), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    names = [f"job{k}" for k in range(K)]
    steps_done = [sorted(int(p.name.split("_")[1]) for p in d.glob("step_*")) for d in jd]
    common = sorted(set.intersection(*map(set, steps_done))) if all(steps_done) else []
    start, resumed = (common[-1], f"step_{common[-1]}") if common else (0, None)
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    for k, n in enumerate(names):
        torch.manual_seed(a.seeds[k])                            # each job's own LoRA initialization
        if resumed:
            if k == 0:
                model = PeftModel.from_pretrained(model, str(jd[k] / resumed), adapter_name=n, is_trainable=True)
            else:
                model.load_adapter(str(jd[k] / resumed), adapter_name=n, is_trainable=True)
        elif k == 0:
            model = get_peft_model(model, cfg, adapter_name=n)
        else:
            model.add_adapter(n, cfg)
    params, opts = [], []
    for n in names:
        ps = [p for name, p in model.named_parameters() if f".{n}." in name and "lora_" in name]
        for p in ps:
            p.data = p.data.float()
        params.append(ps)
        opts.append(torch.optim.AdamW(ps, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0))
    print("adapter parameters per job:", [sum(p.numel() for p in ps) for ps in params], flush=True)
    (out / "grpo_meta.json").write_text(json.dumps({**vars(a), "jobs": K, "resumed_from": resumed,
                                                    "train_questions": len(data)}, indent=1))
    if not resumed:                                              # a failed attempt before any checkpoint
        for d in jd:
            keep_old_log(d)
    logs = [open(d / "log.jsonl", "a") for d in jd]
    sync = out / "_sync"
    sync.mkdir(exist_ok=True)

    def publish(k: int, step: int) -> LoRARequest:
        """Save job k's current adapter (its policy for this step) where the engine will load it."""
        model.set_adapter(names[k])
        d = sync / f"{names[k]}_s{step}"
        if d.exists():
            shutil.rmtree(d)
        model.save_pretrained(str(d), selected_adapters=[names[k]])
        path = d / names[k] if (d / names[k] / "adapter_config.json").exists() else d
        return LoRARequest(f"{names[k]}_s{step}", step * K + k + 1, str(path))

    def batch(k: int, step: int):
        idx = [orders[k][((step - 1) * a.prompts + j) % len(orders[k])] for j in range(a.prompts)]
        sp = SamplingParams(n=a.group, temperature=a.temperature, top_p=1.0, max_tokens=a.max_new,
                            seed=a.seeds[k] * 1_000_003 + step, logprobs=0)
        return idx, [[{"role": "user", "content": data[i]["q"]}] for i in idx], sp

    def generate(jobs: list[tuple[int, int, LoRARequest]]):
        """One engine call for the given (job, step, adapter) triples; returns each job's (idx, outputs)."""
        msgs, sps, reqs, spans, idxs = [], [], [], [], []
        for k, step, req in jobs:
            idx, m, sp = batch(k, step)
            spans.append((len(msgs), len(msgs) + len(m)))
            msgs += m
            sps += [sp] * len(m)
            reqs += [req] * len(m)
            idxs.append(idx)
        outs = llm.chat(msgs, sps, lora_request=reqs, use_tqdm=False,
                        **({"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else {}))
        return [(idx, outs[lo:hi]) for idx, (lo, hi) in zip(idxs, spans)]

    def train(k: int, idx, outs) -> dict:
        model.set_adapter(names[k])
        seqs, adv, rewards, zero, elp = [], [], [], 0, []
        for i, o in zip(idx, outs):
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
                elp.append([d[t].logprob for d, t in zip(c.logprobs, c.token_ids)])
        n_tok = sum(len(c) for _, c in seqs)
        loss_sum, mm_sum, mm_n = 0.0, 0.0, 0
        model.train()
        opts[k].zero_grad(set_to_none=True)
        for b in range(0, len(seqs), a.micro):
            chunk, ach = seqs[b:b + a.micro], adv[b:b + a.micro]
            L = max(len(p) + len(c) for p, c in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            w = torch.zeros((len(chunk), L - 1), dtype=torch.float32)
            for j, ((p, c), A) in enumerate(zip(chunk, ach)):
                ids[j, :len(p) + len(c)] = torch.tensor(p + c)
                att[j, :len(p) + len(c)] = 1
                w[j, len(p) - 1:len(p) + len(c) - 1] = A
            ids, att, w = ids.cuda(), att.cuda(), w.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=att).logits
            lp = token_logprobs(logits, ids)                     # fp32, chunked (train/token_lp.py)
            loss = -(w * lp).sum() / max(n_tok, 1)
            with torch.no_grad():
                for j, (p, c) in enumerate(chunk):
                    e = torch.tensor(elp[b + j], device=lp.device)
                    mm_sum += float((lp[j, len(p) - 1:len(p) + len(c) - 1].detach() - e).abs().sum())
                    mm_n += len(c)
            loss.backward()
            loss_sum += float(loss)
            del logits, lp, loss
        if seqs:
            torch.nn.utils.clip_grad_norm_(params[k], 1.0)
            opts[k].step()
        return {"reward": sum(rewards) / len(rewards), "zero_std_groups": zero, "trained": len(seqs), "tokens": n_tok,
                "loss": loss_sum, "sync_gain": None, "mismatch": round(mm_sum / mm_n, 5) if mm_n else None,
                "len": sum(len(c.token_ids) for o in outs for c in o.outputs) / max(len(rewards), 1)}

    def record(k: int, step: int, rec: dict):
        logs[k].write(json.dumps(rec) + "\n")
        logs[k].flush()
        print(json.dumps({"job": k, **rec}), flush=True)
        if step % a.save_every == 0 and step < a.steps:
            model.set_adapter(names[k])
            model.save_pretrained(str(jd[k] / f"step_{step}"), selected_adapters=[names[k]])
            nested = jd[k] / f"step_{step}" / names[k]          # PEFT nests non-default adapters: flatten for resume
            if nested.exists():
                for f in nested.iterdir():
                    shutil.move(str(f), str(jd[k] / f"step_{step}" / f.name))
                nested.rmdir()

    def cleanup(step: int):
        for d in sync.glob("*_s*"):
            if int(d.name.rsplit("_s", 1)[1]) < step - 1:
                shutil.rmtree(d, ignore_errors=True)

    t_run = time.time()
    if not a.pipeline or K == 1:
        for step in range(start + 1, a.steps + 1):
            t0 = time.time()
            reqs = [publish(k, step) for k in range(K)]
            res = generate([(k, step, reqs[k]) for k in range(K)])
            t_gen = time.time() - t0
            stats = []
            for k, (idx, outs) in enumerate(res):
                t1 = time.time()
                stats.append((train(k, idx, outs), time.time() - t1))
            t = time.time() - t0
            for k, (st, t_tr) in enumerate(stats):
                record(k, step, {"step": step, **st, "t_gen": round(t_gen, 1), "t_train": round(t_tr, 1),
                                 "wall": round(time.time() - t_run, 1), "t": round(t / K, 1)})
            cleanup(step)
    else:
        tasks = [(k, s) for s in range(start + 1, a.steps + 1) for k in range(K)]
        todo, done = queue.Queue(), queue.Queue()

        def worker():
            while True:
                item = todo.get()
                if item is None:
                    return
                k, step, req = item
                t0 = time.time()
                try:
                    done.put((k, step, generate([(k, step, req)])[0], time.time() - t0, None))
                except Exception as e:  # noqa: BLE001 -- surfaced in the main thread
                    done.put((k, step, None, 0.0, e))
        th = threading.Thread(target=worker, daemon=True)
        th.start()
        todo.put((tasks[0][0], tasks[0][1], publish(*tasks[0])))
        t_last = time.time()
        for i, (k, step) in enumerate(tasks):
            k2, step2, res, t_gen, err = done.get()
            if err is not None:
                raise err
            assert (k2, step2) == (k, step)
            if i + 1 < len(tasks):                               # the next job's policy is final: generate it now
                todo.put((tasks[i + 1][0], tasks[i + 1][1], publish(*tasks[i + 1])))
            t1 = time.time()
            st = train(k, *res)
            now = time.time()
            # this job's share of the GPU: the wall time since the previous task finished (the K jobs' tasks tile the run)
            record(k, step, {"step": step, **st, "t_gen": round(t_gen, 1), "t_train": round(now - t1, 1),
                             "wall": round(now - t_run, 1), "t": round(now - t_last, 1)})
            t_last = now
            if k == K - 1:
                cleanup(step)
        todo.put(None)
        th.join()
    for k, n in enumerate(names):
        model.set_adapter(n)
        model.save_pretrained(str(jd[k]), selected_adapters=[n])
        nested = jd[k] / n
        if nested.exists():
            for f in nested.iterdir():
                shutil.move(str(f), str(jd[k] / f.name))
            nested.rmdir()
        tok.save_pretrained(str(jd[k]))
    shutil.rmtree(sync, ignore_errors=True)
    print(f"saved {', '.join(map(str, jd))}; wall {time.time() - t_run:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
