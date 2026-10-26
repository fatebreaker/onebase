"""HOPPER-PDL-E2E (docs/REGISTRY.md): the first (cold) start of a mode and setting, and the output check against stock's.

The engine is built exactly as bench/overhead.py builds it (model, quantization, LoRA slots, lengths, memory share, seed,
prefix caching off) and warmed up the same way, so this start fills the compile caches the measured starts (overhead.py,
W1 and W2) reuse: it is the protocol's cold start. Then the output check:
  check   16 fixed prompts in one batch (a decode batch of 16), greedy, 32 generated tokens each (ignore_eos), top-5
          log-probs, requests on the setting's adapters round-robin as the workloads.
  With --ref (stock's dump of the same setting), the registered criterion: the check passes if all 16 continuations are
  token-identical to stock's, or else if the log-probs agree within 2e-2: the mean, over every prompt's positions up to
  and including its first divergence, of |log p(stock's token) - stock's log p| is at most 2e-2 (a stock token missing
  from this engine's top 5 counts as a failure).
  Descriptive, the same criterion: "strong" (the same prompts on random adapters whose lora_B has std 2e-2 instead of the
  protocol adapters' 1e-3, so that the adapters move the outputs and a misrouted adapter would show) and "b1" (four of
  the prompts decoded one at a time: the batch-1 bucket W1 uses).
Recorded (--out, one JSON row): load time (the cold-start time), KV-cache tokens, the check results, the plugin's layer
decisions and, with the PDL path, its per-layer dispatch (LOWBIT_REPORT lines "pdl-layer ... pdl_dispatch=b:path,...").
Every generated token and top-5 log-prob goes to --dump (JSON), the reference for the other modes.
--workloads W1,W2 also runs overhead.py's W1 and W2 in this start (the smoke run).
    python bench/e2e_outcheck.py --model <dir> --quantization fp8 [--adapter <dir> --rank 64 --n-adapters 1
        --strong-adapter <dir>] --label <label> --out <jsonl> --dump <json> [--ref <stock's dump>] [--workloads W1,W2]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

PROMPTS = ("The capital of France is", "def fibonacci(n):", "Water boils at a temperature of",
           "The three primary colors are", "Once upon a time, in a small village,", "import numpy as np\n",
           "The theory of relativity was developed by", "To make a cup of tea, first",
           "The largest planet in our solar system is", "In 1969, the first human to walk on the Moon",
           "SELECT name FROM users WHERE", "The square root of 144 is", "Photosynthesis is the process by which",
           "A haiku about the ocean:", "The main causes of the French Revolution were",
           "Translate to German: 'Good morning' ->")
N_GEN = 32
TOPK = 5
TOL = 2e-2


def compare(run: dict, ref: dict, tol: float = TOL) -> dict:
    """The registered criterion. run / ref: {"tokens": [[id, ...] per prompt], "logprobs": [[{id: lp} per position]]}
    (ids as strings in JSON). Positions compared per prompt: 0 .. first divergence (inclusive; all when identical)."""
    toks, rtoks = run["tokens"], ref["tokens"]
    assert len(toks) == len(rtoks), (len(toks), len(rtoks))
    devs, missing, div_at = [], 0, []
    for t, rt, lp, rlp in zip(toks, rtoks, run["logprobs"], ref["logprobs"]):
        n = min(len(t), len(rt))
        d = next((i for i in range(n) if t[i] != rt[i]), None)
        div_at.append(d)
        last = n - 1 if d is None else d
        for i in range(last + 1):
            want = str(rt[i])
            mine, theirs = lp[i], rlp[i]
            if want not in mine or want not in theirs:
                missing += 1
                continue
            devs.append(abs(mine[want] - theirs[want]))
    identical = all(d is None for d in div_at) and all(len(a) == len(b) for a, b in zip(toks, rtoks))
    mean_dev = sum(devs) / len(devs) if devs else float("inf")
    return {"tokens_identical": identical, "prompts_diverged": sum(d is not None for d in div_at),
            "first_divergence": div_at, "positions": len(devs) + missing, "missing_from_top5": missing,
            "mean_abs_dlogp": mean_dev, "max_abs_dlogp": max(devs) if devs else None,
            "ok": identical or (missing == 0 and mean_dev <= tol)}


def layer_report(path: str) -> dict:
    """The plugin's layer decisions (lines " dispatch=": fused / fallback per layer) and the PDL path's (lines
    "pdl-layer ... pdl_dispatch=b:path,..."): per bucket the number of layers on each path."""
    out = {"fused_layers": 0, "fallback_layers": 0, "pdl_layers": 0, "pdl_dispatch": {}, "pdl_failures": 0,
           "op_called": False}
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return {"report": None}
    for ln in lines:
        if " dispatch=" in ln:
            if " fused" in ln:
                out["fused_layers"] += 1
            elif " fallback" in ln:
                out["fallback_layers"] += 1
        elif " pdl-layer " in ln and "pdl_dispatch=" in ln:
            out["pdl_layers"] += 1
            for item in ln.split("pdl_dispatch=", 1)[1].split(","):
                b, _, v = item.partition(":")
                if b.isdigit():
                    out["pdl_dispatch"].setdefault(b, {}).setdefault(v, 0)
                    out["pdl_dispatch"][b][v] += 1
        elif "pdl-candidate-failed" in ln or "pdl-warmup-failed" in ln:
            out["pdl_failures"] += 1
        elif "op-called" in ln:
            out["op_called"] = True
    out["pdl_dispatch"] = dict(sorted(out["pdl_dispatch"].items(), key=lambda kv: int(kv[0])))
    return out


def gpu_info() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader", "-i", "0"],
                       capture_output=True, text=True).stdout.strip()
    name, _, cc = q.partition(",")
    return {"gpu": name.strip(), "cc": cc.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--strong-adapter", default="", help="random adapters of the same rank with lora_B std 2e-2")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--n-adapters", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--workloads", default="", help="W1,W2: also overhead.py's workloads in this start (smoke)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref", default="", help="stock's dump of the same setting")
    a = ap.parse_args()
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    kw: dict = {"enable_prefix_caching": False}                        # as bench/overhead.py
    if a.adapter:
        assert a.rank > 0 and a.n_adapters > 0
        kw.update(enable_lora=True, max_lora_rank=a.rank, max_loras=a.n_adapters, max_cpu_loras=a.n_adapters)
    if a.quantization:
        kw["quantization"] = a.quantization
    fused = os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False")
    if fused and not os.environ.get("LOWBIT_REPORT"):
        os.environ["LOWBIT_REPORT"] = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"lowbit_layers_{os.getpid()}.txt")
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util, seed=0, **kw)
    load_s = time.time() - t0
    loras = [LoRARequest(f"a{i}", i + 1, a.adapter) for i in range(a.n_adapters)] if a.adapter else []
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)

    def lr(n: int):
        return [loras[i % len(loras)] for i in range(n)] if loras else None

    def prompts(n: int, length: int) -> list[dict]:
        return [{"prompt_token_ids": [rng.randrange(1000, vocab - 1000) for _ in range(length)]} for _ in range(n)]

    def sp(n_out: int, **extra) -> SamplingParams:
        return SamplingParams(max_tokens=n_out, ignore_eos=True, temperature=0.0, **extra)

    def run(ps, n_out: int, reqs=None, **extra):
        t = time.perf_counter()
        outs = llm.generate(ps, sp(n_out, **extra), lora_request=reqs, use_tqdm=False)
        return time.perf_counter() - t, outs

    n_warm = max(8, len(loras))
    run(prompts(n_warm, 64), 16, lr(n_warm))                          # overhead.py's warm-up, twice
    run(prompts(n_warm, 64), 16, lr(n_warm))

    def gen(ps: list[str], reqs) -> dict:
        _, outs = run(ps, N_GEN, reqs, logprobs=TOPK)
        res = {"tokens": [], "logprobs": []}
        for o in outs:
            c = o.outputs[0]
            res["tokens"].append(list(c.token_ids))
            res["logprobs"].append([{str(k): float(v.logprob) for k, v in pos.items()} for pos in (c.logprobs or [])])
        return res

    dump = {"label": a.label, "prompts": list(PROMPTS), "n_gen": N_GEN, "topk": TOPK}
    dump["check"] = gen(list(PROMPTS), lr(len(PROMPTS)))
    if loras and a.strong_adapter:
        strong = [LoRARequest(f"s{i}", 1001 + i, a.strong_adapter) for i in range(min(4, a.n_adapters))]
        dump["strong"] = gen(list(PROMPTS), [strong[i % len(strong)] for i in range(len(PROMPTS))])
    b1 = {"tokens": [], "logprobs": []}
    for i in range(4):
        g = gen([PROMPTS[i]], [loras[i % len(loras)]] if loras else None)
        b1["tokens"] += g["tokens"]
        b1["logprobs"] += g["logprobs"]
    dump["b1"] = b1
    Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
    Path(a.dump).write_text(json.dumps(dump))
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        kv_tokens = int(cc.num_gpu_blocks) * int(cc.block_size)
    except (AttributeError, TypeError):
        kv_tokens = None
    rec = {"label": a.label, "model": Path(a.model).name, "quantization": a.quantization or "bf16", "rank": a.rank,
           "n_adapters": a.n_adapters, "lora": bool(a.adapter),
           "dual_stream": os.environ.get("VLLM_LORA_ENABLE_DUAL_STREAM", "0") not in ("0", "", "false", "False"),
           "lowbit_fused": fused, "lowbit_pdl": os.environ.get("LOWBIT_PDL", "0") == "1",
           "tuned_folder": os.environ.get("VLLM_TUNED_CONFIG_FOLDER", ""), "load_s": round(load_s, 1),
           "kv_tokens": kv_tokens, "layers": layer_report(os.environ.get("LOWBIT_REPORT", "")) if fused else None,
           "dump": a.dump, "ref": a.ref or None, **gpu_info(), "vllm": vllm.__version__, "torch": torch.__version__,
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "job": os.environ.get("SLURM_JOB_ID", "")}
    if a.ref:
        ref = json.loads(Path(a.ref).read_text())
        rec["check"] = compare(dump["check"], ref["check"])
        for part in ("strong", "b1"):
            if part in dump and part in ref:
                rec["check_" + part] = compare(dump[part], ref[part])
    wl = set(x for x in a.workloads.split(",") if x)
    if "W1" in wl:   # overhead.py's W1 and W2, unchanged
        p = prompts(1, 512)
        tp = []
        for _ in range(3):
            t1, _ = run(p, 1, lr(1))
            t257, o = run(p, 257, lr(1))
            assert len(o[0].outputs[0].token_ids) == 257
            tp.append((t257 - t1) / 256)
        rec["W1_tpot_ms"] = round(1000 * statistics.median(tp), 3)
        rec["W1_tpot_ms_all"] = [round(1000 * x, 3) for x in tp]
    if "W2" in wl:
        dt, outs = run(prompts(256, 512), 256, lr(256))
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        rec["W2_gen_tok_s"] = round(n / dt, 1)
        rec["W2_s"] = round(dt, 2)
    print(json.dumps(rec), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if a.ref and not rec["check"]["ok"]:
        print(f"OUTPUT CHECK FAILED against {a.ref}: {json.dumps(rec['check'])[:400]}", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
