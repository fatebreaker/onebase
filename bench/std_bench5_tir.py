"""bench/std_bench4.py (kept unchanged for the streams that run it) plus --tir (TIR-RL): math benchmarks solved with
tool-integrated reasoning (bench/tir.py: the question with the TIR instruction instead of the boxed-answer one, up to
--tir-rounds sandboxed Python calls within --max-tokens), graded on the trajectory's final boxed answer; avg@K replicates
each problem K times with its own seed. Otherwise std_bench4.py's doc:

bench/std_bench3.py (kept unchanged for the streams that run it) plus --no-think: Qwen3's non-thinking chat mode
(chat_template_kwargs enable_thinking=False), as train/grpo_q.py --no-think trains. Otherwise std_bench3.py's doc:

bench/std_bench2.py (kept unchanged for the streams that run it) plus bfcl (TOOL-RL): the single-turn BFCL categories
in BFCL's prompting format (bench/bfcl_tools.py prep-bfcl: 3,475 chat prompts with BFCL's system prompt), greedy; the
responses go to <variant>/bfcl_raw.jsonl unscored, and bench/bfcl_tools.py score (BFCL's own decoder and checker, in its
venv) writes <variant>/bfcl.jsonl. --max-model-len (default 4096; BFCL's longest prompt needs ~11k tokens).

bench/std_bench.py (kept unchanged for the streams that run it) plus harder math (HARD-MATH, user 10-15: GSM8K and MATH
are easy): olympiad (OlympiadBench's English open-ended competition math, OE_TO_maths_en_COMP, 675; the gold the first
final answer with $ removed), minerva (Minerva Math, 272; the gold the solution's last boxed answer, e-notation written as
x 10^k), aime24, aime25 (30 each) and amc23 (40), all graded with math-verify as math500 is; the OlympiadBench and Minerva
files are Qwen2.5-Math's evaluation copies (data/hardmath_dl). --samples K (with --temperature/--top-p) draws K samples
per problem and records each problem's mean ("correct", a fraction) and the K outcomes ("corrects"): avg@K for the small
sets. Otherwise std_bench.py's doc:

STD-BENCH evaluation (docs/REGISTRY.md, registered 2026-10-13 20:59 UTC): one LoRA adapter under every serving variant on
popular math or coding benchmarks, in one vLLM engine (mlsys VariantWorker rewrites the base's weights in place; exact and
stored adapters are served unmerged as LoRA requests). Greedy decoding through the model's chat template.

  math adapters  gsm8k (1,319; the boxed answer, else "The answer is", else the last number) and math500 (math-verify)
  code adapters  humaneval (164) and mbpp (sanitized test, 257): the last ```python block, executed with the benchmark's
                 tests in an isolated subprocess (python -I, a temporary directory, 10 s CPU, 4 GB address space,
                 human-eval's reliability guard: destructive os/shutil/subprocess functions disabled)
--selftest runs the benchmarks' own reference solutions through the same grader and exits (no GPU).

    PYTHONPATH=<lowbit-lora>:<mlsys>:<mlsys>/pylib venv_vllm/bin/python bench/std_bench.py --model <dir> --adapter <dir> \
        [--stored <NVFP4-stored adapter dir>] --benches gsm8k math500 --out <dir> [--variants ...]
Writes <out>/<variant>/<bench>.jsonl (id, correct, response) per finished (variant, bench); a rerun skips those.
--engine fp8 (the MLSys check): vLLM's own FP8 serving (--quantization fp8) with variants fp8eng_base and fp8eng_exact,
for vLLM's LoRA path (LOWBIT_FUSED=0) or the plugin (LOWBIT_FUSED=1, lowbit_lora on PYTHONPATH); give each its own --out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path("$SCRATCH/data")
DL = DATA / "std_bench_dl"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ALL_VARIANTS = ("bf16_base", "bf16_exact", "nvfp4_base", "nvfp4_exact", "nvfp4_stored", "nvfp4_merge", "nvfp4_srmerge",
                "fp8_base", "fp8_exact", "fp8_merge", "fp8_srmerge")


def load_bench(name: str) -> list[dict]:
    import pandas as pd
    if name == "gsm8k":
        df = pd.read_parquet(DL / "gsm8k_test.parquet")
        return [{"id": f"gsm8k-{i}", "q": r.question + SUFFIX, "a": r.answer.split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(df.itertuples())]
    if name == "math500":
        rows = [json.loads(x) for x in (DATA / "MATH-500/test.jsonl").read_text().splitlines()]
        return [{"id": f"math500-{r['unique_id']}", "q": r["problem"] + SUFFIX, "a": str(r["answer"])} for r in rows]
    if name == "humaneval":
        df = pd.read_parquet(DL / "humaneval_test.parquet")
        return [{"id": r.task_id, "q": "Complete the following Python function. Give the whole function in a ```python "
                 "code block.\n```python\n" + r.prompt + "```", "prompt": r.prompt, "test": r.test, "entry": r.entry_point,
                 "ref": r.prompt + r.canonical_solution} for r in df.itertuples()]
    if name == "mbpp":
        df = pd.read_parquet(DL / "mbpp_sanitized_test.parquet")
        return [{"id": f"mbpp-{r.task_id}", "q": f"{r.prompt}\nYour code should pass these tests:\n"
                 + "\n".join(r.test_list) + "\nGive the function in a ```python code block.",
                 "imports": list(r.test_imports), "tests": list(r.test_list), "ref": r.code} for r in df.itertuples()]
    if name == "olympiad":
        rows = [json.loads(x) for x in (DATA / "hardmath_dl/olympiadbench/test.jsonl").read_text().splitlines()]
        return [{"id": f"olympiad-{r['id']}", "q": r["question"] + SUFFIX, "a": r["final_answer"][0].replace("$", "").strip()}
                for r in rows]
    if name == "minerva":
        rows = [json.loads(x) for x in (DATA / "hardmath_dl/minerva_math/test.jsonl").read_text().splitlines()]
        out = []
        for r in rows:
            g = _last_boxed(r["solution"]) or ""
            g = re.sub(r"^(-?\d+(?:\.\d+)?)\s*e\s*([+-]?\d+)$", r"\1 \\times 10^{\2}", g.strip())
            out.append({"id": f"minerva-{r['idx']}", "q": r["problem"] + SUFFIX, "a": g})
        return out
    if name in ("aime24", "aime25", "amc23"):
        f = {"aime24": "aime_2024/data/train-00000-of-00001.parquet", "aime25": "aime_2025/data/train-00000-of-00001.parquet",
             "amc23": "amc23/test-00000-of-00001.parquet"}[name]
        df = pd.read_parquet(DATA / f)
        qcol = "question" if "question" in df.columns else "problem"
        return [{"id": f"{name}-{i}", "q": str(getattr(r, qcol)) + SUFFIX, "a": str(r.answer)} for i, r in enumerate(df.itertuples())]
    if name == "bfcl":
        return [{"id": r["id"], "msgs": r["messages"]} for r in
                map(json.loads, (DATA / "bfcl_dl/bfcl_prompts.jsonl").read_text().splitlines())]
    raise SystemExit(f"unknown benchmark {name}")


MATH_BENCHES = ("math500", "olympiad", "minerva", "aime24", "aime25", "amc23")


def _last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j:k]
    return None


def _num(x: str) -> float | None:
    x = x.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip().rstrip(".")
    m = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", x)
    if not m:
        return None
    t = m[-1]
    try:
        if "/" in t:
            a, b = t.split("/")
            return float(a) / float(b)
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def grade_gsm8k(resp: str, gold: str) -> int:
    cand = _last_boxed(resp)
    if cand is None:
        m = re.findall(r"answer is[:\s]*([^\n]+)", resp, flags=re.IGNORECASE)
        cand = m[-1] if m else resp
    v, g = _num(cand), _num(gold)
    return int(v is not None and g is not None and abs(v - g) < 1e-6)


def grade_math(resp: str, gold: str) -> int:
    from math_verify import parse, verify
    try:
        g = parse(f"${gold}$") or parse(gold)
        return int(bool(verify(g, parse(resp))))
    except Exception:                       # noqa: BLE001 -- math-verify raises on odd input
        return 0


def _code(resp: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", resp, flags=re.DOTALL)
    return blocks[-1] if blocks else resp


# human-eval's reliability_guard (execution.py), run before the generated code: not a sandbox, a guard against
# destructive accidents
GUARD = """import builtins, os, shutil, subprocess, sys
os.environ["OMP_NUM_THREADS"] = "1"
builtins.exit = None
builtins.quit = None
for _n in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty", "killpg",
           "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
           "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
    setattr(os, _n, None)
shutil.rmtree = shutil.move = shutil.chown = None
subprocess.Popen = None
for _m in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[_m] = None
"""


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setsid()


def run_program(src: str) -> int:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"
        f.write_text(GUARD + src)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, timeout=20,
                               preexec_fn=_limits, stdin=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin"})
            return int(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0


def program(bench: str, item: dict, resp: str) -> str:
    code = _code(resp)
    if bench == "humaneval":
        body = code if re.search(rf"def\s+{re.escape(item['entry'])}\s*\(", code) else item["prompt"] + code
        return body + "\n\n" + item["test"] + f"\n\ncheck({item['entry']})\n"
    return "\n".join(item["imports"]) + "\n" + code + "\n\n" + "\n".join(item["tests"]) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--stored", default="", help="the adapter stored in NVFP4 (lowbit_lora.adapters quantize)")
    ap.add_argument("--benches", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--engine", default="emul", choices=("emul", "fp8"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--merge-seed", type=int, default=0, help="seed of the stochastic merges (SR-SEED: 1, 2)")
    ap.add_argument("--samples", type=int, default=1, help="samples per problem (avg@K); 1 = one completion")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-think", action="store_true", help="Qwen3 non-thinking chat mode")
    ap.add_argument("--tir", action="store_true", help="tool-integrated reasoning on the math benchmarks (bench/tir.py)")
    ap.add_argument("--tir-rounds", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N items per benchmark (smoke tests)")
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="engine batch cap (with CUDA graphs, hybrid GDN models need it within their state-cache blocks)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    a = ap.parse_args()
    if a.selftest:
        for b in a.benches:
            items = load_bench(b)
            with ThreadPoolExecutor(max_workers=8) as ex:
                ok = list(ex.map(lambda it: run_program(program(b, it, "```python\n" + it["ref"] + "\n```")), items))
            bad = [it["id"] for it, c in zip(items, ok) if not c]
            print(json.dumps({"selftest": b, "pass": sum(ok), "n": len(ok), "failing": bad[:20]}), flush=True)
        return 0
    if a.engine == "fp8":
        a.variants = [v for v in a.variants if v.startswith("fp8eng_")] or ["fp8eng_base", "fp8eng_exact"]
    # FlashInfer's sampler compiles on first use and needs nvcc, which the pods lack (jobs/lib.sh sets the same);
    # greedy decoding does not use it
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    out = Path(a.out)
    benches = {b: load_bench(b)[:a.limit] if a.limit else load_bench(b) for b in a.benches}
    fname = lambda b: "bfcl_raw" if b == "bfcl" else b                    # noqa: E731 -- bfcl is scored afterwards
    todo = [(v, b) for v in a.variants for b in a.benches if not (out / v / f"{fname(b)}.jsonl").exists()]
    if not todo:
        print("all evaluated")
        return 0
    if a.engine == "fp8":
        llm = LLM(model=a.model, enable_lora=True, max_lora_rank=64, max_loras=2, max_model_len=a.max_model_len, seed=0, max_num_seqs=a.max_num_seqs,
                  gpu_memory_utilization=a.gpu_util, quantization="fp8")
    else:
        llm = LLM(model=a.model, enable_lora=True, max_lora_rank=64, max_loras=2, max_model_len=a.max_model_len, seed=0, max_num_seqs=a.max_num_seqs,
                  gpu_memory_utilization=a.gpu_util, worker_extension_cls="mlsys.vllm_ext.VariantWorker")
    sp = SamplingParams(n=a.samples, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=0)
    exact, stored = LoRARequest("exact", 1, a.adapter), (LoRARequest("stored", 2, a.stored) if a.stored else None)
    specs = {"bf16": {"scheme": "none"},
             "nvfp4": {"scheme": "nvfp4_fresh", "role": "base", "adapter": a.adapter},
             "fp8": {"scheme": "fp8_fresh", "role": "base", "adapter": a.adapter},
             "nf4": {"scheme": "nf4_fresh", "role": "base", "adapter": a.adapter}}
    for v in a.variants:
        if not any(x == v for x, _ in todo):
            continue
        fmt, kind = v.split("_", 1)
        if fmt == "fp8eng":
            spec = None
        elif fmt == "nf4" and kind in ("merge", "freshmerge"):   # QLORA-MERGE: PEFT's merge into a bitsandbytes
            # 4-bit base re-quantizes with recomputed block maxima (nf4_requant); freshmerge merges in bf16 with the
            # master weights and quantizes fresh (nf4_fresh, role rl)
            spec = {"scheme": "nf4_requant" if kind == "merge" else "nf4_fresh", "role": "rl", "adapter": a.adapter}
        elif kind in ("merge", "srmerge"):
            spec = {"scheme": f"{fmt}_{'frozen' if kind == 'merge' else 'srfrozen'}", "role": "rl", "adapter": a.adapter,
                    "seed": a.merge_seed}
        else:
            spec = specs[fmt]
        req = {"exact": exact, "stored": stored}.get(kind)
        if kind == "stored" and stored is None:
            print(f"{v}: no stored adapter given, skipped")
            continue
        t0 = time.time()
        info = llm.collective_rpc("apply_variant", args=(json.dumps(spec),)) if spec is not None else "fp8 engine"
        llm.reset_prefix_cache()
        for b, items in benches.items():
            f = out / v / f"{fname(b)}.jsonl"
            if f.exists():
                continue
            msgs = [it["msgs"] if "msgs" in it else [{"role": "user", "content": it["q"]}] for it in items]
            if a.tir and b in MATH_BENCHES:                      # multi-round rollouts with the Python sandbox
                import types
                import tir
                tk = llm.get_tokenizer()
                msgs = [[{"role": "user", "content": it["q"].replace(SUFFIX, tir.TIR_SUFFIX)}] for it in items]
                pt = [tk.encode(tk.apply_chat_template(m, add_generation_prompt=True, tokenize=False,
                                                       **({"enable_thinking": False} if a.no_think else {})),
                                add_special_tokens=False) for m in msgs]
                K = a.samples
                mk = lambda i, mt, **kw: SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=mt,  # noqa: E731
                                                        seed=i, logprobs=0, **kw)
                tr = tir.rollout(llm, tk, [p_ for p_ in pt for _ in range(K)], mk, lora_request=req,
                                 rounds=a.tir_rounds, budget=a.max_tokens)
                outs = [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t.text) for t in tr[q * K:(q + 1) * K]])
                        for q in range(len(items))]
            else:
                outs = llm.chat(msgs, sp, lora_request=req, use_tqdm=False,
                                **({"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else {}))
            if a.samples > 1:                                   # avg@K: each problem's mean over its K samples
                g = grade_gsm8k if b == "gsm8k" else grade_math
                ks = [[g(c.text, it["a"]) for c in o.outputs] for o, it in zip(outs, items)]
                resps = [o.outputs[0].text for o in outs]
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text("".join(json.dumps({"id": it["id"], "correct": sum(k) / len(k), "corrects": k, "k": len(k),
                                                   "response": r}) + "\n" for it, k, r in zip(items, ks, resps)))
                os.replace(tmp, f)
                print(json.dumps({"variant": v, "bench": b, "acc": round(100 * sum(sum(k) / len(k) for k in ks) / len(ks), 2),
                                  "n": len(ks), "k": a.samples, "s": round(time.time() - t0)}), flush=True)
                continue
            resps = [o.outputs[0].text for o in outs]
            if b == "bfcl":                                      # unscored: bench/bfcl_tools.py score
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text("".join(json.dumps({"id": it["id"], "response": r}) + "\n" for it, r in zip(items, resps)))
                os.replace(tmp, f)
                print(json.dumps({"variant": v, "bench": b, "n": len(resps), "s": round(time.time() - t0)}), flush=True)
                continue
            if b == "gsm8k" or b in MATH_BENCHES:
                g = grade_gsm8k if b == "gsm8k" else grade_math
                correct = [g(r, it["a"]) for r, it in zip(resps, items)]
            else:
                with ThreadPoolExecutor(max_workers=8) as ex:
                    correct = list(ex.map(lambda p: run_program(program(b, p[0], p[1])), zip(items, resps)))
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f".{os.getpid()}.tmp")       # another evaluator may finish the same file
            tmp.write_text("".join(json.dumps({"id": it["id"], "correct": c, "response": r}) + "\n"
                                   for it, c, r in zip(items, correct, resps)))
            os.replace(tmp, f)
            print(json.dumps({"variant": v, "bench": b, "acc": round(100 * sum(correct) / len(correct), 2),
                              "n": len(correct), "s": round(time.time() - t0)}), flush=True)
        print(f"{v} applied: {json.dumps(info)[:200]}", flush=True)
    if a.engine == "emul":
        llm.collective_rpc("apply_variant", args=(json.dumps({"scheme": "none"}),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
