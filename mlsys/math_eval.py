"""Math evals of rounding variants for public full-parameter RL pairs (deployment study; GPU, vLLM + VariantWorker).

Serve the RL checkpoint (its embeddings, norms and head are the RL model's) and pass its pre-RL base as --base-other:
each variant rounds the update W_RL - W_base with a merge.apply_scheme scheme (role rl). Serve the base checkpoint
itself for the base_q variants (role base). math-verify grades the boxed answer (PYTHONPATH=<mlsys>/pylib).

    PYTHONPATH=<mlsys>/pylib venv_vllm/bin/python -m mlsys.math_eval --model <ckpt> [--base-other <pre-RL ckpt>] \
        --variants none:rl int4g128_fresh:rl ... --sets aime24 aime25 amc23 math500 --n-set aime24=16 math500=4 \
        --out-dir results/math/<pair>

Each variant writes <out-dir>/<scheme>_<role>.jsonl (one row per problem: correct of n) when it finishes, so a
re-run skips finished variants. Sampling uses one fixed seed for every variant (common random numbers): the paired
differences between variants carry less sampling noise than independent draws would.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DATA = Path("$SCRATCH/data")
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def load_set(name: str) -> list[dict]:
    import pandas as pd
    if name == "aime24":
        df = pd.read_parquet(DATA / "aime_2024/data/train-00000-of-00001.parquet")
        return [{"id": f"aime24-{r.id}", "q": r.problem, "a": str(r.answer)} for r in df.itertuples()]
    if name == "aime25":
        df = pd.read_parquet(DATA / "aime_2025/data/train-00000-of-00001.parquet")
        return [{"id": f"aime25-{r.problem_idx}", "q": r.problem, "a": str(r.answer)} for r in df.itertuples()]
    if name == "amc23":
        df = pd.read_parquet(DATA / "amc23/test-00000-of-00001.parquet")
        return [{"id": f"amc23-{r.id}", "q": r.question, "a": str(r.answer)} for r in df.itertuples()]
    if name == "math500":
        rows = [json.loads(x) for x in (DATA / "MATH-500/test.jsonl").read_text().splitlines()]
        return [{"id": f"math500-{r['unique_id']}", "q": r["problem"], "a": str(r["answer"])} for r in rows]
    raise ValueError(name)


def grade(output: str, gold: str) -> int:
    from math_verify import parse, verify
    try:
        g = parse(f"${gold}$") or parse(gold)
        return int(bool(verify(g, parse(output))))
    except Exception:                                            # noqa: BLE001  (math-verify raises on odd input)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-other", default="")
    ap.add_argument("--variants", nargs="+", required=True, help="scheme:role (role rl needs --base-other unless none)")
    ap.add_argument("--sets", nargs="+", default=["aime24", "aime25", "amc23", "math500"])
    ap.add_argument("--n", type=int, default=8, help="samples per problem (avg@n)")
    ap.add_argument("--n-set", nargs="*", default=[], help="per-set overrides of --n, e.g. aime24=16 math500=4")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    n_set = {k: int(v) for k, v in (x.split("=") for x in a.n_set)}
    from vllm import LLM, SamplingParams
    probs = [p for s in a.sets for p in load_set(s)]
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.max_tokens + 2048, gpu_memory_utilization=a.gpu_util,
              worker_extension_cls="mlsys.vllm_ext.VariantWorker", seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["q"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True) for p in probs]
    sps = [SamplingParams(n=n_set.get(p["id"].split("-")[0], a.n), temperature=0.6, top_p=0.95,
                          max_tokens=a.max_tokens, seed=0) for p in probs]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = {p["id"] for p in probs}
    for v in a.variants:
        scheme, role = v.split(":")
        f = out_dir / f"{scheme}_{role}.jsonl"
        if f.exists() and ids <= {json.loads(x)["id"] for x in f.read_text().splitlines()}:
            print(v, "already done:", f, flush=True)
            continue
        spec = {"scheme": scheme, "role": role}
        if a.base_other and scheme != "none":
            spec |= {"base_other": a.base_other, "rl_from": a.model}   # both read in full precision from disk
        t0 = time.time()
        info = llm.collective_rpc("apply_variant", args=(json.dumps(spec),))
        llm.reset_prefix_cache()
        outs = llm.generate(prompts, sps)
        rows, acc, ntok = [], {}, 0
        for p, o in zip(probs, outs):
            c = [grade(x.text, p["a"]) for x in o.outputs]
            lens = [len(x.token_ids) for x in o.outputs]
            ntok += sum(lens)
            rows.append({"variant": v, "model": a.model, "dtype": a.dtype, "id": p["id"], "correct": sum(c),
                         "n": len(c), "mean_len": sum(lens) / len(lens),
                         "truncated": sum(x.finish_reason == "length" for x in o.outputs)})
            acc.setdefault(p["id"].split("-")[0], []).append(sum(c) / len(c))
        tmp = f.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(f)
        dt = time.time() - t0
        print(v, json.dumps({k: round(sum(x) / len(x), 4) for k, x in acc.items()}), f"{dt:.0f} s",
              f"{ntok / dt:.0f} tok/s", json.dumps(info)[:200], flush=True)
    llm.collective_rpc("apply_variant", args=(json.dumps({"scheme": "none"}),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
