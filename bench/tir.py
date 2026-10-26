"""Tool-integrated reasoning (TIR) for math: the policy may write Python in ```python blocks; when it opens an ```output
block the engine stops, the last code block is executed in a sandbox (bench/std_bench.py's guard and limits: isolated
interpreter, temporary directory, 10 s CPU, 4 GB address space, destructive os/shutil/subprocess functions disabled) and
its printed output, truncated, is appended as the block's contents, then generation resumes; at most --rounds tool calls
and a total token budget per trajectory (the ToRL/SimpleTIR protocol). Tool-output tokens are context, not policy
tokens: the trainer masks them out of the loss and the trainer-engine mismatch.
    from tir import TIR_SUFFIX, rollout"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from std_bench import GUARD, _limits  # noqa: E402

TIR_SUFFIX = ("\nSolve the problem step by step. You may write Python code in a ```python code block to compute or check "
              "intermediate results; after the block, write ```output and stop, and the code's printed output will be "
              "shown to you. Use print() to show results. Put your final answer within \\boxed{}.")
STOP = "```output"
MAX_OUT_CHARS = 600


def run_code(src: str, timeout: int = 15) -> str:
    """Execute src in the sandbox; returns stdout (and the last stderr line on error), truncated."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"
        f.write_text(GUARD + src)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, timeout=timeout,
                               preexec_fn=_limits, stdin=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin"}, text=True)
            out = r.stdout
            if r.returncode != 0:
                err = [x for x in r.stderr.strip().splitlines() if x.strip()]
                out += ("\n" if out else "") + (err[-1] if err else f"exit code {r.returncode}")
        except subprocess.TimeoutExpired:
            out = "TimeoutError: the code ran too long"
    out = out.strip()
    if len(out) > MAX_OUT_CHARS:
        out = out[:MAX_OUT_CHARS] + "\n...(truncated)"
    return out


def last_code(text: str) -> str | None:
    m = re.findall(r"```python\s*\n(.*?)```", text, re.S)
    return m[-1] if m else None


class Traj:
    __slots__ = ("ids", "comp", "mask", "lps", "text", "calls", "done")

    def __init__(self, prompt_ids: list[int]):
        self.ids = list(prompt_ids)        # prompt + completion so far (the engine's next input)
        self.comp: list[int] = []          # completion tokens (policy and tool)
        self.mask: list[int] = []          # 1: policy token, 0: tool output
        self.lps: list[float] = []         # engine log-prob of each policy token (in order)
        self.text = ""
        self.calls = 0
        self.done = False


def rollout(llm, tok, prompt_ids: list[list[int]], make_sp, lora_request=None, rounds: int = 4, budget: int = 3072,
            workers: int = 16) -> list[Traj]:
    """One trajectory per prompt id list. make_sp(i, max_tokens, **stop_kwargs) -> SamplingParams for trajectory i (its own
    seed; it must pass the stop keyword arguments through and request logprobs=0). Returns the trajectories."""
    from concurrent.futures import ThreadPoolExecutor
    trajs = [Traj(p) for p in prompt_ids]
    pool = ThreadPoolExecutor(max_workers=workers)
    for _ in range(rounds + 1):
        act = [i for i, t in enumerate(trajs) if not t.done and len(t.comp) < budget]
        if not act:
            break
        sps = [make_sp(i, budget - len(trajs[i].comp), stop=[STOP], include_stop_str_in_output=True) for i in act]
        outs = llm.generate([{"prompt_token_ids": trajs[i].ids} for i in act], sps,
                            lora_request=[lora_request] * len(act) if lora_request is not None else None, use_tqdm=False)
        jobs = []
        for i, o in zip(act, outs):
            t, c = trajs[i], o.outputs[0]
            g = list(c.token_ids)
            t.ids += g
            t.comp += g
            t.mask += [1] * len(g)
            t.lps += [d[x].logprob for d, x in zip(c.logprobs, g)] if c.logprobs else [0.0] * len(g)
            t.text += c.text
            hit = c.finish_reason == "stop" and c.text.endswith(STOP)
            code = last_code(t.text) if hit else None
            if hit and code is not None and t.calls < rounds and len(t.comp) < budget - 16:
                jobs.append((i, pool.submit(run_code, code)))
            else:
                t.done = True
        for i, fut in jobs:
            t = trajs[i]
            res = "\n" + fut.result() + "\n```\n"
            r = tok.encode(res, add_special_tokens=False)
            t.ids += r
            t.comp += r
            t.mask += [0] * len(r)
            t.text += res
            t.calls += 1
    pool.shutdown()
    for t in trajs:
        t.done = True
    return trajs
