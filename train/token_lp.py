"""Per-token log-probabilities for the GRPO trainers, in fp32 as before (log_softmax of the logits upcast to fp32, the
sampled token's entry), computed in sequence chunks that are recomputed in the backward pass, so that the fp32 copies of
a micro-batch's [length, vocabulary] logits are never all alive at once. Each position's value is the same computation
as the unchunked one. Added 10-15 after RL-FP8-QMATH S0G seed 0 ran out of memory at step 25 (micro 2, Qwen3.5's
248k-entry vocabulary, 5 GiB fp32 logit copies).
    from token_lp import token_logprobs; lp = token_logprobs(logits, ids)   # lp[:, t] = log p(ids[:, t+1] | ids[:, :t+1])"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def _lp(lg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(lg.float(), -1).gather(-1, t.unsqueeze(-1)).squeeze(-1)


def token_logprobs(logits: torch.Tensor, ids: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    T = logits.shape[1] - 1
    return torch.cat([checkpoint(_lp, logits[:, s:min(s + chunk, T)], ids[:, s + 1:min(s + chunk, T) + 1], use_reentrant=False)
                      for s in range(0, T, chunk)], 1)


def keep_old_log(out) -> None:
    """A fresh start in a directory that holds an earlier attempt's log.jsonl (an attempt that failed before its first
    checkpoint): keep that log as log_attempt<k>.jsonl instead of appending to it."""
    f = out / "log.jsonl"
    if f.exists() and f.stat().st_size:
        k = 1
        while (out / f"log_attempt{k}.jsonl").exists():
            k += 1
        f.rename(out / f"log_attempt{k}.jsonl")
