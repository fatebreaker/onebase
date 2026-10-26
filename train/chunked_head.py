"""Per-token log-probabilities from the final hidden states, the LM head applied in sequence chunks that are recomputed in the
backward pass, so a micro-batch's [length, vocabulary] logits never exist in full (RL-LONG-16K -CHUNK; mock review R2 v46, M4:
chunked log-probs remove the logits' memory, the gap the 16K comparison may rest on). Each position's value is the same computation
as token_logprobs(model(...).logits, ids): the head's bf16 product of the same final hidden state, upcast to fp32, log_softmax, the
sampled token's entry.
    from chunked_head import head_logprobs; lp = head_logprobs(model, ids, att)   # lp[:, t] = log p(ids[:, t+1] | ids[:, :t+1])"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def _lp(h: torch.Tensor, t: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    return torch.log_softmax(torch.nn.functional.linear(h, w, b).float(), -1).gather(-1, t.unsqueeze(-1)).squeeze(-1)


def head_logprobs(model, ids: torch.Tensor, att: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model   # a PeftModel's causal LM, LoRA layers in place
    h = base.model(input_ids=ids, attention_mask=att).last_hidden_state             # after the final norm, as the head sees it
    head = base.get_output_embeddings()
    T = h.shape[1] - 1
    return torch.cat([checkpoint(_lp, h[:, s:min(s + chunk, T)], ids[:, s + 1:min(s + chunk, T) + 1], head.weight, head.bias,
                                 use_reentrant=False) for s in range(0, T, chunk)], 1)
