"""RL-INTEGRATED: the trainer on the in-process engine's own FP8 tensors (one copy with the matched-precision engine).

lowbit_lora/shared_base.map_info maps CUDA IPC handles exported by an engine in another process. With the matched engine
(lowbit_lora/w16_engine.py) the engine runs in this process, so its worker hands over the tensors themselves
(mlsys.ipc_share_local.SharedBaseLocalWorker.export_base_local) and the trainer's modules point at them: map_info's mapping with
the unpacking step removed (shared_base.py itself unchanged).

    stats = map_local(model, llm.collective_rpc("export_base_local")[0], check_codes=True)   # model on the CPU, before PEFT
"""
from __future__ import annotations

import torch


@torch.no_grad()
def map_local(model: torch.nn.Module, info: dict, check_codes: bool = False, parts: str = "w") -> dict:
    """parts: "w" (the weight-matched trainer: the engine's FP8 weights, bf16 activations; every registered run) or "both" (a trainer
    that rounds activations alike, W8A8 through torch._scaled_mm, as matched_fp8.patch_model(parts="both"): RL-LONG -U's Unsloth-
    FP8-style arms, added 2026-10-24)."""
    from mlsys.ipc_share import _LAYER_MOD
    from mlsys.matched_fp8 import FP8_MAX, SKIP, _make_forward
    tensors = info["tensors"]
    dev = next(iter(tensors.values())).device
    fwd = _make_forward(parts == "both" and torch.cuda.get_device_capability(dev) >= (8, 9), parts)
    lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not SKIP.search(n)}
    mapped, mismatched = 0, []
    for n, m in lins.items():
        mm = _LAYER_MOD.search(n)
        key = f"{int(mm.group(1))}|{mm.group(2)}" if mm else None
        ent = info["modules"].get(key) if key else None
        if ent is None:
            raise KeyError(f"{n}: no engine tensor for {key}")
        codes = tensors[ent["tensor"]][ent["row0"]: ent["row0"] + ent["rows"]]
        if tuple(codes.shape) != tuple(m.weight.shape):
            raise ValueError(f"{n}: engine rows {tuple(codes.shape)} vs HF weight {tuple(m.weight.shape)}")
        scale = torch.tensor(info["scales"][ent["tensor"]], dtype=torch.float32, device=dev)
        if check_codes:      # the codes this process would have built (matched_fp8's rule) against the engine's
            own = torch.clamp(m.weight.detach().float() / scale.cpu(), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            ne = int((own.view(torch.uint8) != codes.cpu().view(torch.uint8)).sum())
            if ne:
                mismatched.append((n, ne, own.numel()))
        m.register_buffer("w_codes", codes, persistent=False)
        m.register_buffer("w_scale", scale, persistent=False)
        bias = m.bias
        m.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=dev), requires_grad=False)
        if bias is not None:
            m.bias = torch.nn.Parameter(bias.detach().to(dev), requires_grad=False)
        m.forward = fwd.__get__(m, type(m))
        mapped += 1
    shared_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    for key, getter in (("embed", lambda: model.get_input_embeddings()), ("lm_head", lambda: model.get_output_embeddings())):
        t = info.get(key)
        if t is None:
            continue
        mod = getter()
        v = mod.weight.shape[0]
        if t.shape[0] < v or t.shape[1] != mod.weight.shape[1]:
            raise ValueError(f"{key}: engine {tuple(t.shape)} vs HF {tuple(mod.weight.shape)}")
        if check_codes and not torch.equal(t[:v].cpu(), mod.weight.detach().to(torch.bfloat16)):
            mismatched.append((key, -1, t[:v].numel()))
        mod.weight = torch.nn.Parameter(t[:v], requires_grad=False)
        shared_bytes += t[:v].numel() * t.element_size()
    model.to(dev)                                                # the rest: norms, GDN conv/gates, biases (small)
    out = {"linear_layers": len(lins), "mapped": mapped, "engine_fp8_tensors": info["n_fp8"],
           "shared_gb": round(shared_bytes / 2**30, 2), "checked": check_codes, "in_process": True, "parts": parts}
    if check_codes:
        out["own_vs_engine"] = {"modules_differing": len(mismatched),
                                "entries_differing": sum(k for _, k, _ in mismatched if k > 0),
                                "examples": [(n, k) for n, k, _ in mismatched[:5]],
                                "embed_or_head_differ": [n for n, k, _ in mismatched if k < 0]}
    return out
