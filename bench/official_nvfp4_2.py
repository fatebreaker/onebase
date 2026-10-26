"""bench/official_nvfp4.py (kept unchanged for the streams that run it) plus --calib toolace (TOOL-RL): the GPTQ merge's
inputs are 128 of ToolACE's RL prompts in BFCL's format with their reference calls as the assistant turn (the adapter's
own task, as GSM8K is for the math adapters), truncated to 2,048 tokens. Otherwise official_nvfp4.py's doc:

W1-NVFP4-OFFICIAL (docs/REGISTRY.md, registered 2026-10-12 20:22 UTC): our NVFP4 conversion against NVIDIA's
official ModelOpt NVFP4 releases (nvidia/Llama-3.1-8B-Instruct-NVFP4, nvidia/Qwen3-8B-NVFP4).

  compare  part 1: for every quantized linear tensor of the release, its E2M1 codes, E4M3 block scales and float32
           tensor scale against our quantizer's (mlsys.quant.nvfp4_quant's rule, per full tensor) on the bf16
           original: the share of equal codes, of equal block scales, of equal entries (both), and the tensor-scale
           ratio; summaries per projection type and whether fused groups (q/k/v, gate/up) share one tensor scale.
  build    a bf16 checkpoint of the release dequantized (weights-only NVFP4: code x block scale x tensor scale), with
           --merge frozen: an adapter's update rounded onto the release's own grid (its block and tensor scales kept),
           --merge absmax: the merged weights re-exported, scales recomputed by the NVFP4 absmax rule (our quantizer).
           --merge srfrozen: as frozen, rounded stochastically onto the release's grid (one fixed seed; NVFP4 releases;
           SR-BEHAVIOR-REL, 2026-10-13).
           --merge gptq: the error-compensated merge onto the release's grid (mlsys gptq_merge.gptq_on_grid; NVFP4
           releases), H of each module's input from --n-cal --calib sequences through the bf16 original with the adapter.
           Tensors the release does not quantize (embeddings, norms, lm_head) are the release's own.
  score    one engine per served checkpoint, every adapter's runs in it (bench/official_fp8_logits.py's dump format);
           the logit gain: bench/official_fp8_logits.py report.
    python bench/official_nvfp4.py compare --release <dir> --orig <bf16 dir> --out x.json
    python bench/official_nvfp4.py build --release <dir> --orig <bf16 dir> --out <dir> [--merge frozen|absmax --adapter <dir>]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, "$ROUNDING")
from mlsys import quant  # noqa: E402

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLK = 16


class Shards:
    """name -> tensor across a safetensors checkpoint (single file or sharded with an index)."""

    def __init__(self, root: Path):
        from safetensors import safe_open
        idx = root / "model.safetensors.index.json"
        files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
        self.h = {f: safe_open(str(root / f), framework="pt") for f in files}
        self.where = {k: f for f, h in self.h.items() for k in h.keys()}

    def keys(self):
        return self.where.keys()

    def get(self, k: str) -> torch.Tensor:
        return self.h[self.where[k]].get_tensor(k)


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [out, in/2] -> E2M1 values [out, in] (float32): two codes per byte, the even element in the low nibble
    (ModelOpt's packing; vLLM's break_fp4_bytes reads it the same way)."""
    lo, hi = packed & 0x0F, packed >> 4
    nib = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1).long()
    mag = _E2M1.to(packed.device)[nib & 0x7]
    return torch.where((nib & 0x8) != 0, -mag, mag)


def release_layer(sh: Shards, mod: str, dev: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """codes [out, in], block scales [out, in/16] (float32 values of the E4M3 scales), tensor scale (0-d float32)."""
    codes = unpack(sh.get(f"{mod}.weight").to(dev))
    bs = sh.get(f"{mod}.weight_scale").to(dev).float()
    s2 = sh.get(f"{mod}.weight_scale_2").to(dev).float().reshape(())
    return codes, bs, s2


def ours(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mlsys.quant.nvfp4_quant's rule on a full tensor, returning codes, block scales and the tensor scale."""
    x = w.float()
    g = x.reshape(x.shape[0], -1, BLK)
    ts = (x.abs().max() / (quant.FP8_E4M3_MAX * quant.E2M1_MAX)).clamp_min(1e-30)
    block = g.abs().amax(dim=-1, keepdim=True) / (quant.E2M1_MAX * ts)
    block = block.clamp(max=quant.FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
    scale = (block * ts).clamp_min(1e-30)
    codes = quant._e2m1(g / scale, "nearest", None)
    return codes.reshape(x.shape), block.squeeze(-1), ts


def dequant(codes: torch.Tensor, bs: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    return codes * (bs * s2).repeat_interleave(BLK, dim=1)


def release_format(sh: Shards) -> str:
    """nvfp4: ModelOpt (weight_scale + weight_scale_2); fp8block: block FP8 (E4M3 codes, 128x128 weight_scale_inv)."""
    if any(k.endswith(".weight_scale_2") for k in sh.keys()):
        return "nvfp4"
    if any(k.endswith(".weight_scale_inv") for k in sh.keys()):
        return "fp8block"
    raise ValueError("unknown release format")


def quantized_modules(sh: Shards) -> list[str]:
    suffix = ".weight_scale_2" if release_format(sh) == "nvfp4" else ".weight_scale_inv"
    return sorted(k[: -len(suffix)] for k in sh.keys() if k.endswith(suffix))


# block FP8 (added 2026-10-13 14:30 UTC for W1-FP8-OFFICIAL-QWEN3, before its data): codes x the block's scale
FP8_MAX, FBLK = 448.0, 128


def fp8_layer(sh: Shards, mod: str, dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 codes as float32 [out, in] and the per-entry dequantization scale (128x128 blocks of weight_scale_inv)."""
    codes = sh.get(f"{mod}.weight").to(dev).float()
    sinv = sh.get(f"{mod}.weight_scale_inv").to(dev).float()
    n, k = codes.shape
    return codes, sinv.repeat_interleave(FBLK, 0)[:n].repeat_interleave(FBLK, 1)[:, :k]


def e4m3(y: torch.Tensor) -> torch.Tensor:
    return y.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def fp8_block_scale(w: torch.Tensor) -> torch.Tensor:
    """The absmax rule of a block-FP8 export: each 128x128 block's amax / 448, per entry."""
    n, k = w.shape
    pn, pk = (-n) % FBLK, (-k) % FBLK
    wp = torch.nn.functional.pad(w.abs(), (0, pk, 0, pn))
    am = wp.view((n + pn) // FBLK, FBLK, (k + pk) // FBLK, FBLK).amax(dim=(1, 3)) / FP8_MAX
    return am.clamp_min(1e-12).repeat_interleave(FBLK, 0)[:n].repeat_interleave(FBLK, 1)[:, :k]


def ptype(mod: str) -> str:
    return mod.rsplit(".", 1)[-1]


def compare(a) -> int:
    rel, orig = Shards(Path(a.release)), Shards(Path(a.orig))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mods = quantized_modules(rel)
    rows, s2_of = [], {}
    for m in mods:
        c_r, b_r, s_r = release_layer(rel, m, dev)
        c_o, b_o, s_o = ours(orig.get(f"{m}.weight").to(dev))
        s2_of[m] = float(s_r)
        same_b = (b_r == b_o)
        same_c = (c_r == c_o) | ((c_r == 0) & (c_o == 0))          # -0 and +0 are one value
        same_e = same_c & same_b.repeat_interleave(BLK, dim=1)
        rows.append({"module": m, "type": ptype(m), "n": c_r.numel(), "codes_equal": float(same_c.float().mean()),
                     "block_scales_equal": float(same_b.float().mean()), "entries_equal": float(same_e.float().mean()),
                     "tensor_scale_ratio": float(s_r / s_o)})
        del c_r, b_r, c_o, b_o
    shared = collections.Counter()
    for m in mods:
        if ptype(m) == "q_proj":
            p = m[: -len("q_proj")]
            shared["qkv"] += int(len({s2_of.get(p + t) for t in ("q_proj", "k_proj", "v_proj")}) == 1)
        if ptype(m) == "gate_proj":
            p = m[: -len("gate_proj")]
            shared["gate_up"] += int(len({s2_of.get(p + t) for t in ("gate_proj", "up_proj")}) == 1)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    summary = {}
    for t, rs in sorted(by.items()):
        n = sum(r["n"] for r in rs)
        summary[t] = {"tensors": len(rs), **{k: sum(r[k] * r["n"] for r in rs) / n
                                             for k in ("codes_equal", "block_scales_equal", "entries_equal")},
                      "tensor_scale_ratio_median": statistics.median(r["tensor_scale_ratio"] for r in rs),
                      "tensor_scale_ratio_range": [min(r["tensor_scale_ratio"] for r in rs),
                                                   max(r["tensor_scale_ratio"] for r in rs)]}
    n = sum(r["n"] for r in rows)
    out = {"release": a.release, "orig": a.orig, "quantized_tensors": len(rows),
           "entries_equal": sum(r["entries_equal"] * r["n"] for r in rows) / n,
           "codes_equal": sum(r["codes_equal"] * r["n"] for r in rows) / n,
           "layers": sum(1 for m in mods if ptype(m) == "q_proj"), "shared_tensor_scale": dict(shared),
           "by_type": summary, "tensors": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "tensors"}, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def adapter_updates(adapter: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    from safetensors import safe_open
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    assert not cfg.get("use_rslora") and not cfg.get("use_dora") and not cfg.get("rank_pattern") \
        and not cfg.get("alpha_pattern"), "plain LoRA only"
    sc = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(str(adapter / "adapter_model.safetensors"), framework="pt")
    pat = re.compile(r"(model\.layers\.\d+\..+)\.lora_A\.weight$")
    out = {}
    for k in f.keys():
        m = pat.search(k)
        if m:
            out[m.group(1)] = (f.get_tensor(k), f.get_tensor(k.replace("lora_A", "lora_B")), sc)
    return out


def calib_pairs(calib: str) -> list:
    """The calibration sequences of calib_hessians: (prompt or messages, answer | None | "messages")."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import std_bench
    dl = Path("$SCRATCH/data/std_bench_dl")
    if calib == "gsm8k":
        df = pd.read_parquet(dl / "gsm8k_train.parquet")
        pairs = [(q + std_bench.SUFFIX, ans) for q, ans in zip(df.question, df.answer)]
    elif calib == "magicoder":
        rows = [json.loads(x) for x, _ in zip(open(dl / "magicoder_oss_75k.jsonl"), range(5000))]
        pairs = [(r["problem"], r["solution"]) for r in rows]
    elif calib == "toolace":          # TOOL-RL: ToolACE prompts in BFCL's format + reference calls (bench/bfcl_tools.py)
        rows = [json.loads(x) for x in (dl.parent / "toolace_dl/toolace_rl.jsonl").read_text().splitlines()]
        ref = lambda g: "[" + ", ".join(f"{n}(" + ", ".join(f"{k}={v!r}" for k, v in a.items()) + ")" for d in g for n, a in d.items()) + "]"  # noqa: E731
        pairs = [(r["messages"] + [{"role": "assistant", "content": ref(r["gold"])}], "messages") for r in rows]
    elif calib.startswith("task:"):   # the agent's own training tasks (user turn only; GPTQ-REL-TASK)
        bench = Path(calib.split(":", 1)[1])
        pairs = [(json.loads((d / "eval_config.json").read_text())["task"], None)
                 for d in sorted((bench / "problems").iterdir())]
    else:   # trans:<transcript dir>@<agent_harness benchmark dir>: the agent's own transcripts (GPTQ-REL-TRANS)
        tdir, bench = (Path(x) for x in calib.split(":", 1)[1].split("@"))
        pairs = []
        for f in sorted(tdir.glob("*.json")):
            d = json.loads(f.read_text())
            if "transcript" not in d:
                continue
            task = json.loads((bench / "problems" / d["id"] / "eval_config.json").read_text())["task"]
            msgs = [{"role": "user", "content": task}]
            for turn in d["transcript"]:
                msgs.append({"role": "assistant" if turn.get("role") == "assistant" else "user",
                             "content": str(turn.get("content", ""))})
            pairs.append((msgs, "messages"))
    return pairs


SHARE = {"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"}   # leaves that share their input's H


def _calib_model(orig_dir: Path, adapter: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig_dir))
    model = AutoModelForCausalLM.from_pretrained(str(orig_dir), dtype=torch.bfloat16, device_map={"": 0})
    return tok, PeftModel.from_pretrained(model, str(adapter)).eval()


def _calib_pass(model, tok, pairs, n_cal: int, dev: str, layers=None) -> dict[str, torch.Tensor]:
    """One pass over the calibration sequences, accumulating X^T X for the adapted modules (of the given layers only)."""
    import random
    idx = list(range(len(pairs)))
    random.Random(0).shuffle(idx)
    H, hooks = {}, []
    for name, mod in model.named_modules():
        m = re.search(r"(model\.layers\.(\d+)\.(self_attn|mlp)\.(q|o|gate|down)_proj)$", name)
        if m and hasattr(mod, "base_layer") and (layers is None or int(m.group(2)) in layers):
            def hook(_m, inp, _o, key=m.group(1)):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                H[key] = H.get(key, 0) + x.T @ x
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for i in idx[:n_cal]:
            if pairs[i][1] == "messages":   # a whole transcript
                t = tok.apply_chat_template(pairs[i][0], tokenize=False)
                model(input_ids=tok(t, return_tensors="pt").input_ids[:, :2048].to(dev))
                continue
            msgs = [{"role": "user", "content": pairs[i][0]}]
            if pairs[i][1] is not None:
                msgs.append({"role": "assistant", "content": pairs[i][1]})
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=pairs[i][1] is None)
            model(input_ids=tok(t, return_tensors="pt").input_ids[:, :1024].to(dev))
    for h in hooks:
        h.remove()
    out = dict(H)
    for k in list(H):
        for leaf, src in SHARE.items():
            if k.endswith(src):
                out[k[: -len(src)] + leaf] = H[k]
    return out


def calib_hessians(orig_dir: Path, adapter: Path, calib: str, n_cal: int, dev: str) -> dict[str, torch.Tensor]:
    """Second moment X^T X of every adapted module's input on n_cal chat sequences (GSM8K or Magicoder-OSS
    question+answer pairs, seeded shuffle) through the bf16 original with the exact adapter (PEFT); q/k/v share one H,
    gate/up another. Keys are module names, e.g. model.layers.3.self_attn.q_proj. All layers at once (models whose
    Hessians fit next to the model; ChunkedHessians for larger ones)."""
    tok, model = _calib_model(orig_dir, adapter)
    H = _calib_pass(model, tok, calib_pairs(calib), n_cal, dev)
    del model
    torch.cuda.empty_cache()
    return H


class ChunkedHessians:
    """calib_hessians for models whose Hessians do not fit at once (Qwen3.8-27B: ~88 GB of fp32 X^T X beside 54 GB of
    weights): the calibration model stays on the GPU and the Hessians are computed for `chunk` decoder layers at a time,
    in the order the build visits them (`order`: module names), one calibration pass per chunk; the same sequences, so
    every H equals calib_hessians' (up to summation order). Use as a mapping: H[module name]."""

    def __init__(self, orig_dir: Path, adapter: Path, calib: str, n_cal: int, dev: str, order: list[str], chunk: int = 8):
        self.tok, self.model = _calib_model(orig_dir, adapter)
        self.pairs, self.n_cal, self.dev = calib_pairs(calib), n_cal, dev
        layers = []
        for m in order:
            mm = re.search(r"model\.layers\.(\d+)\.", m)
            if mm and int(mm.group(1)) not in layers:
                layers.append(int(mm.group(1)))
        self.chunks = [set(layers[i:i + chunk]) for i in range(0, len(layers), chunk)]
        self.cur, self.H = None, {}

    def __getitem__(self, mod: str) -> torch.Tensor:
        layer = int(re.search(r"model\.layers\.(\d+)\.", mod).group(1))
        c = next(i for i, ch in enumerate(self.chunks) if layer in ch)
        if c != self.cur:
            self.H = {}
            torch.cuda.empty_cache()
            self.H = _calib_pass(self.model, self.tok, self.pairs, self.n_cal, self.dev, layers=self.chunks[c])
            self.cur = c
            print(f"hessians: chunk {c + 1}/{len(self.chunks)} (layers {sorted(self.chunks[c])})", flush=True)
        return self.H[mod]


def build(a) -> int:
    from safetensors.torch import save_file
    rel_dir, orig_dir, out = Path(a.release), Path(a.orig), Path(a.out)
    rel, orig = Shards(rel_dir), Shards(orig_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out.mkdir(parents=True, exist_ok=True)
    for f in orig_dir.iterdir():   # config (bf16, no quantization_config), tokenizer, generation config: the original's
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    fmt = release_format(rel)
    mods = set(quantized_modules(rel))
    upd = adapter_updates(Path(a.adapter)) if a.merge != "none" else {}
    if a.merge == "srfrozen" and fmt != "nvfp4":
        raise SystemExit(f"--merge {a.merge} is implemented for NVFP4 releases only")
    Hs = calib_hessians(orig_dir, Path(a.adapter), a.calib, a.n_cal, dev) if a.merge == "gptq" else {}
    gen = torch.Generator(device=dev).manual_seed(0)   # srfrozen: one fixed stream of draws over the modules in order
    stats = {"merged": 0, "bf16_merged": 0, "num": 0.0, "den": 0.0, "moved": 0, "n": 0}
    shard, size, weight_map, part = {}, 0, {}, 0

    def flush():
        nonlocal shard, size, part
        if shard:
            part += 1
            name = f"model-{part:05d}.safetensors"
            save_file(shard, str(out / name), metadata={"format": "pt"})
            weight_map.update({k: name for k in shard})
            shard, size = {}, 0

    for k in sorted(orig.keys()):
        mod = k[: -len(".weight")] if k.endswith(".weight") else None
        if mod in mods:
            if fmt == "nvfp4":
                codes, bs, s2 = release_layer(rel, mod, dev)
                w = dequant(codes, bs, s2)
            else:
                codes, S8 = fp8_layer(rel, mod, dev)
                w = codes * S8
            if mod in upd:
                A, B, sc = upd[mod]
                dW = sc * (B.to(dev).float() @ A.to(dev).float())
                if fmt == "fp8block" and a.merge == "gptq":       # error-compensated onto the release's block-FP8 grid
                    from mlsys import gptq_merge
                    q = gptq_merge.gptq_on_grid(w + dW, Hs[mod], S8[:, ::FBLK].clamp_min(1e-30), FBLK,
                                                quant._e4m3_levels())
                    new = torch.where(S8 > 0, q, w)
                elif fmt == "fp8block" and a.merge == "frozen":     # the release's block scales kept
                    new = torch.where(S8 > 0, e4m3((w + dW) / S8.clamp_min(1e-30)) * S8, w)
                elif fmt == "fp8block":                            # re-export: block scales from the merged weights
                    S2 = fp8_block_scale(w + dW)
                    new = e4m3((w + dW) / S2) * S2
                elif a.merge == "gptq":                     # error-compensated, onto the release's grid
                    from mlsys import gptq_merge
                    Sb = bs * s2
                    lv = torch.cat([-quant._E2M1_POS.flip(0)[:-1], quant._E2M1_POS])
                    q = gptq_merge.gptq_on_grid(w + dW, Hs[mod], Sb.clamp_min(1e-30), BLK, lv)
                    new = torch.where(Sb.repeat_interleave(BLK, dim=1) > 0, q, w)
                elif a.merge in ("frozen", "srfrozen"):     # the release's grid; an all-zero block cannot move
                    S = (bs * s2).repeat_interleave(BLK, dim=1)
                    y = ((w + dW) / S.clamp_min(1e-30)).clamp(-quant.E2M1_MAX, quant.E2M1_MAX)
                    q = (quant._e2m1(y, "stochastic", gen) if a.merge == "srfrozen" else quant._e2m1(y, "nearest", None))
                    new = torch.where(S > 0, q * S, w)
                else:
                    new = quant.nvfp4_quant(w + dW)
                D = new - w
                stats["merged"] += 1
                stats["num"] += float((D.double() * dW.double()).sum())
                stats["den"] += float((dW.double() ** 2).sum())
                stats["moved"] += int((D != 0).sum())
                stats["n"] += D.numel()
                w = new
            t = w.to(torch.bfloat16).cpu()
        else:
            t = rel.get(k) if k in rel.keys() else orig.get(k)
            if t.dtype != torch.bfloat16 and t.is_floating_point():
                t = t.to(torch.bfloat16)
            if mod in upd:     # a module the release keeps in bf16: a bf16 merge there
                A, B, sc = upd[mod]
                t = (t.float() + sc * (B.float() @ A.float())).to(torch.bfloat16)
                stats["bf16_merged"] += 1
        shard[k] = t.contiguous()
        size += t.numel() * t.element_size()
        if size >= 4 << 30:
            flush()
    flush()
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    res = {"out": str(out), "format": fmt, "merge": a.merge, "adapter": a.adapter, "quantized_modules": len(mods),
           "merged_modules": stats["merged"], "bf16_merged_modules": stats["bf16_merged"], "tensors": len(weight_map)}
    if stats["den"]:
        res |= {"g_W": stats["num"] / stats["den"], "moved_share": stats["moved"] / stats["n"]}
    print(json.dumps(res))
    return 0


def score(a) -> int:
    """One vLLM engine for one served checkpoint; for each item of the --spec JSON list ({"results", "template",
    "max_tokens", "out", "runs": {name: adapter dir or null}}), the item's transcripts (selected as mlsys.fidelity
    does) are scored under each run and written to <out>_<name>.json in bench/official_fp8_logits.py's format."""
    import random
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from mlsys.fidelity import _steps, build as build_seq, realised
    items = json.loads(Path(a.spec).read_text())
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    llm = LLM(model=a.model, tokenizer=a.tokenizer or a.model, enable_lora=True, max_lora_rank=a.max_lora_rank,
              max_loras=1, max_model_len=max(it["max_tokens"] for it in items) + 8, gpu_memory_utilization=a.gpu_util,
              max_num_batched_tokens=2048, enable_chunked_prefill=True, max_num_seqs=16)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1)
    lid = 0
    for it in items:
        files = sorted(p for p in Path(it["results"]).glob("*.json") if p.name not in ("summary.json", "harness_v2.json"))
        random.Random(0).shuffle(files)
        seqs, masks = [], []
        for f in files:
            rec = json.loads(f.read_text())
            if rec.get("transcript"):
                ids, mask = build_seq(tok, rec, it["max_tokens"], it["template"])
                if sum(mask) >= 32:
                    seqs.append(ids)
                    masks.append(mask)
            if len(seqs) >= it.get("n", 24):
                break
        for name, ad in it["runs"].items():
            kw = {}
            if ad:
                lid += 1
                kw["lora_request"] = LoRARequest(f"a{lid}", lid, ad)
            outs = llm.generate([{"prompt_token_ids": s} for s in seqs], sp, use_tqdm=False, **kw)
            lp = [realised(_steps(o), s, m) for o, s, m in zip(outs, seqs, masks)]
            Path(f"{it['out']}_{name}.json").write_text(json.dumps({"model": a.model, "adapter": ad, "seqs": seqs,
                                                                    "masks": masks, "lp": lp}))
            print(json.dumps({"model": Path(a.model).name, "item": Path(it["out"]).name, "run": name,
                              "seqs": len(seqs), "tokens": sum(map(sum, masks))}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    ps = sp.add_parser("score")
    ps.add_argument("--model", required=True)
    ps.add_argument("--tokenizer", default="")
    ps.add_argument("--spec", required=True)
    ps.add_argument("--max-lora-rank", type=int, default=128)
    ps.add_argument("--gpu-util", type=float, default=0.8)
    pc = sp.add_parser("compare")
    pc.add_argument("--release", required=True)
    pc.add_argument("--orig", required=True)
    pc.add_argument("--out", required=True)
    pb = sp.add_parser("build")
    pb.add_argument("--release", required=True)
    pb.add_argument("--orig", required=True)
    pb.add_argument("--out", required=True)
    pb.add_argument("--merge", default="none", choices=("none", "frozen", "absmax", "srfrozen", "gptq"))
    pb.add_argument("--calib", default="gsm8k", help="--merge gptq: gsm8k | magicoder | toolace | task:<agent_harness benchmark dir>")
    pb.add_argument("--n-cal", type=int, default=128, help="--merge gptq: calibration sequences")
    pb.add_argument("--adapter", default="")
    a = ap.parse_args()
    return {"compare": compare, "build": build, "score": score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
