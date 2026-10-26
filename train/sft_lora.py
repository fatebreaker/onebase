"""LoRA SFT on a public math or code corpus, for the standard-benchmark tests (STD-BENCH): the adapters whose gains on GSM8K,
MATH-500, HumanEval and MBPP are then measured under every serving variant (bf16, exact adapter on a low-precision base,
FP8/NVFP4-stored adapter, nearest and stochastic merges onto the base's grid).

Data (downloaded files: parsed as data only):
  metamath   MetaMathQA-40K (query, response) -> one user/assistant turn
  magicoder  Magicoder-OSS-Instruct-75K, Python only (problem, solution) -> one user/assistant turn
Loss on the assistant turn only (TRL's conversational prompt/completion format), LoRA on q/k/v/o/gate/up/down.

    venv_vllm/bin/python train/sft_lora.py --model <dir> --data <file> --format metamath --out <dir> [--limit N]
--save-steps N (for preemptible jobs) checkpoints every N steps into <out>/checkpoint-* and resumes from the last one.
<out>/.lock (refreshed at every log line) keeps two runs off one output: a run that finds a fresh lock waits for the
other's adapter and exits; a lock older than 10 minutes is taken over (that run died; resume from its checkpoint).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
from pathlib import Path

STALE = 600


def take_lock(out: Path) -> bool:
    """True when this run owns <out>/.lock; False when another live run trained (or is training) <out>."""
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".lock"
    while True:
        if (out / "adapter_model.safetensors").exists():
            return False
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time():.0f}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > STALE:
                print(f"taking over a stale lock ({age:.0f} s old): {lock.read_text().strip()}", flush=True)
                lock.unlink(missing_ok=True)
                continue
            print(f"{out} is being trained elsewhere ({lock.read_text().strip()}); waiting", flush=True)
            time.sleep(60)


def load(fmt: str, path: str, limit: int, seed: int) -> list[dict]:
    rows = []
    if fmt == "metamath":
        for r in json.loads(Path(path).read_text()):
            rows.append({"prompt": [{"role": "user", "content": str(r["query"])}],
                         "completion": [{"role": "assistant", "content": str(r["response"])}]})
    elif fmt == "magicoder":
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("lang") != "python":
                    continue
                rows.append({"prompt": [{"role": "user", "content": str(r["problem"])}],
                             "completion": [{"role": "assistant", "content": str(r["solution"])}]})
    else:
        raise SystemExit(f"unknown format {fmt}")
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--format", required=True, choices=("metamath", "magicoder"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if not take_lock(out):
        print(f"{out} already trained", flush=True)
        return 0
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class Heartbeat(TrainerCallback):
        def on_log(self, args, state, control, **kw):
            (out / ".lock").touch()
    rows = load(a.format, a.data, a.limit, a.seed)
    print(f"{len(rows)} examples from {a.data} ({a.format})", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
                    warmup_steps=0.03, per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
                    max_length=a.max_len, bf16=True, logging_steps=20, report_to=[], seed=a.seed,
                    save_strategy="steps" if a.save_steps else "no", save_steps=a.save_steps or 500, save_total_limit=1,
                    gradient_checkpointing=True, completion_only_loss=True,
                    model_init_kwargs={"torch_dtype": torch.bfloat16})
    lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    tr = SFTTrainer(model=a.model, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok,
                    peft_config=lora, callbacks=[Heartbeat()])
    t0 = time.time()
    resume = a.save_steps > 0 and any(Path(a.out).glob("checkpoint-*"))
    print(f"resuming from the last checkpoint in {a.out}" if resume else "training from the start", flush=True)
    tr.train(resume_from_checkpoint=True if resume else None)
    tr.model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    (Path(a.out) / "sft_meta.json").write_text(json.dumps(
        {"model": a.model, "data": a.data, "format": a.format, "examples": len(rows), "lora_r": a.lora_r, "lr": a.lr,
         "epochs": a.epochs, "max_len": a.max_len, "batch": a.batch * a.accum, "seed": a.seed, "resumed": resume,
         "train_s": round(time.time() - t0), "final_loss": tr.state.log_history[-1] if tr.state.log_history else None},
        indent=1, default=str))
    (out / ".lock").unlink(missing_ok=True)
    print(f"saved {a.out} after {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
