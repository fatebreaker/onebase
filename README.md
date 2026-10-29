# OneBase

**Multi-tenant LoRA kernels on low-precision bases, as a plugin for vLLM, and a LoRA RL trainer on the same base.**

Low-rank adapters (LoRA) specialize a large language model by adding a small trainable update, *BA*, to each of its weight
matrices. A serving engine can then keep one copy of the base model and batch the requests of many *tenants*, users or
applications that each bring their own adapter. As base models move to 8-bit (FP8) and 4-bit (NVFP4) formats, the base
model's matrix multiplications (GEMMs) get cheaper, but the adapters do not: engines compute them in separate GPU kernels
beside the GEMM, and while generating one token at a time (decoding) the fixed latency of those small kernels becomes a
growing share of every step.

OneBase computes every tenant's adapter together with the low-precision base GEMM:

- **The decode path (Hopper and Blackwell GPUs).** A GEMM needs an adapter's down-projection (*h = xA*ᵀ) only in its
  *epilogue*, the final stage in which it writes its output. OneBase therefore launches the down-projection first and the
  low-precision GEMM right after it as a dependent kernel (CUDA's programmatic dependent launch, PDL). The GEMM streams its
  weights while the down-projection runs and adds each tenant's up-projection (*hB*ᵀ) in its epilogue, so the output is
  written once.
- **Fused kernels (Ada GPUs such as the L40S).** Two kernels per projection instead of vLLM's four: K1 quantizes the
  activations exactly as vLLM does and computes every adapter's down-projection; K2 runs the FP8 or NVFP4 GEMM over tokens
  sorted by adapter and adds each adapter's up-projection in its epilogue.
- **Timed dispatch.** When the model loads, the plugin times its candidate paths for each projection shape and batch size
  and keeps the fastest: the fused kernels, the vendor GEMM followed by OneBase's adapter kernels, or the vendor GEMM
  followed by vLLM's own adapter kernels. Anything it cannot serve falls back to vLLM.
- **Low-precision adapter storage.** Adapters can be stored in FP8 or NVFP4 and are dequantized in registers, which
  frees memory for the KV cache when many tenants are resident.
- **LoRA RL on the same base.** A GRPO trainer that computes on the engine's FP8 weights, so it needs no bf16 copy of the
  model; optionally one weight copy shared with the engine through CUDA IPC, the engine's KV cache released while the
  trainer runs, and a *matched* engine (8-bit weights, 16-bit activations, W8A16) that removes most of the mismatch between
  the model that generates training data and the model being trained.

OneBase quantizes the base exactly as vLLM does (FP8 and NVFP4 codes and scales are bit-identical), works with vLLM's
CUDA graphs and with tensor parallelism, and needs no changes to vLLM's source. In our measurements with vLLM 0.28, one
adapter at batch size 1 takes 0.76–0.88× the time per output token of vLLM's fastest configuration for 8B–9B models on
H100, H200 and GH200 GPUs, and with 32 tenants OneBase delivers 1.05–1.09× vLLM's throughput on most GPUs.

## Requirements

| | tested |
|---|---|
| GPU | NVIDIA Ada (sm_89: L40S), Hopper (sm_90: H100, H200, GH200), Blackwell (sm_120: RTX PRO 6000). The decode path needs Hopper or Blackwell. |
| Software | Python 3.12, vLLM 0.28.0, PyTorch 2.13.0, Triton 3.7.1, FlashInfer 0.6.16, Transformers 5.16.1 |
| RL trainer, additionally | PEFT 0.20.0, flash-linear-attention 0.5.2 (for Qwen3.5's linear-attention layers), math-verify, pandas |

## Installation

From the repository's root directory, install into the Python environment that runs vLLM:

```bash
pip install -e . --no-deps
```

This registers the plugin as a `vllm.general_plugins` entry point, so every vLLM process loads it. It stays inactive
unless `LOWBIT_FUSED=1` is set. The Python package is named `lowbit_lora`.

## Serving with OneBase

Serve an FP8 base (`--quantization fp8`, or an FP8/NVFP4 checkpoint) with LoRA enabled, as you would with plain vLLM:

```bash
export LOWBIT_FUSED=1                                  # OneBase's LoRA layers
# Hopper / Blackwell: add the decode path
export LOWBIT_PDL=1 PYTHONPATH=$PWD/pylib_pdl:$PYTHONPATH    # pylib_pdl4 on Blackwell with NVFP4 bases
export LOWBIT_PDL_CFG=$PWD/results/pdl_k2.jsonl        # tuned tiles: pdl_k2.jsonl (H100), pdl_k2_h200.jsonl (H200)

vllm serve Qwen/Qwen3-8B --quantization fp8 \
    --enable-lora --max-loras 32 --max-lora-rank 64 \
    --lora-modules tenant1=/path/to/adapter1 tenant2=/path/to/adapter2
```

Requests then select an adapter by name (`"model": "tenant1"`), exactly as with vLLM. The same switches work for offline
use (`vllm.LLM(..., enable_lora=True)`). Loading takes longer than plain vLLM (about 20 s instead of 5 s for an 8B model)
because the plugin times its paths before vLLM captures its CUDA graphs.

Main switches (all off unless stated):

| variable | effect |
|---|---|
| `LOWBIT_FUSED=1` | enable OneBase for LoRA layers on FP8 bases and NVFP4 (ModelOpt) checkpoints |
| `LOWBIT_PDL=1` | Hopper/Blackwell: the decode path, for decode batches of up to `LOWBIT_PDL_MAX_M` tokens (default 32); needs `pylib_pdl` (or `pylib_pdl4`) on `PYTHONPATH` |
| `LOWBIT_PDL_CFG=<file>` | tuned tile table for the decode path (`bench/pdl_k2_bench.py` writes one for a new GPU or model) |
| `LOWBIT_LORA_FP8=1`, `LOWBIT_LORA_FP4=1` | store adapters in FP8 or NVFP4 instead of bf16 |
| `LOWBIT_RELEASE_SLOTS` | (default 1) with a stored format, replace vLLM's bf16 adapter slots so the freed memory goes to the KV cache |
| `LOWBIT_LEAN_BASE=1`, `LOWBIT_FAST_LAUNCH=1` | lower host overhead for steps that run outside CUDA graphs; recommended at vLLM's default CUDA-graph sizes |
| `LOWBIT_TP=1` | also serve tensor-parallel shards and layers with a bias |
| `LOWBIT_DISPATCH` | `auto` (default: timed dispatch), or `fused` / `hybrid` to force one path |
| `LOWBIT_REPORT=<file>` | write one line per layer: the chosen path and the dispatch table |
| `LOWBIT_WARMUP=0` | skip the kernel precompilation at load (not for serving) |

Further opt-in variants used in our experiments: `LOWBIT_STOCK_ABOVE=<M>` and `LOWBIT_NOHYB_ABOVE=<M>` (restrict the
dispatch above a batch size), `LOWBIT_PDL_FORCE=1` (always take the decode path where it is a candidate), `LOWBIT_SWAP=1`
with `pylib_swap` (a swap-AB GEMM path on Hopper), and `LOWBIT_W16_ENGINE=1` with `pylib_w16` (the matched W8A16 engine in
`vllm serve`).

## LoRA RL on the same base

The trainers in `train/` run GRPO with the vLLM rollout engine and the trainer on one GPU. `train/grpo_q*.py` are
successive versions; each one's docstring states what it adds to its predecessor, and earlier versions are kept so that
every experiment's exact script remains available. The main entry points:

| script | adds |
|---|---|
| `train/grpo_q13.py` | FP8 rollout engine, the FP8 trainer on the engine's weights (`--engine fp8 --match fp8w --fp8-train w16`), truncated importance sampling (`--tis C`), the matched W8A16 engine (`--w16-engine`), score centering (`--center K`), per-step mismatch diagnostics (`--diag-topk K`) |
| `train/grpo_q14.py`, `grpo_q15.py` | one FP8 weight copy shared by the engine and the trainer (`--shared-base`), also with the matched engine |
| `train/grpo_q16.py` | the full system: matched engine, one shared copy, and the engine's KV cache released while the trainer runs (`--kv-sleep`) |
| `train/grpo_q7.py`, `grpo_q19.py`, `grpo_q20.py` | baselines: the trainer's bf16 base offloaded during rollouts (`--offload-base`), chunked log-probabilities (`--chunked-head`) |

Example, the matched engine without importance correction on Qwen3.5-9B with competition math (DAPO-Math):

```bash
export PYTHONPATH=$PWD:$PWD/train:$PYTHONPATH
export VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the engine runs in-process
export LOWBIT_SWAP_CFG=$PWD/results/w16_swap.jsonl     # Hopper: tiles of the swap-AB W8A16 GEMM (with --w16-swap)
python train/grpo_q13.py --model /path/to/Qwen3.5-9B --out runs/matched_s0 --seed 0 --steps 200 \
    --task dapo --no-think --graphs --lora-r 64 --lr 2e-5 --max-new 2048 --max-model-len 4096 --micro 2 \
    --engine fp8 --match fp8w --fp8-train w16 --w16-engine --w16-swap --diag-topk 32
```

For FP8 (W8A8) rollouts corrected with truncated importance sampling instead, replace `--w16-engine --w16-swap` by
`--tis 2`, and set `LOWBIT_FUSED=1` to serve the rollouts through OneBase's kernels.

Each step logs one JSON line (reward, tokens, step and generation time, and the engine–trainer mismatch). The trainers
read their datasets from a local directory (`DL` near the top of each script; DAPO-Math-17k is read as a parquet file).

## Benchmarks and tests

- `bench/` holds the benchmarks behind our measurements: kernel benchmarks (`kernel_bench.py`, `multi_bench.py`,
  `pdl_k2_bench.py`, `kernel_controls.py`), end-to-end latency and throughput (`overhead.py`, `overhead_tp.py`),
  online serving (`serve_bench.py`, `serve_bench_t.py` with a WildChat trace from `make_wildchat_trace.py`), profiles
  (`prefill_profile.py`, `host_overhead.py`), the mismatch decomposition (`mismatch_bench3.py`) and the scorers
  (`score_*.py`) that turn their JSON lines into the reported numbers.
- `tests/` holds correctness checks. The `*_interp.py` tests run the kernels in Triton's CPU interpreter against fp32
  references, so they need no GPU:

  ```bash
  TRITON_INTERPRET=1 PYTHONPATH=. python tests/test_fp8_lora_vllm_interp.py
  ```

  `tests/compile_check.py` compiles every kernel offline for sm_89, sm_90 and sm_120; the `gpu_*_check.py` and
  `*_plugin_check.py` scripts check the plugin against vLLM's LoRA path on a GPU.
- `tools/make_nvfp4_ckpt.py` builds a ModelOpt-format NVFP4 checkpoint from a local bf16 model.

Paths inside `bench/` and `train/` that pointed to our cluster's storage are replaced by placeholders such as
`$SCRATCH/...` (models, datasets, adapters) and `$ONEBASE` (this repository). Python does not expand them: pass the
corresponding command-line arguments (most scripts take `--model`, `--adapter` and `--out`) or edit the constants.

## Repository layout

| path | contents |
|---|---|
| `lowbit_lora/kernels/` | Triton kernels: `fp8_lora_vllm.py` (K1 and K2 on FP8 bases, vLLM's adapter buffers and batching metadata), `fp8_lora_pdl.py` (the decode path), `nvfp4_lora_vllm.py`, `nvfp4_lora_pdl.py` (NVFP4 bases), `adapter_store.py` (bf16/FP8/NVFP4 adapter storage), `fp8_train_kernels.py`, `w8a16_lora_multi.py` (the trainer's and the matched engine's GEMMs) |
| `lowbit_lora/plugin.py` | the vLLM plugin: a registered custom operator captured in vLLM's CUDA graphs, timed dispatch, slot memory, tensor parallelism |
| `lowbit_lora/hopper_pdl.py`, `blackwell_pdl_plugin.py` | the decode path as a dispatch candidate on Hopper and Blackwell |
| `lowbit_lora/w16_engine*.py`, `fp8_train.py`, `shared_base*.py` | the matched engine, the FP8 trainer, and one weight copy shared with the engine |
| `mlsys/` | helpers the trainers use: FP8 weight matching (`matched_fp8.py`), vLLM worker extensions for adapter syncs (`vllm_ext.py`), CUDA IPC sharing (`ipc_share*.py`), quantizers (`quant.py`) |
| `pylib_*/` | entry-point metadata that enables the opt-in plugin variants when the directory is on `PYTHONPATH` |
| `results/` | tuned tile tables: the decode path (H100, H200, Blackwell NVFP4) and the matched engine's swap-AB GEMM |
| `train/`, `bench/`, `tests/`, `tools/` | as above |

## Citation

OneBase is described in a paper that is currently under review; a citation will be added here.
