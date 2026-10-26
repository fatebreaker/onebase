"""CPU check of fp8_lora_swap (Triton interpreter; DOT_F32 because the interpreter computes bf16 dots wrongly): the swap-AB
FP8 GEMM with no adapters, with two interleaved adapters and split-K shrink partials, and with three slices, three
adapters and a token without one, each against a torch reference (relative error ~4e-3, bf16 rounding).
    TRITON_INTERPRET=1 python tests/tma_interp_check.py"""
import os
import sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, "$ONEBASE")
import torch
from types import SimpleNamespace
import lowbit_lora.kernels.fp8_lora_swap as T
torch.manual_seed(0)
M, N, K, R = 5, 128, 256, 16
w = torch.randn(N, K) * 0.02
s_w = (w.abs().max() / 448.0).reshape(1).float()
w_q = (w / s_w).to(torch.float8_e4m3fn)
x = torch.randn(M, K)
sx = (x.abs().amax(1) / 448.0).float()
xq = (x / sx[:, None]).to(torch.float8_e4m3fn)
ref = (xq.float() @ w_q.float().t()) * sx[:, None] * s_w
op = SimpleNamespace(w_q=w_q, s_w=s_w, N=N, K=K, R=R, S=1, widths=[N], lora_fp8=False, lora_fp4=False,
                     b_ptrs=torch.zeros(1, dtype=torch.int64), sb_ptrs=torch.zeros(1, dtype=torch.int64),
                     bg=torch.zeros(1), n_seg=torch.tensor([1], dtype=torch.int32), end0=N, end1=N, end2=N, dot_f32=True)
y = torch.empty(M, N, dtype=torch.bfloat16)
tok = torch.full((M,), -1, dtype=torch.int32)
seg_ids = torch.tensor([-1], dtype=torch.int32); seg_n = torch.tensor([M], dtype=torch.int32)
hp = torch.zeros(1 * M * R)
cfg = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2, "GROUP_N": 8, "SPLIT_K": 1}
T.gemm_swap(op, xq, sx, hp, y, tok, seg_ids, seg_n, 1, cfg=cfg)
err = float((y.float() - ref).norm() / ref.norm())
print("rel err (no adapters):", err)
# with two adapters interleaved (rows 0,2,4 -> adapter 0; rows 1,3 -> adapter 1), split-K partials of h (2)
L = 2
B = (torch.randn(L, 1, N, R) * 0.05).to(torch.bfloat16)
split = 2
hp = torch.randn(split, 1, M, R) * 0.5
tok = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32)
seg_ids = torch.tensor([0, 1, -1], dtype=torch.int32); seg_n = torch.tensor([3, 2, 0], dtype=torch.int32)
op.b_ptrs = torch.tensor([B.data_ptr()], dtype=torch.int64); op.n_seg = torch.tensor([3], dtype=torch.int32)
y = torch.empty(M, N, dtype=torch.bfloat16)
T.gemm_swap(op, xq, sx, hp.reshape(-1), y, tok, seg_ids, seg_n, split, cfg=cfg)
h = hp.sum(0)[0].to(torch.bfloat16).float()
ref2 = ref.clone()
for i in range(M):
    ref2[i] += h[i] @ B[int(tok[i]), 0].float().t()
err2 = float((y.float() - ref2).norm() / ref2.norm())
print("rel err (2 adapters):", err2, " lora part norm ratio:", float((ref2 - ref).norm() / ref.norm()))
d = (y.float() - ref2)

# --- three slices
torch.manual_seed(1)
M, K, R, L, split = 6, 256, 16, 3, 2
widths = [64, 32, 32]; N = sum(widths); S = 3
w = torch.randn(N, K) * 0.02; s_w = (w.abs().max() / 448.0).reshape(1).float(); w_q = (w / s_w).to(torch.float8_e4m3fn)
x = torch.randn(M, K); sx = (x.abs().amax(1) / 448.0).float(); xq = (x / sx[:, None]).to(torch.float8_e4m3fn)
base = (xq.float() @ w_q.float().t()) * sx[:, None] * s_w
Bs = [(torch.randn(L, 1, wd, R) * 0.05).to(torch.bfloat16) for wd in widths]
hp = torch.randn(split, S, M, R) * 0.5
tok = torch.tensor([2, 0, -1, 2, 1, 0], dtype=torch.int32)        # one token without an adapter
seg_ids = torch.tensor([0, 1, 2, -1], dtype=torch.int32); seg_n = torch.tensor([2, 1, 2, 1], dtype=torch.int32)
ends = [64, 96, 128]
op = SimpleNamespace(w_q=w_q, s_w=s_w, N=N, K=K, R=R, S=S, widths=widths, lora_fp8=False, lora_fp4=False,
                     b_ptrs=torch.tensor([b.data_ptr() for b in Bs], dtype=torch.int64), sb_ptrs=torch.zeros(3, dtype=torch.int64),
                     bg=torch.zeros(1), n_seg=torch.tensor([4], dtype=torch.int32), end0=64, end1=96, end2=N, dot_f32=True)
y = torch.empty(M, N, dtype=torch.bfloat16)
cfg = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2, "GROUP_N": 8, "SPLIT_K": 1}
T.gemm_swap(op, xq, sx, hp.reshape(-1), y, tok, seg_ids, seg_n, split, cfg=cfg)
ref = base.clone()
h = hp.sum(0).to(torch.bfloat16).float()   # [S, M, R]
offs = [0, 64, 96]
for i in range(M):
    if tok[i] < 0: continue
    for s in range(S):
        ref[i, offs[s]:offs[s] + widths[s]] += h[s, i] @ Bs[s][int(tok[i]), 0].float().t()
print("rel err (3 slices, 3 adapters, a no-adapter token):", float((y.float() - ref).norm() / ref.norm()))

# --- split-K (2 and 4) on the 3-slice case: the LoRA term added once by split 0, the scales applied by the reduction
for sk in (2, 4):
    cfgk = dict(cfg, SPLIT_K=sk)
    yk = torch.empty(M, N, dtype=torch.bfloat16)
    T.gemm_swap(op, xq, sx, hp.reshape(-1), yk, tok, seg_ids, seg_n, split, cfg=cfgk)
    print(f"rel err (3 slices, split-K {sk}):", float((yk.float() - ref).norm() / ref.norm()),
          " max |split - unsplit| / max|y|:", float((yk.float() - y.float()).abs().max() / y.float().abs().max()))
# --- ragged K (K not a multiple of BLOCK_K * SPLIT_K) and M above one token tile
torch.manual_seed(2)
M, K = 37, 320
widths = [64, 32, 32]; N = sum(widths)
w = torch.randn(N, K) * 0.02; s_w = (w.abs().max() / 448.0).reshape(1).float(); w_q = (w / s_w).to(torch.float8_e4m3fn)
x = torch.randn(M, K); sx = (x.abs().amax(1) / 448.0).float(); xq = (x / sx[:, None]).to(torch.float8_e4m3fn)
base = (xq.float() @ w_q.float().t()) * sx[:, None] * s_w
op.w_q, op.s_w, op.K = w_q, s_w, K
tok = torch.randint(-1, L, (M,), dtype=torch.int32)
order = sorted(range(M), key=lambda i: (int(tok[i]) < 0, int(tok[i])))
cnt = [int((tok == l).sum()) for l in range(L)] + [int((tok < 0).sum())]
seg_ids = torch.tensor(list(range(L)) + [-1], dtype=torch.int32); seg_n = torch.tensor(cnt, dtype=torch.int32)
hp = torch.randn(split, S, M, R) * 0.5
h = hp.sum(0).to(torch.bfloat16).float()
ref = base.clone()
for i in range(M):
    if tok[i] < 0: continue
    for s_ in range(S):
        ref[i, offs[s_]:offs[s_] + widths[s_]] += h[s_, i] @ Bs[s_][int(tok[i]), 0].float().t()
for sk in (1, 3):
    cfgk = dict(cfg, SPLIT_K=sk, BLOCK_M=16)
    yk = torch.empty(M, N, dtype=torch.bfloat16)
    T.gemm_swap(op, xq, sx, hp.reshape(-1), yk, tok, seg_ids, seg_n, split, cfg=cfgk)
    print(f"rel err (M 37, K 320 ragged, split-K {sk}):", float((yk.float() - ref).norm() / ref.norm()))
