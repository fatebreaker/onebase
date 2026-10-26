"""CPU check of kernels/sr_quant.py (Triton interpreter): every output is one of the two e4m3 neighbours of x / s, and the mean
over seeds of the dequantized output matches x (unbiased) to Monte Carlo error, where round to nearest is biased.
    TRITON_INTERPRET=1 python tests/sr_quant_interp_check.py"""
import os
import sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, "$ONEBASE")
import torch
from lowbit_lora.kernels.sr_quant import quant_rows_sr
from lowbit_lora.w16_engine_fqsr import e4m3_neighbors
torch.manual_seed(0)
M, K, S = 3, 96, 400
x = torch.randn(M, K) * torch.tensor([1.0, 0.01, 30.0])[:, None]
x[0, :8] = 0.0                                           # zeros stay zero
x[1, 8:16] = -x[1, 8:16].abs()                           # negatives
acc = torch.zeros(M, K)
for seed in range(S):
    q, s = quant_rows_sr(x, seed, block_m=4, block_k=32)
    deq = q.float() * s[:, None]
    if seed == 0:
        sref = (x.abs().amax(1) / 448.0).clamp(min=1.0 / (448.0 * 512.0))
        assert torch.allclose(s, sref), "scales"
        y = (x / s[:, None]).clamp(-448, 448)
        lo, hi = e4m3_neighbors(y)
        qf = q.float()
        assert bool(((qf == lo) | (qf == hi)).all()), "an output is not a neighbour"
        rn = y.to(torch.float8_e4m3fn).float() * s[:, None]
    acc += deq
mean = acc / S
step = (hi - lo) * s[:, None]                            # one e4m3 step at each element
err_sr = ((mean - x).abs() / step.clamp_min(1e-30))[step > 0]
err_rn = ((rn - x).abs() / step.clamp_min(1e-30))[step > 0]
print(f"SR: mean |E[q] - x| / step = {float(err_sr.mean()):.4f} (Monte Carlo ~ {0.5 / S ** 0.5:.4f}); "
      f"round to nearest |q - x| / step = {float(err_rn.mean()):.4f}")
assert float(err_sr.mean()) < 3 * 0.5 / S ** 0.5, "biased"
print("zeros:", float(mean[0, :8].abs().max()), " all outputs are neighbours: ok")
