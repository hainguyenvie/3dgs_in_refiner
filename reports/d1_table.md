# D1 — ma trận chẩn đoán (6 scene, 1 seed; Δ so IBGS Protocol R và GADA công bố)

```
--- D1 (32/32 done):
detach       bicycle  raw  25.74 (+0.03 vs ibgs, -0.39 vs mcmc)  final 26.09 (+0.03 vs ibgs, -0.07 vs GADA)
detach       bonsai   raw  31.32 (+0.33 vs ibgs, -1.46 vs mcmc)  final 34.88 (-0.10 vs ibgs, -0.49 vs GADA)
detach       counter  raw  28.48 (+0.19 vs ibgs, -0.95 vs mcmc)  final 30.54 (-0.11 vs ibgs, -0.30 vs GADA)
detach       garden   raw  26.85 (-0.41 vs ibgs, -1.34 vs mcmc)  final 27.13 (-0.44 vs ibgs, -0.61 vs GADA)
detach       stump    raw  27.24 (+0.02 vs ibgs, -0.45 vs mcmc)  final 27.36 (+0.05 vs ibgs, +0.03 vs GADA)
detach       train    raw  20.83 (+0.01 vs ibgs, -1.78 vs mcmc)  final 23.73 (+0.04 vs ibgs, +0.06 vs GADA)
ibgs         bicycle  raw  25.75 (+0.04 vs ibgs, -0.38 vs mcmc)  final 26.00 (-0.06 vs ibgs, -0.16 vs GADA)
ibgs         counter  raw  28.30 (+0.01 vs ibgs, -1.13 vs mcmc)  final 30.67 (+0.02 vs ibgs, -0.17 vs GADA)
mcmc_agg     bicycle  raw  25.77 (+0.06 vs ibgs, -0.36 vs mcmc)  final 26.09 (+0.03 vs ibgs, -0.07 vs GADA)
mcmc_agg     bonsai   raw  31.64 (+0.65 vs ibgs, -1.14 vs mcmc)  final 35.24 (+0.26 vs ibgs, -0.13 vs GADA)
mcmc_agg     counter  raw  28.75 (+0.46 vs ibgs, -0.68 vs mcmc)  final 30.86 (+0.21 vs ibgs, +0.02 vs GADA)
mcmc_agg     garden   raw  27.88 (+0.62 vs ibgs, -0.31 vs mcmc)  final 28.05 (+0.48 vs ibgs, +0.31 vs GADA)
mcmc_agg     stump    raw  27.18 (-0.04 vs ibgs, -0.51 vs mcmc)  final 27.30 (-0.01 vs ibgs, -0.03 vs GADA)
mcmc_agg     train    raw  21.70 (+0.88 vs ibgs, -0.91 vs mcmc)  final 24.12 (+0.43 vs ibgs, +0.45 vs GADA)
mcmc_detach  bicycle  raw  25.71 (+0.00 vs ibgs, -0.42 vs mcmc)  final 25.96 (-0.10 vs ibgs, -0.20 vs GADA)
mcmc_detach  bonsai   raw  31.90 (+0.91 vs ibgs, -0.88 vs mcmc)  final 35.04 (+0.06 vs ibgs, -0.33 vs GADA)
mcmc_detach  counter  raw  28.83 (+0.54 vs ibgs, -0.60 vs mcmc)  final 30.73 (+0.08 vs ibgs, -0.11 vs GADA)
mcmc_detach  garden   raw  27.85 (+0.59 vs ibgs, -0.34 vs mcmc)  final 27.94 (+0.37 vs ibgs, +0.20 vs GADA)
mcmc_detach  stump    raw  27.27 (+0.05 vs ibgs, -0.42 vs mcmc)  final 27.37 (+0.06 vs ibgs, +0.04 vs GADA)
mcmc_detach  train    raw  21.53 (+0.71 vs ibgs, -1.08 vs mcmc)  final 24.01 (+0.32 vs ibgs, +0.34 vs GADA)
mcmc_noagg   bicycle  raw  25.78 (+0.07 vs ibgs, -0.35 vs mcmc)  final -
mcmc_noagg   bonsai   raw  31.86 (+0.87 vs ibgs, -0.92 vs mcmc)  final -
mcmc_noagg   counter  raw  28.81 (+0.52 vs ibgs, -0.62 vs mcmc)  final -
mcmc_noagg   garden   raw  28.02 (+0.76 vs ibgs, -0.17 vs mcmc)  final -
mcmc_noagg   stump    raw  27.24 (+0.02 vs ibgs, -0.45 vs mcmc)  final -
mcmc_noagg   train    raw  21.68 (+0.86 vs ibgs, -0.93 vs mcmc)  final -
noagg        bicycle  raw  25.73 (+0.02 vs ibgs, -0.40 vs mcmc)  final -
noagg        bonsai   raw  31.30 (+0.31 vs ibgs, -1.48 vs mcmc)  final -
noagg        counter  raw  28.53 (+0.24 vs ibgs, -0.90 vs mcmc)  final -
noagg        garden   raw  27.40 (+0.14 vs ibgs, -0.79 vs mcmc)  final -
noagg        stump    raw  27.26 (+0.04 vs ibgs, -0.43 vs mcmc)  final -
noagg        train    raw  20.81 (-0.01 vs ibgs, -1.80 vs mcmc)  final -
```

## Kết luận
- **H2 (co-adaptation) bác bỏ.** `noagg` raw − IBGS raw: +0.02/+0.31/+0.24/+0.14/+0.04/−0.01 (mean +0.12, chỉ indoor có ~0.3); `detach` final − IBGS final: mean −0.09 (garden −0.44). Nhánh residual không làm base yếu đi đáng kể; tách gradient không phải đòn bẩy.
- **H1 (densify MCMC trong pipeline IBGS) có ích vừa phải.** `mcmc_agg` final − IBGS: +0.03/+0.26/+0.21/+0.48/−0.01/+0.43 (**mean +0.23**); so GADA: −0.07/−0.13/+0.02/**+0.31**/−0.03/**+0.45** (mean +0.09). Raw +0.44 trung bình so IBGS raw nhưng vẫn kém MCMC thuần 0.3–1.1 dB → loss hình học/plane rasterizer của IBGS trả giá ở raw.
- Fork tái hiện IBGS trong ±0.06 dB (`ibgs` bicycle/counter) → nhiễu giữa lần chạy ≈ 0.05 dB.
- MCMC thuần (raw) **vượt IBGS final** ở cả 3 scene outdoor; IBGS chỉ thắng nhờ indoor.
