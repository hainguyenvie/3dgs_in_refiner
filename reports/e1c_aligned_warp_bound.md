# E1c — single-source upper bounds on src0 valid & flow-consistent pixels (PSNR from mean MSE)

| scene | raw | raw aligned | final | warp0 | warp0 +colour | **warp0 aligned** | aligned +colour | warp err share HF/MF/LFm/LF | after align | per-band reduction by alignment |
|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 25.95 | 26.29 | 26.69 | 23.32 | 24.73 | **23.14** | 24.70 | 40/16/5/40 | 39/15/5/41 | -1/3/-4/-8 |
| garden | 28.64 | 29.67 | 28.92 | 26.35 | 26.55 | **26.88** | 27.10 | 55/21/4/20 | 53/19/4/23 | 12/18/3/-2 |
| bonsai | 30.26 | 30.38 | 35.34 | 32.53 | 32.67 | **32.22** | 32.34 | 40/29/13/19 | 42/28/11/18 | -14/-5/3/-2 |
| counter | 28.99 | 28.85 | 31.68 | 29.43 | 29.58 | **29.54** | 29.68 | 27/30/18/26 | 28/30/17/26 | -1/3/5/2 |
| train | 19.81 | 20.04 | 23.63 | 22.48 | 23.85 | **22.40** | 23.75 | 27/14/7/52 | 30/12/7/51 | -17/10/-0/-1 |
