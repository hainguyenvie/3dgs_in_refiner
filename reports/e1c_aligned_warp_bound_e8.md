# E1c — single-source upper bounds on src0 valid & flow-consistent pixels (PSNR from mean MSE)

| scene | raw | raw aligned | final | warp0 | warp0 +colour | **warp0 aligned** | aligned +colour | warp err share HF/MF/LFm/LF | after align | per-band reduction by alignment |
|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 39.77 | 43.47 | 34.30 | 27.89 | 28.32 | **27.45** | 27.93 | 54/20/5/21 | 54/20/5/22 | -7/-5/-12/-8 |
| garden | 44.06 | 47.04 | 37.96 | 29.70 | 29.83 | **29.68** | 29.81 | 59/22/4/15 | 59/22/4/15 | -0/-0/-0/-3 |
| bonsai | 120.00 | 64.26 | 43.13 | 36.09 | 36.18 | **35.96** | 36.05 | 47/30/10/13 | 47/30/10/13 | -3/-3/-3/-2 |
