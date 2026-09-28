# E4 — frequency bands & per-view colour affine (IBGS Protocol R test views)

Bands (σ px): HF <1 | MF 1–4 | LFm 4–16 | LF >16 (incl. DC). `share of gain` = which band the raw→final MSE reduction comes from. `+CA` = oracle per-view 3x4 colour affine fitted on the view itself.

| scene | grp | PSNR raw | final | raw+CA | final+CA | raw err share HF/MF/LFm/LF | gain share HF/MF/LFm/LF | per-band err reduction HF/MF/LFm/LF |
|---|---|---|---|---|---|---|---|---|
| bicycle | outdoor | 25.71 | 26.06 | 26.17 | 26.36 | 38/25/8/29 | -1/3/6/91 | -0/2/9/38 |
| flowers | outdoor | 21.98 | 22.34 | 22.12 | 22.51 | 46/35/9/10 | 25/31/12/33 | 4/7/10/24 |
| garden | outdoor | 27.26 | 27.57 | 27.40 | 27.75 | 41/24/11/25 | -4/-1/9/96 | -1/-0/6/28 |
| stump | outdoor | 27.22 | 27.31 | 27.47 | 27.64 | 31/38/12/19 | 63/-30/20/47 | 4/-2/3/5 |
| treehill | outdoor | 22.92 | 23.06 | 23.50 | 23.43 | 35/32/11/22 | -28/-133/24/236 | -2/-9/5/24 |
| bonsai | indoor | 30.99 | 34.98 | 31.53 | 35.07 | 10/19/16/55 | 3/12/14/70 | 21/43/60/85 |
| counter | indoor | 28.29 | 30.65 | 28.37 | 30.73 | 11/26/26/37 | 5/22/28/46 | 17/33/43/49 |
| kitchen | indoor | 30.52 | 32.10 | 30.82 | 32.30 | 19/26/15/39 | 3/8/13/76 | 5/10/27/62 |
| room | indoor | 31.30 | 32.68 | 31.58 | 33.01 | 14/16/14/56 | 5/12/17/66 | 8/17/26/26 |
| train | tnt | 20.82 | 23.69 | 23.61 | 24.80 | 10/10/6/74 | 2/4/4/90 | 14/22/35/68 |
| truck | tnt | 25.43 | 26.10 | 25.66 | 26.26 | 31/25/13/31 | 9/14/12/65 | 4/8/15/32 |
| drjohnson | db | 29.35 | 29.51 | 29.62 | 29.77 | 6/17/18/59 | 12/18/17/53 | 4/2/2/2 |
| playroom | db | 30.32 | 30.34 | 31.03 | 31.07 | 9/18/17/56 | 18/28/23/30 | 1/1/1/0 |
