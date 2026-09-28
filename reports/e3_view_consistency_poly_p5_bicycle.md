# E3 — view consistency (MCMC Protocol R, RAFT GT->render)

| scene | split | n | PSNR | +align affine | +align poly3 | +align full | flow rms px | affine rms px | affine expl. var | poly3 expl. var | poly3 resid rms px | transl. px | HF share err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | train | 57 | 26.68 | +0.78 | +1.06 | +1.13 | 0.393 | 0.304 | 64% | 89% | 0.122 | 0.193 | 60% |
| bicycle | test | 25 | 26.28 | +0.43 | +0.73 | +0.80 | 0.377 | 0.247 | 40% | 82% | 0.136 | 0.137 | 55% |
