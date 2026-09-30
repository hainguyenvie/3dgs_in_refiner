# E3 — view consistency (MCMC Protocol R, RAFT GT->render)

| scene | split | n | PSNR | +align affine | +align poly3 | +align full | flow rms px | affine rms px | affine expl. var | poly3 expl. var | poly3 resid rms px | transl. px | HF share err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | train | 57 | 27.16 | +0.12 | +0.24 | +0.36 | 0.255 | 0.149 | 37% | 72% | 0.133 | 0.086 | 58% |
| bicycle | test | 25 | 25.72 | +0.20 | +0.42 | +0.49 | 0.350 | 0.206 | 37% | 78% | 0.150 | 0.124 | 56% |
