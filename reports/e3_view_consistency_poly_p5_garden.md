# E3 — view consistency (MCMC Protocol R, RAFT GT->render)

| scene | split | n | PSNR | +align affine | +align poly3 | +align full | flow rms px | affine rms px | affine expl. var | poly3 expl. var | poly3 resid rms px | transl. px | HF share err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| garden | train | 54 | 30.31 | +0.16 | +1.05 | +1.17 | 0.253 | 0.098 | 15% | 89% | 0.086 | 0.050 | 64% |
| garden | test | 24 | 28.33 | +0.15 | +0.95 | +1.10 | 0.377 | 0.117 | 6% | 46% | 0.192 | 0.055 | 58% |
