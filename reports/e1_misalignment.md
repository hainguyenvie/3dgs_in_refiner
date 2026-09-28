# E1 — misalignment vs warp gain (Protocol R test renders, RAFT fwd-bwd consistent pixels)

| scene | grp | mis. median px | >1px | >2px | raw err expl. by misalign. (IBGS) | (MCMC) | rel. gain final vs raw | gain @<0.5px | gain @1-2px | gain @>4px | HF share resid. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | outdoor | 0.16 | 2% | 0% | 4% | 5% | +6.9% | +7.5% | +0.1% | -0.1% | 36% |
| flowers | outdoor | 0.26 | 9% | 1% | 5% | nan% | +8.8% | +9.2% | +6.6% | -0.7% | 56% |
| garden | outdoor | 0.15 | 2% | 0% | 8% | 10% | +3.7% | +3.7% | +2.3% | +4.1% | 41% |
| stump | outdoor | 0.15 | 2% | 0% | 5% | 5% | +1.9% | +2.2% | -2.2% | +1.4% | 48% |
| treehill | outdoor | 0.39 | 25% | 10% | 7% | nan% | +7.0% | +11.9% | +1.6% | -0.4% | 39% |
| bonsai | indoor | 0.18 | 3% | 1% | 2% | 3% | +40.3% | +39.8% | +43.8% | +5.4% | 16% |
| counter | indoor | 0.08 | 0% | 0% | 0% | 0% | +25.4% | +25.5% | +25.0% | +14.8% | 17% |
| kitchen | indoor | 0.10 | 1% | 0% | 1% | 0% | +19.9% | +19.7% | +28.8% | +10.0% | 23% |
| room | indoor | 0.08 | 0% | 0% | 0% | 0% | +17.0% | +17.1% | +16.4% | +3.8% | 16% |
| train | tnt | 0.29 | 9% | 2% | 3% | 4% | +36.1% | +38.9% | +22.4% | +17.8% | 16% |
| truck | tnt | 0.22 | 7% | 1% | 6% | 8% | +11.7% | +12.0% | +13.0% | +24.4% | 21% |
| drjohnson | db | 0.39 | 16% | 3% | 7% | 7% | +4.2% | +5.5% | +2.5% | +0.8% | 36% |
| playroom | db | 0.32 | 8% | 1% | 4% | 2% | +0.8% | +0.7% | +1.4% | +0.5% | 43% |
