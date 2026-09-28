# E9 — mechanism of the intrinsic warp error (synthetic consistent world)

Quantile bins of the conditioning variable (0–50 / 50–75 / 75–90 / 90–97 / 97–100 %). `share` = share of total squared error in the bin.

## bicycle  (25 views)

Error concentration — warp0: 4.3% of pixels carry 50% of error, 17.2% carry 80%. Raw render: 3.1% / 12.4%.

| conditioning | bin | px% | warp0 MSE ×1e3 | warp0 err share | raw MSE ×1e3 | raw err share |
|---|---|---|---|---|---|---|
| |∇ log depth| (edge/multi-layer proxy) | 0–50% | 50 | 1.21 | 32% | 0.08 | 40% |
|  | 50–75% | 25 | 1.79 | 23% | 0.10 | 23% |
|  | 75–90% | 15 | 2.64 | 23% | 0.15 | 21% |
|  | 90–97% | 7 | 3.34 | 14% | 0.16 | 11% |
|  | 97–100% | 3 | 3.62 | 7% | 0.18 | 5% |
| |w0 − w1| source disagreement | 0–50% | 50 | 0.53 | 23% | 0.05 | 31% |
|  | 50–75% | 25 | 0.88 | 18% | 0.07 | 23% |
|  | 75–90% | 15 | 1.62 | 18% | 0.10 | 20% |
|  | 90–97% | 7 | 3.42 | 18% | 0.17 | 16% |
|  | 97–100% | 3 | 10.23 | 23% | 0.29 | 11% |

## garden  (24 views)

Error concentration — warp0: 7.5% of pixels carry 50% of error, 27.1% carry 80%. Raw render: 4.2% / 16.7%.

| conditioning | bin | px% | warp0 MSE ×1e3 | warp0 err share | raw MSE ×1e3 | raw err share |
|---|---|---|---|---|---|---|
| |∇ log depth| (edge/multi-layer proxy) | 0–50% | 50 | 0.69 | 32% | 0.04 | 45% |
|  | 50–75% | 25 | 0.99 | 23% | 0.03 | 19% |
|  | 75–90% | 15 | 1.51 | 20% | 0.05 | 19% |
|  | 90–97% | 7 | 2.45 | 16% | 0.07 | 12% |
|  | 97–100% | 3 | 3.46 | 9% | 0.06 | 5% |
| |w0 − w1| source disagreement | 0–50% | 50 | 0.59 | 29% | 0.03 | 36% |
|  | 50–75% | 25 | 0.90 | 22% | 0.04 | 25% |
|  | 75–90% | 15 | 1.30 | 19% | 0.05 | 19% |
|  | 90–97% | 7 | 2.05 | 14% | 0.07 | 12% |
|  | 97–100% | 3 | 5.31 | 15% | 0.11 | 9% |

## bonsai  (37 views)

Error concentration — warp0: 1.4% of pixels carry 50% of error, 11.5% carry 80%. Raw render: 0.0% / 0.0%.

| conditioning | bin | px% | warp0 MSE ×1e3 | warp0 err share | raw MSE ×1e3 | raw err share |
|---|---|---|---|---|---|---|
| |∇ log depth| (edge/multi-layer proxy) | 0–50% | 50 | 0.10 | 19% | 0.00 | nan% |
|  | 50–75% | 25 | 0.17 | 17% | 0.00 | nan% |
|  | 75–90% | 15 | 0.30 | 17% | 0.00 | nan% |
|  | 90–97% | 7 | 0.73 | 19% | 0.00 | nan% |
|  | 97–100% | 3 | 2.44 | 28% | 0.00 | nan% |
| |w0 − w1| source disagreement | 0–50% | 51 | 0.09 | 20% | 0.00 | nan% |
|  | 50–75% | 26 | 0.15 | 17% | 0.00 | nan% |
|  | 75–90% | 15 | 0.27 | 18% | 0.00 | nan% |
|  | 90–97% | 7 | 0.57 | 17% | 0.00 | nan% |
|  | 97–100% | 3 | 2.25 | 29% | 0.00 | nan% |

