# 2023 DNN trained federated (step 8): FedAvg + server Adam, no DP

Run `20261127-104159`. 100 rounds x 1 local epoch, client Adam lr 0.01, server Adam lr 0.05, eps 1e-07, batch 1024, Keras-style init. Mean ± sample std over seeds [0, 1, 2]. **The honest test set is not used here (step 9).** Numbers are the FINAL-round model on the VALIDATION set at threshold 0.5; for `as_published` that validation set is the authors' reported evaluation set (no test set exists in their pipeline).

## as_published

| clients | inputs | val rows / attacks | always-normal acc | accuracy % | precision % | recall % | AUC | best round | train loss r1 -> r100 |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 35 | 1635 / 215 | 0.8685 | 89.15 ± 2.68 | 60.69 ± 14.02 | 59.38 ± 2.20 | 0.852 ± 0.005 | 66 | 0.687 -> 0.497 |
| 10 | 35 | 1640 / 192 | 0.8829 | 86.40 ± 5.39 | 48.23 ± 12.62 | 75.69 ± 7.82 | 0.908 ± 0.002 | 98 | 0.688 -> 0.409 |
| 5 | 35 | 1635 / 183 | 0.8881 | 90.01 ± 2.60 | 55.68 ± 9.13 | 69.58 ± 5.79 | 0.911 ± 0.016 | 94 | 0.674 -> 0.331 |

Paper Table III (the authors' notebook output), for comparison:

| clients | accuracy % | precision % | recall % |
|---|---|---|---|
| 15 | 90.64 | 70.39 | 49.76 |
| 10 | 91.40 | 65.05 | 61.42 |
| 5 | 90.70 | 56.40 | 76.63 |

## as_published_fixed

| clients | inputs | val rows / attacks | always-normal acc | accuracy % | precision % | recall % | AUC | best round | train loss r1 -> r100 |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 35 | 1635 / 215 | 0.8685 | 89.48 ± 1.08 | 60.88 ± 6.30 | 59.07 ± 5.28 | 0.856 ± 0.018 | 96 | 0.736 -> 0.368 |
| 10 | 35 | 1640 / 192 | 0.8829 | 82.72 ± 6.93 | 40.51 ± 10.10 | 74.83 ± 8.42 | 0.904 ± 0.007 | 87 | 0.735 -> 0.345 |
| 5 | 35 | 1635 / 183 | 0.8881 | 87.97 ± 2.73 | 48.61 ± 6.61 | 75.96 ± 5.46 | 0.910 ± 0.002 | 74 | 0.717 -> 0.535 |

## honest

| clients | inputs | val rows / attacks | always-normal acc | accuracy % | precision % | recall % | AUC | best round | train loss r1 -> r100 |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 36 | 1958 / 246 | 0.8744 | 81.34 ± 2.17 | 39.12 ± 2.79 | 85.37 ± 4.79 | 0.920 ± 0.006 | 62 | 0.710 -> 0.364 |
| 10 | 36 | 1958 / 246 | 0.8744 | 75.11 ± 5.99 | 32.96 ± 4.78 | 89.70 ± 3.99 | 0.912 ± 0.005 | 86 | 0.711 -> 0.299 |
| 5 | 36 | 1958 / 246 | 0.8744 | 83.78 ± 6.13 | 45.55 ± 13.24 | 81.84 ± 9.70 | 0.916 ± 0.008 | 85 | 0.712 -> 0.341 |
