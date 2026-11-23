# 2023 DNN trained centrally on the honest pipeline (step 7)

Run `20261123-142929`, seeds [0, 1, 2]. Adam lr 0.01, eps 1e-07, batch 1024, 100 epochs, Keras-style init, 1:1 balanced training rows (2782), 36 features. Checkpoint = best validation AUC; threshold = F1-maximising on the VALIDATION set; test set scored once per seed. Mean ± sample std over seeds.

Test set: 3263 rows, 409 attacks (199 Spoofing + 210 Data Alteration). **Always-predict-normal accuracy: 0.8747.**

| metric | threshold 0.5 | validation-tuned threshold |
|---|---|---|
| accuracy | 0.864 ± 0.021 | 0.930 ± 0.004 |
| precision | 0.479 ± 0.041 | 0.718 ± 0.022 |
| recall | 0.883 ± 0.022 | 0.725 ± 0.005 |
| F1 | 0.621 ± 0.033 | 0.721 ± 0.013 |
| macro-F1 | 0.769 ± 0.024 | 0.841 ± 0.008 |
| normal-class recall | 0.861 ± 0.026 | 0.959 ± 0.004 |
| AUC | 0.949 ± 0.002 | 0.949 ± 0.002 |
| recall Spoofing | 0.760 ± 0.045 | 0.434 ± 0.010 |
| recall Data Alteration | 1.000 ± 0.000 | 1.000 ± 0.000 |
| threshold | 0.50 | 0.923 ± 0.035 |

Confusion matrices per seed (rows = actual, columns = predicted; TN FP / FN TP):

| seed | threshold | TN | FP | FN | TP | best epoch | best val AUC | train seconds |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.50 | 2372 | 482 | 42 | 367 | 74 | 0.9435 | 3.7 |
| 0 | 0.96 | 2745 | 109 | 112 | 297 | 74 | 0.9435 | 3.7 |
| 1 | 0.50 | 2501 | 353 | 58 | 351 | 100 | 0.9386 | 5.2 |
| 1 | 0.89 | 2724 | 130 | 115 | 294 | 100 | 0.9386 | 5.2 |
| 2 | 0.50 | 2499 | 355 | 43 | 366 | 88 | 0.9444 | 5.6 |
| 2 | 0.92 | 2744 | 110 | 111 | 298 | 88 | 0.9444 | 5.6 |
