# Final comparison: Track Paper vs Track MoE

Mean ± std over seeds. `thr` = decision threshold: `@0.5` fixed, `+thr` tuned on a validation slice (never on the test set). Rows marked *no DP guarantee* are not private.

## Protocol: honest

_honest protocol: split first, balance training data only -> real test set (~12.5% attack). This is the protocol we trust._

### Centralized

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| - | Paper (reported) | WUSTL DNN, noise 1.5 | - | 0.932 | 0.915 | 0.942 | 0.928 | - | 0.44 | 0.0001 | yes (reported) |
| - | Baseline | predict everything normal | - | 0.875 | - | 0.000 | 0.000 | 0.500 | - | - | - |
| moe-track | MoE | M1 MoE (momentum+weights) | +thr | 0.950 ± 0.002 | 0.863 ± 0.020 | 0.716 ± 0.020 | 0.782 ± 0.009 | 0.954 ± 0.004 | - | - | no (no DP) |
| moe-track | MoE | M1 MoE (momentum+weights) | @0.5 | 0.889 ± 0.008 | 0.538 ± 0.023 | 0.860 ± 0.021 | 0.661 ± 0.011 | 0.954 ± 0.004 | - | - | no (no DP) |
| paper-track | DNN | P0 notebook recipe (SGD, SMOTE, 100 ep) | @0.5 | 0.889 ± 0.028 | 0.556 ± 0.102 | 0.779 ± 0.060 | 0.642 ± 0.044 | 0.940 ± 0.001 | - | - | no (no DP) |
| paper-track | DNN | P1 adam+smote | +thr | 0.943 ± 0.004 | 0.831 ± 0.047 | 0.693 ± 0.037 | 0.754 ± 0.013 | 0.944 ± 0.006 | - | - | no (no DP) |
| paper-track | DNN | P1 adam+smote | @0.5 | 0.919 ± 0.012 | 0.650 ± 0.062 | 0.784 ± 0.027 | 0.709 ± 0.025 | 0.944 ± 0.006 | - | - | no (no DP) |
| paper-track | DNN | P1 adam+weights | +thr | 0.945 ± 0.002 | 0.853 ± 0.005 | 0.674 ± 0.019 | 0.753 ± 0.012 | 0.952 ± 0.004 | - | - | no (no DP) |
| paper-track | DNN | P1 adam+weights | @0.5 | 0.859 ± 0.022 | 0.470 ± 0.041 | 0.881 ± 0.017 | 0.612 ± 0.032 | 0.952 ± 0.004 | - | - | no (no DP) |
| paper-track | DNN | P1 momentum+smote | +thr | 0.944 ± 0.001 | 0.829 ± 0.026 | 0.695 ± 0.038 | 0.755 ± 0.012 | 0.947 ± 0.006 | - | - | no (no DP) |
| paper-track | DNN | P1 momentum+smote | @0.5 | 0.901 ± 0.010 | 0.576 ± 0.034 | 0.802 ± 0.006 | 0.670 ± 0.024 | 0.947 ± 0.006 | - | - | no (no DP) |
| paper-track | DNN | P1 momentum+weights | +thr | 0.942 ± 0.005 | 0.810 ± 0.067 | 0.710 ± 0.061 | 0.753 ± 0.014 | 0.952 ± 0.006 | - | - | no (no DP) |
| paper-track | DNN | P1 momentum+weights | @0.5 | 0.862 ± 0.038 | 0.482 ± 0.072 | 0.881 ± 0.048 | 0.620 ± 0.050 | 0.952 ± 0.006 | - | - | no (no DP) |
| paper-track | DNN | P1 sgd+smote | +thr | 0.930 ± 0.002 | 0.791 ± 0.016 | 0.604 ± 0.013 | 0.685 ± 0.010 | 0.920 ± 0.006 | - | - | no (no DP) |
| paper-track | DNN | P1 sgd+smote | @0.5 | 0.876 ± 0.011 | 0.504 ± 0.029 | 0.759 ± 0.009 | 0.605 ± 0.024 | 0.920 ± 0.006 | - | - | no (no DP) |
| paper-track | DNN | P1 sgd+weights | +thr | 0.932 ± 0.002 | 0.838 ± 0.026 | 0.566 ± 0.006 | 0.675 ± 0.006 | 0.908 ± 0.002 | - | - | no (no DP) |
| paper-track | DNN | P1 sgd+weights | @0.5 | 0.850 ± 0.018 | 0.445 ± 0.033 | 0.765 ± 0.026 | 0.562 ± 0.020 | 0.908 ± 0.002 | - | - | no (no DP) |
| paper-track | CNN | P3 best recipe (momentum+weights) | +thr | 0.934 ± 0.006 | 0.833 ± 0.098 | 0.610 ± 0.051 | 0.700 ± 0.003 | 0.918 ± 0.002 | - | - | no (no DP) |
| paper-track | CNN | P3 best recipe (momentum+weights) | @0.5 | 0.880 ± 0.020 | 0.520 ± 0.055 | 0.740 ± 0.022 | 0.609 ± 0.033 | 0.918 ± 0.002 | - | - | no (no DP) |
| paper-track | GBT (centralized: no FL, no DP; upper bound) | P6 HistGradientBoosting | +thr | 0.974 ± 0.001 | 0.917 ± 0.001 | 0.870 ± 0.006 | 0.893 ± 0.003 | 0.991 ± 0.001 | - | - | no (no FL, no DP) |
| paper-track | GBT (centralized: no FL, no DP; upper bound) | P6 HistGradientBoosting | @0.5 | 0.970 ± 0.003 | 0.847 ± 0.015 | 0.924 ± 0.009 | 0.884 ± 0.012 | 0.991 ± 0.001 | - | - | no (no FL, no DP) |

### Federated (no DP)

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-track | MoE | M2 FedAvg 3 IID clients (10r x 3e) | +thr | 0.938 ± 0.008 | 0.826 ± 0.081 | 0.654 ± 0.035 | 0.728 ± 0.020 | 0.936 ± 0.001 | - | - | no (no DP) |
| moe-track | MoE | M2 FedAvg 3 IID clients (10r x 3e) | @0.5 | 0.870 ± 0.016 | 0.492 ± 0.039 | 0.826 ± 0.024 | 0.616 ± 0.023 | 0.936 ± 0.001 | - | - | no (no DP) |
| paper-track | DNN | P4 FedAvg 3 IID clients (10r x 3e) | +thr | 0.935 ± 0.005 | 0.858 ± 0.049 | 0.579 ± 0.020 | 0.691 ± 0.018 | 0.916 ± 0.004 | - | - | no (no DP) |
| paper-track | DNN | P4 FedAvg 3 IID clients (10r x 3e) | @0.5 | 0.862 ± 0.016 | 0.471 ± 0.037 | 0.763 ± 0.018 | 0.582 ± 0.025 | 0.916 ± 0.004 | - | - | no (no DP) |
| paper-track | FedGBT (prediction averaging) | P7 3 IID clients | +thr | 0.943 ± 0.006 | 0.780 ± 0.058 | 0.770 ± 0.038 | 0.773 ± 0.015 | 0.967 ± 0.003 | - | - | no DP guarantee |
| paper-track | FedGBT (prediction averaging) | P7 3 IID clients | @0.5 | 0.937 ± 0.010 | 0.719 ± 0.051 | 0.820 ± 0.016 | 0.765 ± 0.025 | 0.967 ± 0.003 | - | - | no DP guarantee |

### Federated + DP

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-track | MoE | DP eps-matched 0.44 clip=1 opt=momentum | +thr | 0.929 ± 0.006 | 0.910 ± 0.053 | 0.484 ± 0.026 | 0.632 ± 0.032 | 0.770 ± 0.007 | 0.43 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP eps-matched 0.44 clip=1 opt=momentum | @0.5 | 0.920 ± 0.014 | 0.899 ± 0.055 | 0.417 ± 0.154 | 0.554 ± 0.148 | 0.770 ± 0.007 | 0.43 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP eps-matched 0.44 clip=1 opt=sgd | +thr | 0.917 ± 0.010 | 0.759 ± 0.068 | 0.498 ± 0.025 | 0.601 ± 0.039 | 0.785 ± 0.010 | 0.43 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP eps-matched 0.44 clip=1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.785 ± 0.010 | 0.43 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=0.0001 opt=sgd | +thr | 0.125 ± 0.000 | 0.125 ± 0.000 | 1.000 ± 0.000 | 0.223 ± 0.000 | 0.333 ± 0.023 | 24.53 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=0.0001 opt=sgd | @0.5 | 0.601 ± 0.272 | 0.053 ± 0.046 | 0.207 ± 0.233 | 0.081 ± 0.074 | 0.333 ± 0.023 | 24.53 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=0.1 opt=sgd | +thr | 0.681 ± 0.118 | 0.239 ± 0.082 | 0.592 ± 0.082 | 0.330 ± 0.064 | 0.685 ± 0.057 | 24.53 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=0.1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.685 ± 0.057 | 24.53 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=1 opt=momentum | +thr | 0.934 ± 0.003 | 0.926 ± 0.038 | 0.519 ± 0.003 | 0.665 ± 0.007 | 0.864 ± 0.003 | 24.53 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=1 opt=momentum | @0.5 | 0.936 ± 0.000 | 0.946 ± 0.007 | 0.515 ± 0.001 | 0.667 ± 0.002 | 0.864 ± 0.003 | 24.53 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=1 opt=sgd | +thr | 0.916 ± 0.010 | 0.750 ± 0.076 | 0.501 ± 0.017 | 0.600 ± 0.035 | 0.793 ± 0.008 | 24.53 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.793 ± 0.008 | 24.53 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=0.0001 opt=sgd | +thr | 0.125 ± 0.000 | 0.125 ± 0.000 | 1.000 ± 0.000 | 0.223 ± 0.000 | 0.333 ± 0.023 | 1.92 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=0.0001 opt=sgd | @0.5 | 0.601 ± 0.272 | 0.053 ± 0.046 | 0.207 ± 0.233 | 0.081 ± 0.074 | 0.333 ± 0.023 | 1.92 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=0.1 opt=sgd | +thr | 0.682 ± 0.121 | 0.241 ± 0.087 | 0.589 ± 0.085 | 0.330 ± 0.067 | 0.685 ± 0.057 | 1.92 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=0.1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.685 ± 0.057 | 1.92 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=1 opt=momentum | +thr | 0.935 ± 0.002 | 0.943 ± 0.030 | 0.516 ± 0.006 | 0.667 ± 0.004 | 0.836 ± 0.002 | 1.92 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=1 opt=momentum | @0.5 | 0.935 ± 0.001 | 0.937 ± 0.017 | 0.515 ± 0.003 | 0.665 ± 0.002 | 0.836 ± 0.002 | 1.92 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=1 opt=sgd | +thr | 0.917 ± 0.011 | 0.776 ± 0.095 | 0.489 ± 0.030 | 0.598 ± 0.038 | 0.792 ± 0.009 | 1.92 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.792 ± 0.009 | 1.92 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP eps-matched 0.44 clip=1 opt=momentum | +thr | 0.934 ± 0.001 | 0.922 ± 0.024 | 0.514 ± 0.005 | 0.660 ± 0.003 | 0.805 ± 0.007 | 0.43 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP eps-matched 0.44 clip=1 opt=momentum | @0.5 | 0.931 ± 0.001 | 0.886 ± 0.020 | 0.519 ± 0.004 | 0.655 ± 0.004 | 0.805 ± 0.007 | 0.43 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP eps-matched 0.44 clip=1 opt=sgd | +thr | 0.904 ± 0.008 | 0.676 ± 0.062 | 0.465 ± 0.009 | 0.550 ± 0.015 | 0.797 ± 0.010 | 0.43 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP eps-matched 0.44 clip=1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.797 ± 0.010 | 0.43 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=0.0001 opt=sgd | +thr | 0.126 ± 0.000 | 0.125 ± 0.000 | 0.998 ± 0.001 | 0.223 ± 0.000 | 0.348 ± 0.053 | 24.53 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=0.0001 opt=sgd | @0.5 | 0.851 ± 0.040 | 0.102 ± 0.131 | 0.012 ± 0.019 | 0.016 ± 0.023 | 0.348 ± 0.053 | 24.53 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=0.1 opt=sgd | +thr | 0.217 ± 0.125 | 0.132 ± 0.013 | 0.930 ± 0.064 | 0.231 ± 0.018 | 0.479 ± 0.108 | 24.53 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=0.1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.479 ± 0.108 | 24.53 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=1 opt=momentum | +thr | 0.934 ± 0.000 | 0.923 ± 0.006 | 0.516 ± 0.000 | 0.662 ± 0.002 | 0.840 ± 0.003 | 24.53 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=1 opt=momentum | @0.5 | 0.932 ± 0.001 | 0.892 ± 0.013 | 0.517 ± 0.001 | 0.654 ± 0.004 | 0.840 ± 0.003 | 24.53 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=1 opt=sgd | +thr | 0.905 ± 0.012 | 0.689 ± 0.095 | 0.466 ± 0.014 | 0.554 ± 0.028 | 0.798 ± 0.011 | 24.53 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.798 ± 0.011 | 24.53 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=0.0001 opt=sgd | +thr | 0.126 ± 0.000 | 0.125 ± 0.000 | 0.998 ± 0.001 | 0.223 ± 0.000 | 0.348 ± 0.053 | 1.92 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=0.0001 opt=sgd | @0.5 | 0.851 ± 0.040 | 0.102 ± 0.131 | 0.012 ± 0.019 | 0.016 ± 0.023 | 0.348 ± 0.053 | 1.92 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=0.1 opt=sgd | +thr | 0.217 ± 0.126 | 0.132 ± 0.013 | 0.929 ± 0.066 | 0.231 ± 0.018 | 0.479 ± 0.108 | 1.92 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=0.1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.479 ± 0.108 | 1.92 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=1 opt=momentum | +thr | 0.934 ± 0.000 | 0.923 ± 0.002 | 0.516 ± 0.000 | 0.662 ± 0.001 | 0.837 ± 0.011 | 1.92 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=1 opt=momentum | @0.5 | 0.931 ± 0.002 | 0.887 ± 0.019 | 0.518 ± 0.001 | 0.654 ± 0.006 | 0.837 ± 0.011 | 1.92 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=1 opt=sgd | +thr | 0.911 ± 0.002 | 0.733 ± 0.010 | 0.455 ± 0.015 | 0.561 ± 0.013 | 0.798 ± 0.011 | 1.92 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=1 opt=sgd | @0.5 | 0.875 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.798 ± 0.011 | 1.92 | 0.0001 | yes (epsilon, delta) |

### Non-IID

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-track | DNN | M4 FedAvg global (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e) | @0.5 | 0.856 ± 0.004 | 0.456 ± 0.009 | 0.774 ± 0.016 | 0.574 ± 0.008 | 0.912 ± 0.005 | - | - | no (no DP) |
| moe-track | MoE | M4 FedAvg global (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e) | @0.5 | 0.867 ± 0.022 | 0.486 ± 0.052 | 0.808 ± 0.033 | 0.605 ± 0.031 | 0.929 ± 0.006 | - | - | no (no DP) |
| moe-track | MoE domain-aware | M4 personalised (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e) | @0.5 | 0.859 ± 0.025 | 0.470 ± 0.055 | 0.820 ± 0.020 | 0.595 ± 0.039 | 0.932 ± 0.006 | - | - | no (no DP) |
| moe-track | MoE local-only | M4 personalised (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e) | @0.5 | 0.841 ± 0.018 | 0.425 ± 0.035 | 0.747 ± 0.012 | 0.541 ± 0.026 | 0.884 ± 0.002 | - | - | no (no DP) |

### Non-IID + DP

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-track | DNN | M5 FedAvg global (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e, DP noise 1.5 clip 1 (delta 1e-4)) | @0.5 | 0.931 ± 0.002 | 0.889 ± 0.019 | 0.517 ± 0.001 | 0.654 ± 0.006 | 0.825 ± 0.008 | 7.04 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | M5 FedAvg global (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e, DP noise 1.5 clip 1 (delta 1e-4)) | @0.5 | 0.933 ± 0.003 | 0.906 ± 0.039 | 0.517 ± 0.003 | 0.658 ± 0.008 | 0.836 ± 0.016 | 7.04 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE domain-aware | M5 personalised (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e, DP noise 1.5 clip 1 (delta 1e-4)) | @0.5 | 0.932 ± 0.001 | 0.904 ± 0.009 | 0.516 ± 0.000 | 0.657 ± 0.002 | 0.830 ± 0.016 | 7.04 | 0.0001 | yes (weights only; descriptors un-noised) |
| moe-track | MoE local-only | M5 personalised (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e, DP noise 1.5 clip 1 (delta 1e-4)) | @0.5 | 0.926 ± 0.009 | 0.937 ± 0.025 | 0.436 ± 0.082 | 0.592 ± 0.078 | 0.811 ± 0.009 | 7.04 | 0.0001 | yes (local DP, no communication) |

### Winners

- Best federated + DP result (honest): **moe-track / MoE / DP noise=1.5 clip=1 opt=momentum** - acc 0.935, F1 0.667, epsilon 1.92.
- Best result at epsilon <= 0.44 (honest): **paper-track / DNN / DP eps-matched 0.44 clip=1 opt=momentum** - acc 0.934, F1 0.660, epsilon 0.43.
  - Beats the paper on accuracy AND F1 (0.932 / 0.928)? **NO**  (note: the paper's numbers come from a different, likely leaky, test set)
  - Best DP run beats the paper on accuracy AND F1? **NO** (at epsilon 1.92, not matched to 0.44)
- Best in *Centralized*: paper-track / GBT (centralized: no FL, no DP; upper bound) / P6 HistGradientBoosting - F1 0.893, acc 0.974.
- Best in *Federated (no DP)*: paper-track / FedGBT (prediction averaging) / P7 3 IID clients - F1 0.773, acc 0.943.
- Best in *Federated + DP*: moe-track / MoE / DP noise=1.5 clip=1 opt=momentum - F1 0.667, acc 0.935.
- MoE minus DNN, F1, same DP setting: DP eps-matched 0.44 clip=1 opt=momentum: -0.028; DP eps-matched 0.44 clip=1 opt=sgd: +0.052; DP noise=0.5 clip=0.0001 opt=sgd: +0.000; DP noise=0.5 clip=0.1 opt=sgd: +0.099; DP noise=0.5 clip=1 opt=momentum: +0.003; DP noise=0.5 clip=1 opt=sgd: +0.047; DP noise=1.5 clip=0.0001 opt=sgd: +0.000; DP noise=1.5 clip=0.1 opt=sgd: +0.099; DP noise=1.5 clip=1 opt=momentum: +0.005; DP noise=1.5 clip=1 opt=sgd: +0.037

## Protocol: paper

_paper protocol: SMOTE on the whole dataset BEFORE the split -> test set is ~50% attack. Use only for a direct comparison with the paper's numbers._

### Centralized

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| - | Paper (reported) | WUSTL DNN, noise 1.5 | - | 0.932 | 0.915 | 0.942 | 0.928 | - | 0.44 | 0.0001 | yes (reported) |
| - | Baseline | predict everything normal | - | 0.500 | - | 0.000 | 0.000 | 0.500 | - | - | - |
| moe-track | MoE | M1 MoE (adam+smote) | +thr | 0.946 ± 0.003 | 0.955 ± 0.008 | 0.937 ± 0.002 | 0.946 ± 0.003 | 0.988 ± 0.001 | - | - | no (no DP) |
| moe-track | MoE | M1 MoE (adam+smote) | @0.5 | 0.944 ± 0.006 | 0.945 ± 0.013 | 0.944 ± 0.003 | 0.944 ± 0.005 | 0.988 ± 0.001 | - | - | no (no DP) |
| paper-track | DNN | P0 notebook recipe (SGD, SMOTE, 100 ep) | @0.5 | 0.898 ± 0.004 | 0.889 ± 0.016 | 0.910 ± 0.017 | 0.899 ± 0.003 | 0.968 ± 0.001 | - | - | no (no DP) |
| paper-track | DNN | P1 adam+smote | +thr | 0.923 ± 0.004 | 0.929 ± 0.016 | 0.917 ± 0.021 | 0.923 ± 0.005 | 0.981 ± 0.001 | - | - | no (no DP) |
| paper-track | DNN | P1 adam+smote | @0.5 | 0.924 ± 0.005 | 0.920 ± 0.015 | 0.929 ± 0.010 | 0.924 ± 0.004 | 0.981 ± 0.001 | - | - | no (no DP) |
| paper-track | DNN | P1 momentum+smote | +thr | 0.920 ± 0.010 | 0.924 ± 0.007 | 0.915 ± 0.027 | 0.919 ± 0.011 | 0.978 ± 0.005 | - | - | no (no DP) |
| paper-track | DNN | P1 momentum+smote | @0.5 | 0.918 ± 0.010 | 0.932 ± 0.030 | 0.904 ± 0.045 | 0.917 ± 0.012 | 0.978 ± 0.005 | - | - | no (no DP) |
| paper-track | DNN | P1 sgd+smote | +thr | 0.873 ± 0.003 | 0.863 ± 0.024 | 0.887 ± 0.025 | 0.874 ± 0.001 | 0.953 ± 0.002 | - | - | no (no DP) |
| paper-track | DNN | P1 sgd+smote | @0.5 | 0.875 ± 0.004 | 0.889 ± 0.003 | 0.857 ± 0.009 | 0.873 ± 0.005 | 0.953 ± 0.002 | - | - | no (no DP) |
| paper-track | CNN | P3 best recipe (adam+smote) | +thr | 0.913 ± 0.003 | 0.915 ± 0.025 | 0.913 ± 0.026 | 0.913 ± 0.002 | 0.976 ± 0.002 | - | - | no (no DP) |
| paper-track | CNN | P3 best recipe (adam+smote) | @0.5 | 0.913 ± 0.008 | 0.936 ± 0.039 | 0.889 ± 0.028 | 0.911 ± 0.005 | 0.976 ± 0.002 | - | - | no (no DP) |
| paper-track | GBT (centralized: no FL, no DP; upper bound) | P6 HistGradientBoosting | +thr | 0.980 ± 0.001 | 0.977 ± 0.003 | 0.984 ± 0.002 | 0.981 ± 0.001 | 0.998 ± 0.000 | - | - | no (no FL, no DP) |
| paper-track | GBT (centralized: no FL, no DP; upper bound) | P6 HistGradientBoosting | @0.5 | 0.980 ± 0.000 | 0.987 ± 0.001 | 0.973 ± 0.001 | 0.980 ± 0.000 | 0.998 ± 0.000 | - | - | no (no FL, no DP) |

### Federated (no DP)

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-track | MoE | M2 FedAvg 3 IID clients (10r x 3e) | +thr | 0.923 ± 0.003 | 0.929 ± 0.019 | 0.915 ± 0.016 | 0.922 ± 0.002 | 0.979 ± 0.001 | - | - | no (no DP) |
| moe-track | MoE | M2 FedAvg 3 IID clients (10r x 3e) | @0.5 | 0.923 ± 0.001 | 0.937 ± 0.011 | 0.908 ± 0.014 | 0.922 ± 0.002 | 0.979 ± 0.001 | - | - | no (no DP) |
| paper-track | DNN | P4 FedAvg 3 IID clients (10r x 3e) | +thr | 0.890 ± 0.005 | 0.879 ± 0.016 | 0.905 ± 0.015 | 0.892 ± 0.004 | 0.963 ± 0.002 | - | - | no (no DP) |
| paper-track | DNN | P4 FedAvg 3 IID clients (10r x 3e) | @0.5 | 0.891 ± 0.003 | 0.901 ± 0.021 | 0.880 ± 0.019 | 0.890 ± 0.001 | 0.963 ± 0.002 | - | - | no (no DP) |
| paper-track | FedGBT (prediction averaging) | P7 3 IID clients | +thr | 0.970 ± 0.002 | 0.966 ± 0.008 | 0.974 ± 0.006 | 0.970 ± 0.002 | 0.996 ± 0.000 | - | - | no DP guarantee |
| paper-track | FedGBT (prediction averaging) | P7 3 IID clients | @0.5 | 0.971 ± 0.002 | 0.981 ± 0.002 | 0.960 ± 0.004 | 0.970 ± 0.002 | 0.996 ± 0.000 | - | - | no DP guarantee |

### Federated + DP

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-track | MoE | DP eps-matched 0.44 clip=1 opt=adam | +thr | 0.684 ± 0.004 | 0.643 ± 0.002 | 0.827 ± 0.016 | 0.724 ± 0.006 | 0.804 ± 0.005 | 0.44 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP eps-matched 0.44 clip=1 opt=adam | @0.5 | 0.724 ± 0.003 | 0.816 ± 0.022 | 0.579 ± 0.020 | 0.677 ± 0.007 | 0.804 ± 0.005 | 0.44 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP eps-matched 0.44 clip=1 opt=sgd | +thr | 0.664 ± 0.021 | 0.630 ± 0.039 | 0.816 ± 0.081 | 0.708 ± 0.009 | 0.784 ± 0.008 | 0.44 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP eps-matched 0.44 clip=1 opt=sgd | @0.5 | 0.707 ± 0.003 | 0.801 ± 0.011 | 0.552 ± 0.007 | 0.653 ± 0.003 | 0.784 ± 0.008 | 0.44 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=0.0001 opt=sgd | +thr | 0.500 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.394 ± 0.031 | 18.93 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=0.0001 opt=sgd | @0.5 | 0.455 ± 0.041 | 0.374 ± 0.106 | 0.283 ± 0.300 | 0.277 ± 0.244 | 0.394 ± 0.031 | 18.93 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=0.1 opt=sgd | +thr | 0.642 ± 0.027 | 0.623 ± 0.049 | 0.750 ± 0.088 | 0.676 ± 0.010 | 0.723 ± 0.006 | 18.93 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=0.1 opt=sgd | @0.5 | 0.667 ± 0.015 | 0.775 ± 0.119 | 0.509 ± 0.154 | 0.594 ± 0.092 | 0.723 ± 0.006 | 18.93 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=1 opt=adam | +thr | 0.853 ± 0.002 | 0.854 ± 0.022 | 0.852 ± 0.030 | 0.853 ± 0.005 | 0.934 ± 0.001 | 18.93 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=1 opt=adam | @0.5 | 0.848 ± 0.005 | 0.897 ± 0.002 | 0.786 ± 0.014 | 0.838 ± 0.007 | 0.934 ± 0.001 | 18.93 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=1 opt=sgd | +thr | 0.664 ± 0.015 | 0.624 ± 0.030 | 0.838 ± 0.083 | 0.713 ± 0.014 | 0.788 ± 0.011 | 18.93 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=0.5 clip=1 opt=sgd | @0.5 | 0.709 ± 0.002 | 0.806 ± 0.010 | 0.551 ± 0.005 | 0.655 ± 0.001 | 0.788 ± 0.011 | 18.93 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=0.0001 opt=sgd | +thr | 0.500 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.394 ± 0.031 | 1.38 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=0.0001 opt=sgd | @0.5 | 0.455 ± 0.041 | 0.374 ± 0.106 | 0.283 ± 0.300 | 0.277 ± 0.244 | 0.394 ± 0.031 | 1.38 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=0.1 opt=sgd | +thr | 0.646 ± 0.027 | 0.630 ± 0.048 | 0.735 ± 0.085 | 0.674 ± 0.009 | 0.723 ± 0.006 | 1.38 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=0.1 opt=sgd | @0.5 | 0.666 ± 0.015 | 0.774 ± 0.120 | 0.509 ± 0.155 | 0.594 ± 0.093 | 0.723 ± 0.006 | 1.38 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=1 opt=adam | +thr | 0.781 ± 0.006 | 0.756 ± 0.023 | 0.834 ± 0.031 | 0.792 ± 0.003 | 0.886 ± 0.002 | 1.38 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=1 opt=adam | @0.5 | 0.788 ± 0.001 | 0.884 ± 0.014 | 0.662 ± 0.013 | 0.757 ± 0.004 | 0.886 ± 0.002 | 1.38 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=1 opt=sgd | +thr | 0.663 ± 0.018 | 0.624 ± 0.035 | 0.838 ± 0.088 | 0.712 ± 0.012 | 0.787 ± 0.011 | 1.38 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | DP noise=1.5 clip=1 opt=sgd | @0.5 | 0.709 ± 0.003 | 0.804 ± 0.010 | 0.552 ± 0.005 | 0.655 ± 0.002 | 0.787 ± 0.011 | 1.38 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP eps-matched 0.44 clip=1 opt=adam | +thr | 0.691 ± 0.009 | 0.639 ± 0.013 | 0.879 ± 0.026 | 0.740 ± 0.003 | 0.812 ± 0.001 | 0.44 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP eps-matched 0.44 clip=1 opt=adam | @0.5 | 0.702 ± 0.001 | 0.796 ± 0.001 | 0.543 ± 0.004 | 0.646 ± 0.002 | 0.812 ± 0.001 | 0.44 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP eps-matched 0.44 clip=1 opt=sgd | +thr | 0.682 ± 0.005 | 0.631 ± 0.008 | 0.878 ± 0.018 | 0.734 ± 0.002 | 0.805 ± 0.002 | 0.44 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP eps-matched 0.44 clip=1 opt=sgd | @0.5 | 0.692 ± 0.001 | 0.802 ± 0.003 | 0.510 ± 0.001 | 0.623 ± 0.001 | 0.805 ± 0.002 | 0.44 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=0.0001 opt=sgd | +thr | 0.500 ± 0.001 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.408 ± 0.079 | 18.93 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=0.0001 opt=sgd | @0.5 | 0.479 ± 0.035 | 0.153 ± 0.132 | 0.019 ± 0.032 | 0.030 ± 0.051 | 0.408 ± 0.079 | 18.93 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=0.1 opt=sgd | +thr | 0.618 ± 0.048 | 0.581 ± 0.044 | 0.887 ± 0.051 | 0.700 ± 0.015 | 0.755 ± 0.022 | 18.93 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=0.1 opt=sgd | @0.5 | 0.574 ± 0.092 | 0.920 ± 0.084 | 0.180 ± 0.235 | 0.251 ± 0.292 | 0.755 ± 0.022 | 18.93 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=1 opt=adam | +thr | 0.799 ± 0.011 | 0.765 ± 0.030 | 0.867 ± 0.029 | 0.812 ± 0.003 | 0.899 ± 0.002 | 18.93 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=1 opt=adam | @0.5 | 0.795 ± 0.004 | 0.872 ± 0.005 | 0.692 ± 0.006 | 0.771 ± 0.004 | 0.899 ± 0.002 | 18.93 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=1 opt=sgd | +thr | 0.677 ± 0.005 | 0.622 ± 0.004 | 0.900 ± 0.003 | 0.736 ± 0.003 | 0.805 ± 0.001 | 18.93 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=0.5 clip=1 opt=sgd | @0.5 | 0.691 ± 0.003 | 0.804 ± 0.005 | 0.505 ± 0.005 | 0.620 ± 0.005 | 0.805 ± 0.001 | 18.93 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=0.0001 opt=sgd | +thr | 0.500 ± 0.001 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.408 ± 0.079 | 1.38 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=0.0001 opt=sgd | @0.5 | 0.479 ± 0.035 | 0.153 ± 0.132 | 0.019 ± 0.032 | 0.030 ± 0.051 | 0.408 ± 0.079 | 1.38 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=0.1 opt=sgd | +thr | 0.617 ± 0.049 | 0.579 ± 0.044 | 0.891 ± 0.048 | 0.700 ± 0.016 | 0.755 ± 0.022 | 1.38 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=0.1 opt=sgd | @0.5 | 0.574 ± 0.093 | 0.919 ± 0.084 | 0.180 ± 0.237 | 0.251 ± 0.293 | 0.755 ± 0.022 | 1.38 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=1 opt=adam | +thr | 0.711 ± 0.012 | 0.663 ± 0.023 | 0.866 ± 0.046 | 0.750 ± 0.003 | 0.836 ± 0.001 | 1.38 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=1 opt=adam | @0.5 | 0.739 ± 0.003 | 0.818 ± 0.008 | 0.616 ± 0.003 | 0.702 ± 0.001 | 0.836 ± 0.001 | 1.38 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=1 opt=sgd | +thr | 0.677 ± 0.004 | 0.623 ± 0.003 | 0.897 ± 0.004 | 0.735 ± 0.003 | 0.805 ± 0.001 | 1.38 | 0.0001 | yes (epsilon, delta) |
| paper-track | DNN | DP noise=1.5 clip=1 opt=sgd | @0.5 | 0.692 ± 0.003 | 0.804 ± 0.004 | 0.507 ± 0.003 | 0.622 ± 0.003 | 0.805 ± 0.001 | 1.38 | 0.0001 | yes (epsilon, delta) |

### Non-IID

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-track | DNN | M4 FedAvg global (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e) | @0.5 | 0.888 ± 0.008 | 0.903 ± 0.009 | 0.869 ± 0.027 | 0.885 ± 0.011 | 0.960 ± 0.005 | - | - | no (no DP) |
| moe-track | MoE | M4 FedAvg global (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e) | @0.5 | 0.917 ± 0.004 | 0.947 ± 0.003 | 0.884 ± 0.005 | 0.914 ± 0.004 | 0.977 ± 0.002 | - | - | no (no DP) |
| moe-track | MoE domain-aware | M4 personalised (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e) | @0.5 | 0.913 ± 0.001 | 0.922 ± 0.027 | 0.904 ± 0.028 | 0.912 ± 0.001 | 0.976 ± 0.001 | - | - | no (no DP) |
| moe-track | MoE local-only | M4 personalised (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e) | @0.5 | 0.889 ± 0.008 | 0.917 ± 0.013 | 0.855 ± 0.025 | 0.885 ± 0.010 | 0.957 ± 0.006 | - | - | no (no DP) |

### Non-IID + DP

| track | model | setting | thr | Acc | Prec | Rec | F1 | AUC | eps | delta | DP guarantee |
|---|---|---|---|---|---|---|---|---|---|---|---|
| moe-track | DNN | M5 FedAvg global (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e, DP noise 1.5 clip 1 (delta 1e-4)) | @0.5 | 0.725 ± 0.007 | 0.808 ± 0.015 | 0.593 ± 0.022 | 0.683 ± 0.012 | 0.828 ± 0.004 | 5.56 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE | M5 FedAvg global (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e, DP noise 1.5 clip 1 (delta 1e-4)) | @0.5 | 0.780 ± 0.024 | 0.867 ± 0.009 | 0.661 ± 0.047 | 0.750 ± 0.034 | 0.872 ± 0.026 | 5.56 | 0.0001 | yes (epsilon, delta) |
| moe-track | MoE domain-aware | M5 personalised (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e, DP noise 1.5 clip 1 (delta 1e-4)) | @0.5 | 0.773 ± 0.017 | 0.851 ± 0.025 | 0.662 ± 0.041 | 0.744 ± 0.025 | 0.868 ± 0.020 | 5.56 | 0.0001 | yes (weights only; descriptors un-noised) |
| moe-track | MoE local-only | M5 personalised (6 clients, Dirichlet 0.3, k-means domains, 10r x 3e, DP noise 1.5 clip 1 (delta 1e-4)) | @0.5 | 0.765 ± 0.020 | 0.853 ± 0.019 | 0.641 ± 0.033 | 0.731 ± 0.027 | 0.853 ± 0.021 | 5.56 | 0.0001 | yes (local DP, no communication) |

### Winners

- Best federated + DP result (paper): **moe-track / MoE / DP noise=0.5 clip=1 opt=adam** - acc 0.853, F1 0.853, epsilon 18.93.
- Best result at epsilon <= 0.44 (paper): **paper-track / DNN / DP eps-matched 0.44 clip=1 opt=adam** - acc 0.691, F1 0.740, epsilon 0.44.
  - Beats the paper on accuracy AND F1 (0.932 / 0.928)? **NO**  (paper protocol only: comparison with the paper's own, likely leaky, pipeline)
  - Best DP run beats the paper on accuracy AND F1? **NO** (at epsilon 18.93, not matched to 0.44)
- Best in *Centralized*: paper-track / GBT (centralized: no FL, no DP; upper bound) / P6 HistGradientBoosting - F1 0.981, acc 0.980.
- Best in *Federated (no DP)*: paper-track / FedGBT (prediction averaging) / P7 3 IID clients - F1 0.970, acc 0.970.
- Best in *Federated + DP*: moe-track / MoE / DP noise=0.5 clip=1 opt=adam - F1 0.853, acc 0.853.
- MoE minus DNN, F1, same DP setting: DP eps-matched 0.44 clip=1 opt=adam: -0.016; DP eps-matched 0.44 clip=1 opt=sgd: -0.026; DP noise=0.5 clip=0.0001 opt=sgd: -0.000; DP noise=0.5 clip=0.1 opt=sgd: -0.023; DP noise=0.5 clip=1 opt=adam: +0.040; DP noise=0.5 clip=1 opt=sgd: -0.023; DP noise=1.5 clip=0.0001 opt=sgd: -0.000; DP noise=1.5 clip=0.1 opt=sgd: -0.026; DP noise=1.5 clip=1 opt=adam: +0.042; DP noise=1.5 clip=1 opt=sgd: -0.023
