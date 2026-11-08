# Privacy-Preserving Federated Learning-Based Intrusion Detection System for IoHT Devices

This is a student replication (Level 300 DCIT research assignment) of the paper below, working through it stage by stage: preprocessing, model building, federated learning, and differential privacy, with every methodology decision and deviation from the paper documented as we go rather than assumed silently.

## Paper being replicated

Fatemeh Mosaiyebzadeh, Seyedamin Pouriyeh, Meng Han, Liyuan Liu, Yixin Xie, Liang Zhao, Daniel Macêdo Batista. *Privacy-Preserving Federated Learning-Based Intrusion Detection System for IoHT Devices*. Electronics 2025, 14, 67.
https://www.mdpi.com/2079-9292/14/1/67 (DOI: https://doi.org/10.3390/electronics14010067)

Original authors' repo: https://github.com/fatemehm/SECIoHT-FL-based-IDS

## The problem it addresses

Internet of Healthcare Things (IoHT) devices — heart monitors, oxygen sensors, blood pressure trackers — connect patients to hospitals over open internet connections vulnerable to spoofing, DoS, data alteration, and ARP attacks. A traditional Intrusion Detection System (IDS) needs to centralize patient data to train a security model, which is a serious privacy problem in healthcare. This paper's answer, **SECIoHT-FL** (Secure Edge Computing IoHT Federated Learning), combines three techniques:

1. **Federated Learning (FL)** — each device trains locally; only model weights are shared with a server, never raw patient data. The server aggregates via weighted averaging (FedAvg).
2. **Differential Privacy (DP)** — even shared weights can leak information (gradient inversion, membership inference), so each device adds calibrated noise to its gradients before sharing them (DP-SGD, via the Opacus library), quantified by a privacy budget epsilon (lower = more private).
3. **Intrusion Detection** — two deep learning models (DNN and CNN) classify traffic as Attack or Normal.

## Datasets

- **wustl-ehms-2020** (Washington University in St. Louis Electronic Health Monitoring System): https://www.cse.wustl.edu/~jain/ehms/index.html — biometric + network-flow features (ECG, SpO2, temperature, blood pressure, etc.), 16,318 raw rows, attacks are spoofing/data alteration, imbalanced 7:1 normal:attack.
- **ECU-IoHT** (Edith Cowan University): https://ro.ecu.edu.au/datasets/48/ — network traffic features, 111,207 raw rows, attacks are ARP spoofing/DoS/Nmap/Smurf. **Currently parked** — see the Stage 1 section below for why.

The WUSTL CSV (3.9 MB) is tracked in the repo at `our_work/data/`. ECU-IoHT is not committed; download it from the link above if you want to use it.

## Repository layout

```
our_work/
  stages/        shared building blocks, one folder per pipeline stage, in order:
                 s1_preprocessing -> s2_models -> s3_training -> s4_federated -> s5_privacy -> s6_evaluation
  baseline_2023/ approach: reconstruct the 2023 DNN-FL paper (its own preprocessing / model / loss / aggregation)
  paper_2025/    approach: improve the 2025 paper's own models (optimizer, threshold, DP, gradient boosting)
  moe/           approach: Mixture-of-Experts + domain-aware aggregation (model.py, aggregation.py)
  compare/       merges all approaches into one final table
  notebooks/     00_original_replication.ipynb (class notebook, Stages 1-6) and 01_pipeline.ipynb (calls the stage blocks)
  kaggle/        ready-made Kaggle notebooks + runbook for long runs
  data/          WUSTL-EHMS-2020 CSV
  archive/       old exploratory runs
docs/            methodology notes for Stages 1-4
references/      papers, replication-audit deck, MoE design document (PDFs are not committed)
streamlit_app/   the demo app (not yet updated with the new results)
```

**Design rule:** every approach's `run.py` is a thin list of calls into `stages/`. To change how something works, edit that one block; never the notebook or the run script. Full map and a "how to change X" table: [`our_work/README.md`](our_work/README.md).

## Project status

### Done

1. **Original replication** of the 2025 paper (Stages 1-6 in `our_work/notebooks/00_original_replication.ipynb`, methodology in `docs/`).
2. **Replication audit** (`references/SECIoHT_FL_Replication_Audit.pptx`). Its main points: the published notebook's settings differ from the paper's text (local epochs 100 vs 1, clip 1e-4 vs 1.0, 2 clients, undersampling vs oversampling, 26 vs 36 features); its test sets (21,814-27,297 rows, one ~89 % attack) are larger than the public data (16,318 rows) and come from a private file; 93.2 % accuracy is only 3.8 points above the 89.4 % majority baseline; the reported epsilon needs independent verification. The audit stresses these are reproducibility findings, not allegations.
3. **Read the 2023 DNN-FL paper** and reconstructed it as `baseline_2023/` (breakdown of what it states and omits: [`baseline_2023/README.md`](our_work/baseline_2023/README.md)). Implemented and smoke-tested; **the full 5/10/15-client run has not been executed yet.**
4. **Accuracy experiments, 3 seeds, both protocols, on Kaggle:** `paper_2025` (stages P0-P7) and `moe` (M1-M5) are complete; results are in each folder's `results/` and merged in [`our_work/compare/final_comparison.md`](our_work/compare/final_comparison.md).
5. **Restructure** into `stages/` blocks + approach folders + one pipeline notebook. Checked: the refactored code reproduces the previous numbers exactly, and the notebook executes end to end.
6. **Honest evaluation**: a leak-free protocol (`honest`), validation-only threshold tuning, macro-F1 and normal-class recall reported next to accuracy and F1.

### What we found (honest protocol, 3 seeds)

* The notebook's training recipe (plain SGD + SMOTE) is weak: moving to momentum/Adam, class weights or SMOTE-on-train, and a validation-tuned threshold lifts attack-class F1 from 0.64 to 0.75.
* The **MoE** adds about +0.03 F1 centrally and +0.04 federated over the DNN, with no reliable advantage under differential privacy. Domain-aware aggregation gave no gain over plain FedAvg.
* **Gradient boosting** is far ahead (F1 0.89) but has no privacy; federated GBT reaches 0.77, also without a privacy guarantee.
* **Differential privacy**: with the paper's clip of 1e-4 the model cannot learn (SGD); with clip 1.0 it learns, but recall stalls near 0.52 whatever the privacy budget (epsilon 0.43 to 24.5), so privacy noise is not the bottleneck.
* **We did not beat the paper under privacy at epsilon 0.44**, under either protocol. The paper's 93.2 % comes from a test set the audit could not reconstruct from public data, so the numbers are not directly comparable.
* The `paper` protocol is only a *balanced-leak imitation* (SMOTE before the split, ~50 % attack test set). It shows how an easy test set inflates scores (same recipe: F1 0.90 vs 0.64), but it is **not** the authors' actual test set.

### To do

| # | Task | Why |
|---|---|---|
| 1 | Run `kaggle/07_baseline_2023.ipynb` (full settings) and add it to the comparison | The 2023 paper is the best-specified federated baseline |
| 2 | Investigate the recall plateau (~0.52) under DP: learning rate, clip, class-weight strength, more rounds | Where a real private improvement is most likely |
| 3 | Run longer federated training (more rounds) | Current runs use 10 rounds x 3 epochs and may be under-trained |
| 4 | Match the authors' stated settings (100 local epochs, 2 clients, noise 0.5) as an explicit "match" run | The audit's step 2 |
| 5 | Ask the authors for `clean_wustl1111.csv` and the preprocessing details | Needed to reproduce their test set |
| 6 | Re-check ECU-IoHT label counts (the 2023 paper says 87,754 *normal*, our Stage 1 recorded the opposite) and decide whether to un-park it | Could unlock a second dataset |
| 7 | Privacy for gradient boosting; add noise to the MoE routing descriptors as the design requires | Currently not private / not implemented |
| 8 | Update the audit deck's results slide with the 3-seed numbers | It still shows early 1-2 repeat results |
| 9 | Update `streamlit_app/` (Benchmark Arena, FL & Privacy Lab) and `docs/` with the agreed numbers | Deliberately postponed until results are settled |

## What's been done so far

The original replication lives in one notebook, `our_work/notebooks/00_original_replication.ipynb`, structured as four stages that build on each other in order. All decisions below were made deliberately and are also documented inline in the notebook's markdown cells, not just here.

**For the full reasoning behind every deviation from the paper — not just the summary below — see the detailed methodology notes in [`docs/`](docs/):**
- [`docs/01-preprocessing-methodology.md`](docs/01-preprocessing-methodology.md) — Stage 1
- [`docs/02-model-building-methodology.md`](docs/02-model-building-methodology.md) — Stage 2
- [`docs/03-federated-learning-and-privacy-methodology.md`](docs/03-federated-learning-and-privacy-methodology.md) — Stages 3–4

Each includes a task log attributing the work per stage between the two of us (Sonpon Ye-shua Chief and Edem Doe Lawson).

### Stage 1 — Preprocessing

Loaded both raw datasets and compared them against the paper's claims:

| | Paper claims | Actually found |
|---|---|---|
| WUSTL | 36 features | 45 raw columns → 36 after dropping 8 non-predictive/leaky columns (`Dir`, `Flgs`, `SrcAddr`, `DstAddr`, `SrcMac`, `DstMac`, `Packet_num`, `Attack Category`) |
| ECU | 7 features | 9 raw columns → 5 usable after dropping 4 (`No.`, `Info`, `Type`, `Type of attack`) |

Other findings worth flagging:
- **WUSTL's `Sport` column** mixes numeric ports with service names (`'http'`, `'dircproxy'`) as text — fixed with `pd.to_numeric(errors='coerce')` + drop unconvertible rows.
- **ECU's label distribution is inverted**: Attack (87,754) outnumbers Normal (23,453) — the opposite of what a real network looks like, suggesting an artificially constructed dataset. **This is why ECU is currently parked** rather than used in Stages 2–4 — we don't want that discrepancy silently contaminating model results until it's understood.
- Class imbalance fixed with SMOTE; features normalized with `StandardScaler`.

### Stage 2 — Model Building & Centralized Baseline (WUSTL only)

- **A methodology fix over the initial Stage 1 approach**: SMOTE was originally applied to the *entire* dataset with no train/test split, which risks synthetic rows leaking between train and test and inflating accuracy. Stage 2 instead: cleans → **splits 80/20 first** (stratified, on real data) → SMOTE the training split only → scales (fit on train, applied to test). The test set stays 100% real, unseen data.
- **DNN**: paper specifies `64(8×8) → 32 → 2`. We treat `(8×8)` as how the paper's figure *drew* a 64-unit layer (an 8×8 grid = 64 neurons), not an instruction to reshape tabular features into a 2D image — there's no spatial structure to preserve in flow/biometric features. Implemented as `Linear(36→64→32→2)` with ReLU, `CrossEntropyLoss` standing in for the paper's softmax output (mathematically equivalent, numerically standard).
- **CNN**: paper gives layer sizes `128→64→32→16→2` but no kernel/stride/pooling spec for non-image input. Reconstructed as four `Conv1d` layers (channels 128/64/32/16, kernel=3, `padding=1` to preserve sequence length) → `AdaptiveAvgPool1d(1)` → `Linear(16→2)`. This is a best-effort reconstruction, explicitly not a verified match to the authors' code.
- **Epoch sweep**: rather than guessing an epoch count, both models train up to 150 epochs with periodic test-set checkpointing (not just train accuracy), automatically keeping the best-by-test-accuracy checkpoint — standard early-stopping-by-checkpointing, addressing that a fixed 50-epoch guess had the CNN still visibly improving.
- Verified results from the one successful local run (50-epoch baseline, before the epoch sweep): **DNN 87.7% accuracy** (precision 0.51, recall 0.73, F1 0.60), **CNN 70.2% accuracy** (precision 0.27, recall 0.80, F1 0.40). Lower than the paper's 93.2% — expected, since this has no FL/DP yet and uses reconstructed (not verified) architectures. Re-running the notebook (with the epoch sweep now included) will produce updated numbers.

### Stage 3 — Federated Learning (FedAvg, WUSTL, DNN)

Simulates 3 clients (representing separate devices/hospitals) holding disjoint local shards of the training data, training locally, and exchanging only model weights with a server that FedAvg-aggregates them (weighted by each client's sample count).

- **Data split across clients**: i.i.d. random shuffle-split — real hospitals would likely be non-i.i.d. (different attack mixes, patient populations), which this doesn't model; flagged as a simplification.
- **An intentional asymmetry between scaling and SMOTE**: the `StandardScaler` is fit once on the combined training set (sharing aggregate per-feature mean/variance across the federation — a standard low-risk federated normalization pattern, not raw data sharing), but **SMOTE runs independently per client**, strictly on that client's own local shard. This is the part that has to stay local — a shared SMOTE (like Stage 2 uses, deliberately, since Stage 2 has no privacy requirement) would let one client's synthetic rows leak signal derived from another client's real data.
- **Rounds/epochs**: paper's config (`100 epochs`, `100 communication rounds`) read together implies a very large amount of compute whose exact meaning isn't fully clear. Used `NUM_ROUNDS=20`, `LOCAL_EPOCHS=5` (100 total local epochs per client, same order of magnitude as Stage 2's 50) — both are one-line changes to scale up.
- DNN only for this stage (stronger performer in Stage 2); reuses Stage 2's exact train/test split (saved to `.npy` so the centralized-vs-federated comparison is apples-to-apples).

### Stage 4 — Differential Privacy (Opacus, DP-SGD)

Wraps each Stage 3 client's local training with Opacus so noise is added **on the device, before anything is sent to the server** (Local DP) — closing the gap that Stage 3's exact, un-noised weights still leave open (gradient inversion / membership inference risk even without sharing raw data).

- **Mechanism**: per-sample gradient clipping (L2 norm ≤ `max_grad_norm`) + Gaussian noise scaled by `noise_multiplier × max_grad_norm`, added before the optimizer step. Opacus's `PrivacyEngine.get_epsilon(delta)` tracks cumulative privacy budget spent.
- **A structural detail that matters**: each client is a *persistent* object (`DPClient`) whose model/optimizer/`PrivacyEngine` are created once and live for the whole run, with only weights overwritten each round from the global model. Re-wrapping a fresh `PrivacyEngine` every round (mirroring Stage 3's fresh-model-per-round pattern) would silently reset the privacy accountant and under-report the true cumulative epsilon — a real correctness bug, not cosmetic.
- **Runs both of the paper's noise settings** for direct comparison: `noise_multiplier=1.5` (paper's recommended) and `0.5` (paper's "not recommended," where they report their CNN collapsing to TN=0).
- **Flagged, not silently changed**: the paper states `max_grad_norm=10⁻⁴`, unusually aggressive versus the typical ~1.0 for DP-SGD — implemented as stated, with an explicit note that a collapsed/non-learning model at this value is worth checking against the paper by hand rather than assumed to be a code bug.
- Verified (without running on real data locally — see **Known limitations** below): a synthetic smoke test confirmed the exact `DPClient` pattern (wrap once, copy weights in per round, accumulate epsilon across rounds) works correctly on this Opacus/PyTorch version — epsilon grew from 1.17 → 1.61 across two simulated rounds, as it should.
- Produces the full comparison chain: centralized → federated (no DP) → federated+DP(1.5) → federated+DP(0.5), saved to `stage4_full_comparison_wustl.csv`.

### Known limitations / honest caveats

- **Stage 3 and Stage 4 of the original notebook** were later re-implemented at scale in the new pipeline and run with 3 seeds on the real dataset (see **Project status** above); the table below shows those results.
- The development machine used for local work has 2 CPU cores with unusually high per-operation overhead for PyTorch training (measured: a small CNN took ~40s/epoch, when it should be near-instant) — full-scale local execution wasn't practical, hence moving to Colab.
- ECU-IoHT is parked, not abandoned — the inverted label distribution needs to be understood before it's trusted for training.
- Several architecture/hyperparameter choices are documented assumptions, not verified matches to the paper's code (see Stage 2–4 sections above): the DNN's `(8×8)` notation, the CNN's convolutional design, client count, communication rounds/local epochs, and the DP clipping norm.

## How to run

1. Download both datasets from the links above.
2. Open `our_work/notebooks/00_original_replication.ipynb` in Google Colab (or upload it there).
3. Upload both dataset files into the Colab session's working directory (`/content/`) — same directory as the notebook.
4. Runtime → Change runtime type → GPU recommended (Stage 3/4 and the epoch sweep are CPU-heavy).
5. Runtime → Run all.

The notebook is fully linear — no hidden dependency on cells run out of order.

## Results snapshot

Honest protocol (real test set, 12.5 % attacks), **mean over 3 seeds**, validation-tuned threshold unless noted. Always-"normal" scores 87.5 % accuracy, so read F1 and macro-F1, not accuracy alone. Full tables with std, both protocols and all DP settings: [`our_work/compare/final_comparison.md`](our_work/compare/final_comparison.md).

| Setting | Accuracy | Precision | Recall | F1 | Macro-F1 | epsilon |
|---|---|---|---|---|---|---|
| Predict everything normal | 0.875 | - | 0.000 | 0.000 | - | - |
| Paper reported (different, unreconstructable test set) | 0.932 | 0.915 | 0.942 | 0.928 | - | 0.44 |
| Notebook recipe, centralized DNN (SGD + SMOTE, @0.5) | 0.889 | 0.556 | 0.779 | 0.642 | 0.788 | - |
| Improved DNN, centralized (momentum + weights) | 0.942 | 0.810 | 0.710 | 0.753 | 0.860 | - |
| CNN, centralized | 0.934 | 0.833 | 0.610 | 0.700 | 0.831 | - |
| **MoE**, centralized | 0.950 | 0.863 | 0.716 | 0.782 | 0.877 | - |
| Gradient boosting, centralized (no FL, no DP) | 0.974 | 0.917 | 0.870 | 0.893 | 0.939 | - |
| FedAvg DNN, 3 clients | 0.935 | 0.858 | 0.579 | 0.691 | 0.827 | - |
| FedAvg MoE, 3 clients | 0.938 | 0.826 | 0.654 | 0.728 | 0.846 | - |
| Federated GBT (no DP guarantee) | 0.943 | 0.780 | 0.770 | 0.773 | 0.870 | - |
| Federated DNN + DP (noise 1.5, clip 1.0) | 0.934 | 0.923 | 0.516 | 0.662 | 0.813 | 1.92 |
| Federated MoE + DP (noise 1.5, clip 1.0) | 0.935 | 0.943 | 0.516 | 0.667 | 0.815 | 1.92 |
| Federated DNN + DP (epsilon-matched) | 0.934 | 0.922 | 0.514 | 0.660 | 0.812 | 0.43 |
| Federated MoE + DP (epsilon-matched) | 0.929 | 0.910 | 0.484 | 0.632 | 0.796 | 0.43 |
| 2023 DNN-FL reconstruction (5/10/15 clients) | *pending: Kaggle notebook 07* | | | | | - |

All DP rows use delta = 1e-4. Epsilon is accumulated per client over 10 rounds x 3 local epochs.

## Stages implemented in the notebook

- **Stage 1 — Preprocessing & Data Cleaning**: Drop 8 non-predictive/leaky columns, fix `Sport` mixed types, and park ECU-IoHT.
- **Stage 2 — Model Building & Centralized Baseline**: Stratified 80/20 split before SMOTE, training DNN (100 epochs) and CNN (100 epochs) baselines.
- **Stage 3 — Federated Learning (FedAvg)**: 3 simulated hospital edge clients exchanging model weights over 100 communication rounds (100 local epochs per client).
- **Stage 4 — Differential Privacy (Opacus DP-SGD)**: Persistent `DPClient` accounting with noise multipliers 1.5 and 0.5 over 100 rounds.
- **Stage 5 — Formal Evaluation & Visualizations**: Complete evaluation comparison table matching MDPI benchmark format, performance metrics bar charts, and confusion matrix heatmaps (`stage5_benchmark_evaluation.csv`, `stage5_evaluation_plots.png`).
- **Stage 6 — Model Explainability (SHAP)**: KernelExplainer computing global feature attribution rankings, summary beeswarm plots, and local instance waterfall explanations (`stage6_shap_summary.png`, `stage6_feature_importance.csv`, `stage6_local_waterfall_attack.png`).

## Algorithms used

Neural: DNN, CNN (2025 paper), DNN-FL 64-64-32 sigmoid (2023 paper), Mixture-of-Experts (design document).
Tree-based: gradient boosting (centralized, and federated by prediction averaging; no DP guarantee).
Federated: FedAvg; FedAvg with server-side Adam (2023); domain-aware aggregation (MoE). Privacy: Opacus DP-SGD.

## Citation

Fatemeh Mosaiyebzadeh, Seyedamin Pouriyeh, Meng Han, Liyuan Liu, Yixin Xie, Liang Zhao, Daniel Macêdo Batista. Privacy-Preserving Federated Learning-Based Intrusion Detection System for IoHT Devices. https://www.mdpi.com/2079-9292/14/1/67

## Accuracy experiments: how to run them

Three approaches share the same data, split and building blocks (`our_work/stages/`) so their results are directly comparable:

- [`our_work/paper_2025/`](our_work/paper_2025/README.md) - improves the 2025 paper's own models and pipeline (optimizer, class weights, threshold tuning, fair DP settings, gradient boosting). No MoE.
- [`our_work/moe/`](our_work/moe/README.md) - the Mixture-of-Experts model and domain-aware aggregation from `references/MoE-FL_Gateway_Design_Document.docx`, applied to WUSTL.
- [`our_work/baseline_2023/`](our_work/baseline_2023/README.md) - reconstruction of the 2023 DNN-FL paper.
- [`our_work/compare/`](our_work/compare/README.md) merges them; long runs go through [`our_work/kaggle/RUNBOOK.md`](our_work/kaggle/RUNBOOK.md).

Every experiment supports two protocols: `honest` (split first, balance training data only; the one we trust) and `paper` (a balanced-leak imitation, not the authors' real test set).
