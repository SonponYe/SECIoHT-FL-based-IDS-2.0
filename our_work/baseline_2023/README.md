# baseline_2023: a reconstruction of the 2023 DNN-FL paper

**Source:** F. Mosaiyebzadeh, S. Pouriyeh, R. M. Parizi, M. Han, D. M. Batista, *Intrusion Detection System for IoHT Devices using Federated Learning*, IEEE INFOCOM 2023 Workshops (AidTSP). The PDF is kept locally in `references/` (not committed: third-party licence). Authors' code: `github.com/fatemehm/Federated-Learning-IDS`.

**Why it is here (the audit's "Task A"):** it is the earlier, simpler paper by the same group. It gives a transparent federated baseline **without differential privacy**, on the same WUSTL dataset. It is a better-specified starting point than the 2025 paper, and it is a fair "what did federated learning alone achieve" reference.

Run from `our_work/`:

```bash
python -m baseline_2023.run --stage all --clients 5 10 15 --seeds 0 1 2     # paper's settings (100 rounds x 1 local epoch)
python -m baseline_2023.run --stage all --quick --seeds 0                   # smoke test
```

## 1. The paper, step by step

| # | Step | What the paper says | Where |
|---|---|---|---|
| 1 | **Goal** | Federated deep neural network (DNN-FL) to detect attacks in IoHT traffic; no data leaves a hospital. | I, III |
| 2 | **Dataset 1: wustl-ehms-2020** | Built by Hady et al. on a medical testbed (ECG, SpO2, temperature, blood-pressure sensors) with a second computer launching attacks. **2,046 attack + 14,272 normal rows (16,318)**, **36 features** (network + biometric), two attack categories: **spoofing** and **data alteration**. | IV-B1 |
| 3 | **Dataset 2: ECU-IoHT** | MySignals healthcare kit; **7 network features** (source, destination, protocol, type of attack...); 4 attack types (ARP spoofing, DoS, nmap scan, smurf). Counts given as **87,754 normal, 23,453 attack**. | IV-B2 |
| 4 | **Client split** | The whole data set is divided into **5, 10 or 15 clients** *before* cleaning. | IV-C |
| 5 | **Cleaning** | Remove NaN values and duplicate rows. Label: 0 = normal, 1 = attack. | IV-C |
| 6 | **Balancing** | **Undersampling** of the majority class (cites an over/under-sampling overview). | IV-C |
| 7 | **Scaling** | `StandardScaler()` (the text says "0 to 1", which is not what StandardScaler does). | IV-C |
| 8 | **Model** | 1 input layer, hidden layers of **64, 64, 32** neurons, ReLU, **1 output neuron with sigmoid**, binary cross-entropy. Architecture taken from Rey et al. (2022) and built in TensorFlow Federated. | III-B1 |
| 9 | **Federated setup (Table II)** | 15/10/5 participants; client optimizer **Adam, lr 0.01**; server optimizer **Adam, lr 0.05**; **100 epochs**; **batch 1024**; **100 rounds**. (The authors' notebook shows "100 epochs" is the same thing as the 100 rounds - see section 6.) | Table II |
| 10 | **Federated process** | Server sends the global model; each client trains locally; clients send weights; server **averages** them; repeat. | III-B1, Fig. 1 |
| 11 | **Centralised baselines** | **DNN** (64-64-32, lr 0.1, batch 20, sigmoid); **LSTM** (size 4, dropout 0.5, lr 0.1, 100 epochs, Adam, BCE); **CNN-LSTM** (LSTM size 20, dropout 0.2, lr 0.1, 100 epochs, Adam, BCE). Hyper-parameters tuned with `GridSearchCV`. | III-B2 |
| 12 | **Environment** | Google Colab, 12 GB RAM, 2 CPU cores; Pandas; TensorFlow/Keras/TFF. | IV-A |
| 13 | **Metrics** | Accuracy, precision, recall, loss, train and test time. (No F1, no confusion matrix, no AUC.) | IV |
| 14 | **Results (WUSTL)** | DNN-FL 10 clients: **91.40 % acc / 65.05 % precision / 61.42 % recall**; 15 clients 90.64/70.39/49.76; 5 clients 90.70/56.40/76.63; centralised DNN 90.52/93.00/88.00; CNN-LSTM 89.08/86/93; LSTM 76.69/79/73; SVM (from the dataset authors) 92.40. | Table III |
| 15 | **Results (ECU)** | DNN-FL 10 clients 98.47 % acc, 93.88 precision, 99.18 recall; the centralised models 94.5-94.7 % acc with 100 % recall. | Table IV |
| 16 | **Conclusion** | Federated learning gives "equivalent or best" performance and keeps data private; future work: CNN in FL, larger data sets. | V |

## 2. What the paper does NOT say (and what this reconstruction assumes)

| Missing detail | Our assumption (so the code can run) |
|---|---|
| Train/test split (ratio, method, stratified?) | Stratified 80/20 on real data, seed 42 - **the same test set as our `honest` protocol** (~12.5 % attack). Plus a 15 % validation slice from train. |
| Whether the **test set** was balanced | Real, imbalanced test set. Reason: from Table III's accuracy/precision/recall we can solve for the attack share of the test set (see section 3); the federated rows imply ~12 %. |
| How data is split across clients (IID?) | Random IID shards. |
| Whether undersampling happens before or after the split, and per client or globally | Each client undersamples **its own training shard** to 50/50 after the split. |
| Whether the scaler is fit per client or globally | One scaler fit on the real training part. |
| Whether "100 epochs" means local epochs per round | **No** (corrected after reading the authors' notebook, section 6): 100 is the number of rounds; local epochs are never set, so we use 1. |
| Server-side Adam details (betas, epsilon, what is "gradient") | PyTorch Adam defaults; the pseudo-gradient is (global - average of client weights), like TFF's `server_optimizer_fn`. |
| Optimizer and epochs of the centralised DNN | SGD lr 0.1, batch 20, 100 epochs. |
| Decision threshold | 0.5 (we also report a validation-tuned threshold). |
| Number of runs / seeds / error bars | None given. We use seeds 0, 1, 2 and report mean ± std. |
| How the metrics are computed in FL (global model or average of clients) | Global model on the shared test set. |
| What "train time 0.03-0.05 s" measures | Not interpretable for 100 rounds; we do not try to reproduce it. |

## 3. What the numbers imply

* **Attack share of each test set, back-solved** from accuracy *A*, precision *P*, recall *R*: errors = π·(1−R) + π·R·(1/P−1), so π = (1−A) / [(1−R) + R(1/P−1)].

  | Row | Implied attack share π |
  |---|---|
  | DNN-FL 15 / 10 / 5 clients | 13 % / 12 % / 11 % |
  | Centralised DNN, CNN-LSTM, LSTM | 51 % / 49 % / 50 % |

  So the federated models appear to have been scored on a **real, imbalanced** test set, and the centralised baselines on a **balanced** one. The paper's "federated is equivalent or better" comparison therefore mixes two different test sets.
* **Accuracy flatters the federated models.** With ~12 % attacks, "always normal" scores ~88 %, so 91.4 % is only ~3.5 points above it, and precision/recall (65 / 61) are far below the centralised DNN's (93 / 88).
* **F1 of the federated rows** (from the table): 15 clients 0.58, 10 clients 0.63, 5 clients 0.65. Our `honest`-protocol results for comparison: FedAvg DNN 0.69, FedAvg MoE 0.73 (3 seeds, tuned threshold).
* **ECU-IoHT:** the paper states 87,754 *normal* and 23,453 *attack* rows. Our Stage 1 notes recorded the opposite (87,754 attack) and parked ECU because of it. Worth re-checking before calling that dataset artificial. The paper also lists "type of attacks" among the 7 features, which would leak the label.

## 4. How the code maps to the paper

| Paper step | File |
|---|---|
| 4-7 split, clean, undersample, scale | `preprocessing.py` (uses `stages/s1_preprocessing`) |
| 8 model | `model.py` (`DNN2023`) |
| 9, 11 training (BCE, Adam; centralised DNN) | `training.py` |
| 9-10 federated rounds with server Adam | `federated.py` |
| 13-14 metrics and results table | `run.py` (uses `stages/s6_evaluation`) |

## 5. Is the 2023 paper clearer than the 2025 one?

**Mostly yes, with limits.** (This compares the 2023 text I read in full against the 2025 paper *as described by the replication audit and by our own methodology notes* in `docs/`; I did not re-read the 2025 paper for this note.)

| Question | 2023 paper | 2025 paper |
|---|---|---|
| Architecture | Layer sizes, activation, output and loss all stated (64-64-32, ReLU, sigmoid, BCE) | DNN drawn as "(8x8)"; CNN gives layer sizes but no kernel/stride/pooling; audit found the notebook has 26 features and no conv layers |
| Training hyper-parameters | One table (Table II): optimizers, learning rates, batch, rounds, clients | "100 epochs" and "100 rounds" ambiguous; clip 1e-4 vs 1.0 in code; local epochs 100 vs 1; client count unspecified vs 2 |
| Preprocessing order | Listed in order (split into clients, clean, undersample, scale) | Oversampling described, undersampling in code (audit) |
| Dataset | Sizes given (2,046 / 14,272; 36 features) and they add up to the public file | Reported test sets (up to 27,297 rows) are larger than the public data (16,318); depends on a private file |
| Privacy | None - nothing to verify | Differential privacy with an epsilon whose accounting the audit questions |
| Code | Link given; paper and architecture roughly agree | Notebook output matches Table 4 but settings differ from the text |
| Evaluation protocol | **Not stated** (split, seeds, test set) | Also not fully stated |

So the 2023 paper is clearer about **what the model and the training are**, and its numbers are close enough to a real, imbalanced test set that we can reconstruct it. It is still **not a complete specification**: the split, the test-set construction, the seeds and the aggregation details are missing, and its centralised-vs-federated comparison uses different test sets. It also says nothing about privacy, which is the 2025 paper's main claim.

## 6. What the authors' own notebook shows (read after the first build)

Source: `github.com/fatemehm/Federated-Learning-IDS`, notebook `DNN_FL_based_IDS.ipynb` (repo has 3 notebooks and a README, **no data files**). Read through a web fetch that summarised the first ~100k of ~1.2M characters, so treat the details as strong evidence, not a line-by-line audit. The printed final metrics (accuracy 0.906422, precision 0.70394737, recall 0.4976744, loss 0.3533) match the paper's Table III "15 clients" row exactly, and `num_examples: 1635` is the size of the evaluation set.

| Topic | Paper text | Authors' notebook | What our current reconstruction does | Status |
|---|---|---|---|---|
| Data file | public wustl-ehms-2020 | reads a pre-cleaned `clean_wustl.csv` from the authors' Google Drive (not public) | public CSV, our cleaning | cannot match exactly |
| Input features | 36 | **35** (an index column and the label are dropped from the file) | 36 | differs |
| Cleaning | NaN + duplicates removed | no `dropna` / `drop_duplicates` in the notebook (presumably done when the clean file was made) | dropna + dedupe (changes nothing; the public file has none) | same effect |
| "Epochs 100" | "Epochs (Federated model) 100" and "Training rounds 100" | `EPOCHS = 100` is the loop over **rounds**; local epochs never set | fixed: 1 local epoch | **corrected** |
| Client split | "divided the total data into 5, 10, 15 clients" | `np.array_split(df, 15)`: **contiguous chunks of the file**, each then shuffled | random IID shards | **to align** (contiguous chunks of a time-ordered capture are probably non-IID) |
| Balancing | undersample to balance | undersamples normals to **2 normal : 1 attack** (`negative_ratio=2`), training data only | 50 / 50 | **to align** |
| Scaling | StandardScaler | one StandardScaler **per client**, fit on that client's training part | one global scaler | **to align** |
| Train/test split | not stated | **per client 90 / 10** (`random_state=1337`); **no separate test set** | global 80 / 20 + 15 % validation | **to align** |
| Evaluation data | not stated | the pooled 10 % splits of all clients: real, **unbalanced** (1,635 rows) | our common 3,263-row real test set | explains why the federated rows look "real distribution" |
| Decision rule | not stated | Keras metrics at threshold 0.5 | 0.5 and a validation-tuned threshold | same at 0.5 |
| Server optimizer | Adam lr 0.05 | TFF FedAvg with `server_optimizer_fn` Adam 0.05 | same idea in PyTorch | ok |
| Training time | "0.033-0.045 s" | the timer wraps no work (prints ~5e-05 s) | not reproduced | paper value not meaningful |

Things the fetch summary flagged but we have **not verified**: a possible column-name misalignment (`columns.difference(['Label'])` sorts names alphabetically) and a metric (`AUC`) used without being imported.

**Consequences.** (1) The federated rows in Table III are scored on a real, imbalanced evaluation set; our earlier back-solved 11-13 % attack share is therefore confirmed. (2) Our reconstruction is not yet faithful on five points (client split, 2:1 balancing, per-client scaler, per-client 90/10 evaluation, 35 vs 36 features). A second mode, `--as-published`, should reproduce their exact pipeline, while the current mode stays as the common-test-set comparison.
