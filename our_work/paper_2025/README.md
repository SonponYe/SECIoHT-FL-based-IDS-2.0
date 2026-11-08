# Track Paper: improving the paper's own models and pipeline

Everything here improves what the paper already uses (DNN, CNN, FedAvg, DP-SGD). There is **no MoE** in this track.

Run from `our_work/`:

```bash
python -m track_paper.run_paper --stage optim --protocol honest --seeds 0 1 2
```

Flags: `--stage` (below, or `all`), `--protocol paper|honest|both`, `--seeds`, `--csv`, `--quick` (tiny run for smoke tests, writes to `results/quick/`), `--rounds`, `--local-epochs`, `--epochs`, `--grid full|small` (DP grid), `--threads`.
Results: `results/results.jsonl` (one line per run), `results/summary.csv` (mean ± std), `results/timing.jsonl`.

## Two protocols

* `paper` — SMOTE on the whole dataset, then scale, then split. Test set is ~50% attack. **Hypothesis:** this is what the paper did; stage P0 tests it.
* `honest` — split first, balance the training data only. Test set is real (~12.5% attack). **This is the protocol we trust.**

Every score is reported next to the "predict everything normal" accuracy (~87.5% honest, ~50% paper) because accuracy alone is misleading on this imbalanced data.

## Stages: what each changes compared with the paper, and why

| Stage | What it does | Why |
|---|---|---|
| **P0 `reproduce`** | The notebook's exact DNN recipe (plain SGD, lr 0.01, batch 64, 100 epochs, SMOTE) under **both** protocols. | If `paper` lands near the paper's 93% while `honest` does not, that is evidence for the leak explanation. |
| **P1 `optim`** | Same DNN: plain SGD vs SGD + momentum 0.9 vs Adam (lr 1e-3), each with SMOTE-on-train vs class weights. Keeps the best-**validation**-F1 checkpoint. | Plain SGD at lr 0.01 is slow to converge on this data; the balancing method changes what the model learns. |
| **P2 `threshold`** | Picks the best P1 setting by mean validation F1, writes it to `shared/best_recipe.json` (used by every later stage and by Track MoE), and prints the fixed-0.5 vs tuned-threshold scores. | After balancing, 0.5 is rarely the best cut-off. The threshold is tuned on a validation slice, never on the test set. |
| **P3 `cnn`** | The notebook's CNN with the best recipe. | Covers the paper's second model. |
| **P4 `fedavg`** | Federated DNN, 3 IID clients, best recipe. Honest: SMOTE/weights per client on its own shard. | Same federated setup as the paper with the improved local training. |
| **P5 `dp`** | Federated DNN + Opacus (persistent clients, accountant never reset), delta = 1e-4. Grid: noise 1.5 and 0.5 × clip 1e-4 / 0.1 / 1.0 with plain SGD, the recipe optimizer at clip 1.0, and **epsilon-matched** runs (Opacus picks the noise to reach epsilon 0.44). | The paper's clip of 1e-4 prevents learning with SGD (see the earlier near-zero recall). Clip values are crossed with SGD because Adam rescales gradients and makes the clip nearly irrelevant. The eps-matched run compares with the paper at the **same privacy budget**. |
| **P6 `gbt`** | Centralised HistGradientBoosting. **No FL, no DP — an upper bound, not a federated result.** | Shows how much headroom the neural models leave. |
| **P7 `fedgbt`** | Each client fits its own GBT; the server averages predicted probabilities. **No DP guarantee.** | Tests whether the strong tree model survives federation. |

## Reading the results

* Compare `@0.5` (fixed threshold) and `+thr` (validation-tuned) rows — the gap shows how much of an improvement is just threshold choice.
* Rows marked *no DP guarantee* or *no FL, no DP* must not be compared with DP rows as if they were private.
