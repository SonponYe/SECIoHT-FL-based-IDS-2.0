# our_work: project map

Everything we run lives here. The idea: **the pipeline is split into numbered building blocks (`stages/`), and every experiment ("approach") is just a thin script that calls those blocks in order.** To change how something works you edit one block; you never edit the notebook or the run script.

```
our_work/
  stages/                         SHARED BUILDING BLOCKS, in pipeline order
    s1_preprocessing/             loading.py  balancing.py  protocols.py  partitioning.py
    s2_models/                    dnn.py  cnn.py  registry.py
    s3_training/                  recipes.py (+ best_recipe.json)  optimizers.py  loops.py  centralized.py
    s4_federated/                 state.py  client.py  simulation.py   (rounds + FedAvg)
    s5_privacy/                   engine.py (Opacus wiring)  dp_settings.py (the DP grid)
    s6_evaluation/                metrics.py  reporting.py  summary.py

  baseline_2023/                  APPROACH: reconstruct the 2023 DNN-FL paper (own preprocessing, model, loss, aggregation)
  paper_2025/                     APPROACH: improve the 2025 paper's own models (optimizer, threshold, DP, gradient boosting)
  moe/                            APPROACH: Mixture-of-Experts + domain-aware aggregation (model.py, aggregation.py)
    each has:  run.py  README.md  results/ (results.jsonl, timing.jsonl, summary.csv)

  compare/                        merges all approaches into final_comparison.md / .csv
  notebooks/                      00_original_replication.ipynb (the class notebook, Stages 1-6)
                                  01_pipeline.ipynb  (calls the stage blocks in order; edit nothing here)
  kaggle/                         ready-made Kaggle notebooks + runbook for long runs
  data/                           WUSTL-EHMS-2020 CSV (tracked)
  archive/                        old exploratory runs (kept, not part of the pipeline)
```

(`references/` at the repo root holds the papers, the audit deck and the design document; PDFs are not committed.)

## The pipeline, in order

| Stage | Folder | Question it answers | Typical thing you change |
|---|---|---|---|
| S1 | `stages/s1_preprocessing` | How is raw data cleaned, balanced, scaled, split and shared out to clients? | the protocol, balancing method, non-IID split |
| S2 | `stages/s2_models` | What network are we training? | layer sizes; add a model to `registry.py` |
| S3 | `stages/s3_training` | How is it trained (optimizer, learning rate, epochs)? | the recipe in `best_recipe.json` |
| S4 | `stages/s4_federated` | How do clients train and how are results combined? | the aggregation rule |
| S5 | `stages/s5_privacy` | How is privacy added and measured? | noise, clip, delta, the DP grid |
| S6 | `stages/s6_evaluation` | How are results scored, logged and summarised? | metrics, thresholds, tables |

Approach-specific parts (things that differ from the shared blocks) sit in the approach folder: `moe/model.py`, `moe/aggregation.py`, and all of `baseline_2023/`.

## How to change something

| I want to... | Edit this |
|---|---|
| change the cleaning or the train/test split | `stages/s1_preprocessing/loading.py` or `protocols.py` |
| try another balancing method | `stages/s1_preprocessing/balancing.py` |
| change the DNN | `stages/s2_models/dnn.py` |
| change the training recipe | `stages/s3_training/best_recipe.json` (or `recipes.py`) |
| add a new aggregation rule | a new function like `fedavg_agg` (see `moe/aggregation.py`) |
| change the DP settings | `stages/s5_privacy/dp_settings.py` |
| add a metric | `stages/s6_evaluation/metrics.py` |
| add a whole new experiment | copy an approach folder; keep `run.py` as a list of calls into `stages/` |

## How to run

All commands run from this folder (`our_work/`).

```bash
pip install torch scikit-learn pandas numpy imbalanced-learn scipy opacus

python -m paper_2025.run   --stage optim --protocol honest --seeds 0 1 2
python -m moe.run          --stage central --protocol honest --seeds 0 1 2
python -m baseline_2023.run --stage all --clients 5 10 15 --seeds 0 1 2
python -m compare.compare                      # builds compare/final_comparison.md
```

Add `--quick` to any run for a tiny smoke test (writes to `results/quick/`, which is git-ignored). Long runs: see `kaggle/RUNBOOK.md`.

## Two evaluation protocols

* `honest` - split first on real data, balance only the training data. **The one we trust.**
* `paper` - a *balanced-leak imitation* (SMOTE on the whole dataset before the split). It does **not** reproduce the authors' test set (the audit found a larger, ~89 %-attack one); it only shows how an easy test set inflates scores.

Details: `stages/s1_preprocessing/protocols.py`.
