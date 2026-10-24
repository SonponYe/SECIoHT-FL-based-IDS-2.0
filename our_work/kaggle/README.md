# Running the experiments on Kaggle

Seven ready-made notebooks. Each one clones this (public) repo, installs `opacus imbalanced-learn scipy`, runs its stages with **one process per seed**, and zips the results.

| Notebook | What it runs | Rough time* |
|---|---|---|
| `00_recipe` | **Run first.** Track Paper P1 (optimizer × balancing), P2 (pick best recipe), P0 (leak test) | 1–2 h |
| `01_paper_no_dp` | Track Paper: CNN, FedAvg, gradient boosting, federated GBT | 2–3 h |
| `02_paper_dp` | Track Paper: DP grid (P5) | 2–3 h |
| `03_moe_no_dp` | Track MoE: central, FedAvg, non-IID comparison | 2–3 h |
| `04_moe_dp_honest` | Track MoE: DP grid, honest protocol | 3–5 h |
| `05_moe_dp_paper` | Track MoE: DP grid, paper protocol | 3–5 h |
| `06_moe_noniid_dp` | Track MoE: non-IID + DP | 1–2 h |

\*Rough guesses scaled from our laptop smoke test. Check the real speed from the first notebook's logs and adjust. Kaggle sessions stop after about 12 h, so if one is too slow, re-run it with `--grid small` (edit the `run_parallel(...)` line to add `extra=("--grid", "small")`).

## Steps

1. **Account:** sign in at kaggle.com and verify your phone number (needed to turn Internet on for notebooks).
2. **Import:** *Create → New Notebook → File → Import Notebook* and upload `00_recipe.ipynb` (from `our_work/kaggle/` in this repo).
3. **Settings (right sidebar):** *Internet = On*, *Accelerator = None* (CPU only; a GPU does not help here).
4. **Run in the background:** *Save Version → Save & Run All (Commit)*. You can close the browser. Check progress under the notebook's *Versions* tab.
5. **Download** the output file `results_00_recipe.zip` from the finished version's *Output* tab.
6. **Hand the recipe back (important):** the other notebooks need `shared/best_recipe.json`. Merge the zip locally (below), then commit and push `our_work/shared/best_recipe.json`. The other notebooks clone the repo, so they pick it up.
7. **Run notebooks 01–06** the same way (they can run at the same time if Kaggle lets you; otherwise one after another). Each stops with a clear message if the recipe file is missing.
8. **Merge everything locally** (from `our_work/`):

   ```bash
   python kaggle/merge_results.py path/to/results_*.zip
   python -m compare.compare
   ```

   `merge_results.py` appends the new lines into `track_*/results/results.jsonl` (skipping duplicates) and copies the recipe file. Then `compare.compare` writes `compare/final_comparison.md` and `.csv`.

## Good to know

* **Do not paste a GitHub token into a notebook.** Results come back as zip downloads and get committed from your computer.
* If a notebook fails, open the `log_*.log` files it printed in its output to see which seed or stage failed. Re-running appends duplicates of finished runs, but the comparison averages per seed, so delete the failed notebook's partial lines first, or re-run only the failed stage.
* The notebooks always clone the latest `main`. If you change the code, push it before starting a notebook.
