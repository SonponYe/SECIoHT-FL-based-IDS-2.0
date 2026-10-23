# Comparing the two tracks

`compare.py` merges `track_paper/results/results.jsonl` and `track_moe/results/results.jsonl` into one table per protocol, and writes:

* `final_comparison.csv`
* `final_comparison.md` (with the paper's reported row, the "predict everything normal" row, and an auto-generated *Winners* section)

## Rerun everything, in order (copy-paste)

Run all commands from `our_work/`. On Colab, clone the repo first (the dataset CSV is tracked in git).

```bash
# 0. install (Colab already has torch/numpy/pandas/scikit-learn)
pip install opacus imbalanced-learn scipy

# 1. Track Paper: pick the best recipe first (P1 -> P2 writes shared/best_recipe.json)
python -m track_paper.run_paper --stage optim     --protocol both --seeds 0 1 2
python -m track_paper.run_paper --stage threshold --protocol both
python -m track_paper.run_paper --stage reproduce --protocol both --seeds 0 1 2

# 2. Everything else in Track Paper (needs the recipe from step 1)
python -m track_paper.run_paper --stage cnn     --protocol both --seeds 0 1 2
python -m track_paper.run_paper --stage fedavg  --protocol both --seeds 0 1 2
python -m track_paper.run_paper --stage gbt     --protocol both --seeds 0 1 2
python -m track_paper.run_paper --stage fedgbt  --protocol both --seeds 0 1 2
python -m track_paper.run_paper --stage dp      --protocol both --seeds 0 1 2   # slow: add --grid small to halve it

# 3. Track MoE (reads the same recipe)
python -m track_moe.run_moe --stage central   --protocol both --seeds 0 1 2
python -m track_moe.run_moe --stage fedavg    --protocol both --seeds 0 1 2
python -m track_moe.run_moe --stage dp        --protocol both --seeds 0 1 2     # slow
python -m track_moe.run_moe --stage noniid    --protocol both --seeds 0 1 2
python -m track_moe.run_moe --stage noniid_dp --protocol both --seeds 0 1 2

# 4. Merge
python -m compare.compare
```

Tips for long runs:

* Smoke test any stage first with `--quick --seeds 0` (results go to `results/quick/`, never to the real results; use `python -m compare.compare --quick` to check the table).
* The runs are independent per seed/protocol, so you can launch several processes at once, e.g. one per seed (`--seeds 0`, `--seeds 1`, `--seeds 2`) and `--threads 2` each. They append to the same `results.jsonl`.
* Commit `results/` and `compare/final_comparison.*` from Colab to bring the numbers back.
* **Do not** run `--stage threshold` in quick mode expecting it to save a recipe: quick mode never writes `shared/best_recipe.json`.
