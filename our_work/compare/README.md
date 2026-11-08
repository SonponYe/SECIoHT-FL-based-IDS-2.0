# Comparing the two tracks

`compare.py` merges `baseline_2023/`, `paper_2025/` and `moe/` `results/results.jsonl` into one table per protocol, and writes:

* `final_comparison.csv`
* `final_comparison.md` (with the paper's reported row, the "predict everything normal" row, and an auto-generated *Winners* section)

## Rerun everything, in order (copy-paste)

Run all commands from `our_work/`. On Colab, clone the repo first (the dataset CSV is tracked in git).

```bash
# 0. install (Colab already has torch/numpy/pandas/scikit-learn)
pip install opacus imbalanced-learn scipy

# 1. Track Paper: pick the best recipe first (P1 -> P2 writes stages/s3_training/best_recipe.json)
python -m paper_2025.run --stage optim     --protocol both --seeds 0 1 2
python -m paper_2025.run --stage threshold --protocol both
python -m paper_2025.run --stage reproduce --protocol both --seeds 0 1 2

# 2. Everything else in Track Paper (needs the recipe from step 1)
python -m paper_2025.run --stage cnn     --protocol both --seeds 0 1 2
python -m paper_2025.run --stage fedavg  --protocol both --seeds 0 1 2
python -m paper_2025.run --stage gbt     --protocol both --seeds 0 1 2
python -m paper_2025.run --stage fedgbt  --protocol both --seeds 0 1 2
python -m paper_2025.run --stage dp      --protocol both --seeds 0 1 2   # slow: add --grid small to halve it

# 3. Track MoE (reads the same recipe)
python -m moe.run --stage central   --protocol both --seeds 0 1 2
python -m moe.run --stage fedavg    --protocol both --seeds 0 1 2
python -m moe.run --stage dp        --protocol both --seeds 0 1 2     # slow
python -m moe.run --stage noniid    --protocol both --seeds 0 1 2
python -m moe.run --stage noniid_dp --protocol both --seeds 0 1 2

# 4. 2023 paper reconstruction (independent of the recipe)
python -m baseline_2023.run --stage all --clients 5 10 15 --seeds 0 1 2

# 5. Merge
python -m compare.compare
```

Tips for long runs:

* Smoke test any stage first with `--quick --seeds 0` (results go to `results/quick/`, never to the real results; use `python -m compare.compare --quick` to check the table).
* The runs are independent per seed/protocol, so you can launch several processes at once, e.g. one per seed (`--seeds 0`, `--seeds 1`, `--seeds 2`) and `--threads 2` each. They append to the same `results.jsonl`.
* Commit `results/` and `compare/final_comparison.*` from Colab to bring the numbers back.
* **Do not** run `--stage threshold` in quick mode expecting it to save a recipe: quick mode never writes `stages/s3_training/best_recipe.json`.
