"""S3a - training recipes.

A *recipe* is a dict: {"opt": "sgd"|"momentum"|"adam", "lr": float, "balance": "smote"|"weights",
"batch": int, "epochs": int}.  paper_2025 stage `threshold` picks the best recipe per protocol on the
validation slice and writes best_recipe.json (next to this file); the MoE track reads it so the
two cannot drift apart.
"""
import json
from pathlib import Path

RECIPE_PATH = Path(__file__).resolve().parent / "best_recipe.json"
DEFAULT_RECIPE = {"opt": "adam", "lr": 1e-3, "balance": "weights", "batch": 64, "epochs": 60}


def load_recipe(protocol):
    if RECIPE_PATH.exists():
        r = json.loads(RECIPE_PATH.read_text()).get(protocol)
        if r:
            return dict(r)
    return dict(DEFAULT_RECIPE)


def save_recipe(protocol, recipe, note=""):
    cur = json.loads(RECIPE_PATH.read_text()) if RECIPE_PATH.exists() else {}
    cur[protocol] = recipe
    cur["note"] = note or "selected by mean validation F1 (threshold-tuned) in paper_2025 stage `threshold`"
    RECIPE_PATH.write_text(json.dumps(cur, indent=2))
