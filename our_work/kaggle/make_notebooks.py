"""Generate the Kaggle notebooks (one per chunk of work).  Run from anywhere:  python make_notebooks.py

Each notebook: clone the repo -> install deps -> run its stages with one process per seed -> zip the results.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = '''import os, subprocess, sys
REPO = "https://github.com/SonponYe/SECIoHT-FL-based-IDS-2.0.git"
os.chdir("/kaggle/working")
if not os.path.exists("repo"):
    subprocess.run(["git", "clone", "--depth", "1", REPO, "repo"], check=True)
else:
    subprocess.run(["git", "-C", "repo", "pull"], check=True)
os.chdir("/kaggle/working/repo/our_work")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opacus", "imbalanced-learn", "scipy"], check=True)
print("CPU cores:", os.cpu_count())
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)'''

NEED_RECIPE = '''# Every stage except the recipe notebook needs the recipe chosen by 00_recipe.
import os
assert os.path.exists("shared/best_recipe.json"), (
    "shared/best_recipe.json is missing. Run notebook 00_recipe first, commit+push the file, then re-run this one.")
print(open("shared/best_recipe.json").read())'''

HELPER = '''import subprocess, sys, time

def run_parallel(module, stage, protocol, seeds=(0, 1, 2), extra=()):
    """One process per seed (1 thread each) so the 4 cores stay busy. Results append to results.jsonl."""
    t0, procs = time.time(), []
    for s in seeds:
        cmd = [sys.executable, "-m", module, "--stage", stage, "--protocol", protocol,
               "--seeds", str(s), "--threads", "1", *extra]
        log = open(f"log_{module.split('.')[-1]}_{stage}_{protocol}_s{s}.log", "w")
        procs.append((s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
    for s, p in procs:
        rc = p.wait()
        print(f"{module} {stage} {protocol} seed {s}: exit {rc}")
        assert rc == 0, f"failed - see log_{module.split('.')[-1]}_{stage}_{protocol}_s{s}.log"
    print(f"{stage}/{protocol} done in {(time.time()-t0)/60:.1f} min")'''

PACK = '''import zipfile, glob
out = "/kaggle/working/results_NAME.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ["track_paper/results/*", "track_moe/results/*", "shared/best_recipe.json", "log_*.log"]:
        for f in glob.glob(pat):
            if os.path.isfile(f):
                z.write(f)
print("saved", out, os.path.getsize(out) // 1024, "KB")'''

P, M = "track_paper.run_paper", "track_moe.run_moe"
NOTEBOOKS = {
    "00_recipe": ("Pick the training recipe (Track Paper P1+P2) and run the leak test (P0). RUN THIS FIRST.", False, [
        (P, "optim", "both"), "THRESHOLD", (P, "reproduce", "both")]),
    "01_paper_no_dp": ("Track Paper without privacy noise: CNN, FedAvg, gradient boosting, federated GBT.", True, [
        (P, "cnn", "both"), (P, "fedavg", "both"), (P, "gbt", "both"), (P, "fedgbt", "both")]),
    "02_paper_dp": ("Track Paper with differential privacy (P5, full grid).", True, [(P, "dp", "both")]),
    "03_moe_no_dp": ("Track MoE without privacy noise: central, FedAvg, non-IID comparison.", True, [
        (M, "central", "both"), (M, "fedavg", "both"), (M, "noniid", "both")]),
    "04_moe_dp_honest": ("Track MoE with differential privacy, honest protocol (M3).", True, [(M, "dp", "honest")]),
    "05_moe_dp_paper": ("Track MoE with differential privacy, paper protocol (M3).", True, [(M, "dp", "paper")]),
    "06_moe_noniid_dp": ("Track MoE non-IID with differential privacy (M5).", True, [(M, "noniid_dp", "both")]),
}


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


for name, (desc, needs_recipe, steps) in NOTEBOOKS.items():
    cells = [cell("markdown", f"# {name}\n\n{desc}\n\nSettings (right sidebar): **Internet = On**, Accelerator = **None** (CPU). "
                              "Then *Save Version -> Save & Run All (Commit)* so it runs in the background."),
             cell("code", SETUP)]
    if needs_recipe:
        cells.append(cell("code", NEED_RECIPE))
    cells.append(cell("code", HELPER))
    for st in steps:
        if st == "THRESHOLD":
            cells.append(cell("code", 'subprocess.run([sys.executable, "-m", "track_paper.run_paper", "--stage", "threshold", '
                                      '"--protocol", "both"], check=True)\nprint(open("shared/best_recipe.json").read())'))
        else:
            mod, stage, proto = st
            cells.append(cell("code", f'run_parallel("{mod}", "{stage}", "{proto}")'))
    cells.append(cell("code", PACK.replace("NAME", name)))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (HERE / f"{name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf8")
    print("wrote", f"{name}.ipynb")
