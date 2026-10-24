# Runbook: running all the experiments on Kaggle (for a teammate)

**Read this top to bottom once, then follow the steps in order.** Do not skip the checkpoints - they stop you from wasting hours on a run that was going to fail.

## 1. What this is and what you will produce

We are testing whether we can beat the published paper's results (93.2% accuracy for a private federated intrusion detector). Two lines of work ("tracks") are compared on the same data:

* **Track Paper** - improves the paper's own models (better training settings, fair privacy settings, gradient boosting).
* **Track MoE** - tests a "Mixture of Experts" model (a small panel of specialist networks) from our design document.

Every experiment runs under two data setups called **protocols**: `paper` (copies the way we think the paper prepared its data) and `honest` (our leak-free version). You do not need to understand them to run the notebooks.

**Your job:** run 7 Kaggle notebooks in the right order, download one zip file from each, and merge them into one final comparison table. Total compute is roughly 15-25 hours, but it runs in the background and you do not need to keep the browser open.

## 2. Before you start (one-time setup)

- [ ] A Kaggle account at kaggle.com, signed in.
- [ ] Your **phone number verified** on Kaggle (Settings -> Phone verification). Without it you cannot turn Internet on, and the notebooks cannot download the code.
- [ ] The repo on your computer. Either `git clone https://github.com/SonponYe/SECIoHT-FL-based-IDS-2.0.git` or download it as a ZIP from GitHub (Code -> Download ZIP). You need the folder `our_work/kaggle/` (the 7 `.ipynb` files).
- [ ] Python 3 on your computer (only needed for the merge steps at the end) with `pandas` installed: `pip install pandas`.
- [ ] Know who will **push to GitHub**. Only someone with write access to the repo can. If that is not you, you will send the files to Sonpon instead (the steps say where).

## 3. The order, at a glance

| Step | Notebook | What it does | Rough time | Must finish before |
|---|---|---|---|---|
| **A** | `00_recipe` | Picks the best training settings and runs the data-leak test | 1-2 h | everything else |
| **B** | *(you)* | Merge + push `best_recipe.json` | 10 min | step C |
| **C** | `01_paper_no_dp` | Track Paper without privacy noise | 2-3 h | step D |
| **C** | `02_paper_dp` | Track Paper with privacy noise | 2-3 h | step D |
| **C** | `03_moe_no_dp` | Track MoE without privacy noise | 2-3 h | step D |
| **C** | `04_moe_dp_honest` | Track MoE with privacy, honest data | 3-5 h | step D |
| **C** | `05_moe_dp_paper` | Track MoE with privacy, paper data | 3-5 h | step D |
| **C** | `06_moe_noniid_dp` | Track MoE, uneven hospitals, with privacy | 1-2 h | step D |
| **D** | *(you)* | Merge all zips and build the final table | 10 min | done |

Times are rough guesses. **Notebooks in step C can run at the same time** (if Kaggle lets you start several), in any order. **Step A must be completely finished and step B done before you start any notebook in step C.**

## 4. Step A - run `00_recipe` (do this first, alone)

1. Go to kaggle.com -> **Create** -> **New Notebook**.
2. **File -> Import Notebook** -> upload `our_work/kaggle/00_recipe.ipynb`.
3. In the **right sidebar** (open it with the small arrow if hidden), set:
   - **Internet: On**
   - **Accelerator: None** (CPU only - a GPU does not help and wastes your quota)
4. Click **Save Version** (top right) -> choose **Save & Run All (Commit)** -> **Save**.
   This runs it in the background. You can close the browser.
5. Check progress later: open the notebook -> **Versions** (or the Activity/Viewer tab). A green tick means finished; a red cross means failed.

**Checkpoint A - how to know it worked:**
- The version shows a green tick.
- Open the version -> **Output** tab. You should see `results_00_recipe.zip`.
- Open the notebook's log view. Near the end you should see lines starting with `P2/paper:` and `P2/honest:`, each followed by `-> best = P1 ...` and a recipe like `{'opt': 'adam', 'lr': 0.001, ...}`, then the text `saved /kaggle/working/results_00_recipe.zip`.

**If it failed (red cross):** open the version log and read the last red lines.
- `git clone` / "could not resolve host" -> Internet was off. Turn it on (needs phone verification) and run again.
- Anything else -> copy the last 30 lines of the log and send them to Sonpon. Do not guess.

6. **Download** `results_00_recipe.zip` from the Output tab into a folder, e.g. `C:\kaggle_results\`.

## 5. Step B - hand the recipe to the other notebooks (critical)

The other six notebooks read a file called `best_recipe.json` from GitHub. It does not exist there until you put it there.

1. Open a terminal in the repo folder, then:
   ```bash
   cd our_work
   python kaggle/merge_results.py C:\kaggle_results\results_00_recipe.zip
   ```
   You should see `copied shared/best_recipe.json` and some `+N lines` messages.
2. Open `our_work/shared/best_recipe.json` and check it has two entries, `"honest"` and `"paper"`, each with an `"opt"`, `"lr"`, `"balance"`, `"batch"`, `"epochs"`.
3. (Optional but useful) see the first results - the data-leak test (P0):
   ```bash
   python -m compare.compare
   ```
   Do not worry about the long output. Just look for rows named `P0 notebook recipe`: the `paper` protocol should score much higher than `honest`. Save the output; Sonpon will want it.
4. **Push** the recipe so Kaggle can see it:
   ```bash
   cd ..
   git add our_work/shared/best_recipe.json
   git commit -m "Add best training recipe from Kaggle notebook 00"
   git push origin main
   ```
   If you do not have push access, **send `best_recipe.json` and the zip to Sonpon** and wait for them to say it is pushed.

**Checkpoint B:** open https://github.com/SonponYe/SECIoHT-FL-based-IDS-2.0/blob/main/our_work/shared/best_recipe.json in your browser. If you see the file, go on. If you get a 404, the push did not work - **do not start step C.**

## 6. Step C - run notebooks 01 to 06

Repeat this for **each** of: `01_paper_no_dp`, `02_paper_dp`, `03_moe_no_dp`, `04_moe_dp_honest`, `05_moe_dp_paper`, `06_moe_noniid_dp`.

1. New Notebook -> File -> Import Notebook -> upload the `.ipynb`.
2. Sidebar: **Internet On**, **Accelerator None**.
3. **Save Version -> Save & Run All (Commit)**.
4. When it finishes (green tick), open the version's **Output** tab and download `results_<notebook name>.zip` (for example `results_02_paper_dp.zip`) into `C:\kaggle_results\`.

**Checkpoint C (for each notebook):**
- Green tick.
- The zip exists in the Output tab.
- The log ends with `saved /kaggle/working/results_...zip`.
- No line says `failed - see log_...`.

**If a notebook stops immediately with** `shared/best_recipe.json is missing` -> step B was not finished. Do step B, then run it again.

**If it runs out of time** (Kaggle stops sessions after about 12 hours) -> tell Sonpon. The fix is a smaller privacy grid (see the Kaggle README), and it needs a one-line edit.

**Important rules for step C**
- **Download one zip per notebook, from the run that fully succeeded.** If you ran a notebook twice, use only the latest good one. Merging two zips of the same notebook would count the same experiments twice and distort the averages.
- Do not edit any code in the notebooks.
- Do not paste any password or GitHub token into a notebook.

## 7. Step D - merge everything and build the final table

When all six zips are in `C:\kaggle_results\` (plus the one from step A):

```bash
cd our_work
python kaggle/merge_results.py C:\kaggle_results\results_01_paper_no_dp.zip C:\kaggle_results\results_02_paper_dp.zip C:\kaggle_results\results_03_moe_no_dp.zip C:\kaggle_results\results_04_moe_dp_honest.zip C:\kaggle_results\results_05_moe_dp_paper.zip C:\kaggle_results\results_06_moe_noniid_dp.zip
python -m compare.compare
```

(Do **not** merge `results_00_recipe.zip` a second time; it was already merged in step B. If you do, nothing breaks - duplicate lines are skipped - but there is no need.)

**Checkpoint D:**
- `merge_results.py` printed `+N lines` for `track_paper/results/results.jsonl` and `track_moe/results/results.jsonl`.
- `python -m compare.compare` finished without an error and created `our_work/compare/final_comparison.md` and `final_comparison.csv`.
- Open `final_comparison.md`. It should have two sections, **Protocol: honest** and **Protocol: paper**, each with tables headed *Centralized*, *Federated (no DP)*, *Federated + DP*, *Non-IID*, and a *Winners* list.

Then push the results (or send them to Sonpon):

```bash
cd ..
git add our_work/track_paper/results our_work/track_moe/results our_work/compare/final_comparison.md our_work/compare/final_comparison.csv
git commit -m "Add full experiment results and final comparison"
git push origin main
```

## 8. What to send back to Sonpon

- The link to the pushed commit (or the files if you cannot push): `final_comparison.md`, `final_comparison.csv`.
- A list of any notebook that failed or needed a re-run, and why.
- Anything strange you noticed (a notebook that was much slower than the table above, a result table that is mostly empty, etc.).

## 9. Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| Red cross, log mentions `could not resolve host` or `git clone` failed | Internet is off in the notebook | Sidebar -> Internet On (needs phone verification), run again |
| Stops with `shared/best_recipe.json is missing` | Step B not done or not pushed | Finish step B, check the file on GitHub, run again |
| `failed - see log_...` | One seed's process crashed | Open the `log_*` file named in the message and send its last 30 lines to Sonpon |
| Notebook has been running for more than 12 hours | Kaggle limit | Tell Sonpon, do not wait |
| `merge_results.py` says `+0 lines` | The zip was already merged or is empty | Check you picked the right zip; this is harmless if you already merged it |
| `compare.compare` says `warning: ... results.jsonl missing` | A track has no results yet | Merge the missing zips first |
| `ModuleNotFoundError: pandas` on your computer | pandas not installed | `pip install pandas` |

## 10. Final checklist

- [ ] A: `00_recipe` ran, zip downloaded
- [ ] B: recipe merged, `best_recipe.json` is on GitHub (checked in the browser)
- [ ] C: six notebooks ran, six zips downloaded (one each)
- [ ] D: all zips merged, `final_comparison.md` created
- [ ] Results pushed (or sent to Sonpon)
- [ ] Report of any problems sent

## 11. Words you will see

- **Protocol** - how the data is prepared before testing (`paper` vs `honest`).
- **Federated** - several simulated hospitals train separately and only share model weights.
- **DP / differential privacy** - noise added so the model reveals little about any single patient. **Epsilon** is the privacy cost (smaller = more private); the paper reports 0.44.
- **MoE** - mixture of experts, a model made of several small specialist networks.
- **Seed** - a random starting point. We repeat every experiment with seeds 0, 1, 2 and report the average.
- **F1 / AUC** - scores that are fairer than accuracy when attacks are rare.
