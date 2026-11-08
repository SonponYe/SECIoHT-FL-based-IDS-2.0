# Track MoE: the design document's Mixture-of-Experts applied to WUSTL

This track takes the model and the aggregation idea from `MoE-FL_Gateway_Design_Document.docx` and tests them on WUSTL-EHMS-2020, with the **same data, split, recipe, rounds, local epochs and DP settings** as Track Paper so the two can be compared directly.

Run from `our_work/`:

```bash
python -m track_moe.run_moe --stage central --protocol honest --seeds 0 1 2
```

Flags are the same as Track Paper's. Results go to `results/results.jsonl` and `results/summary.csv`.

## The model in plain language

Think of it as a small panel of specialists:

1. An **embedding** layer turns the 36 features into a 64-number summary of the sample.
2. A **gate** looks at that summary and decides which 2 of the 4 **experts** should look at the sample.
3. The two chosen experts each give an attack/normal opinion; the answer is their weighted blend.
4. During training a small **balancing penalty** (weight 0.01) stops the gate from sending everything to one expert, and a little noise on the gate encourages exploration.

Code: `models_moe.py` (copied unchanged from `experiments/common.py`). All 4 experts run on every sample and are masked by the gate; this is needed so Opacus can compute per-sample gradients. Under DP the balancing penalty is **switched off** (weight 0) because it averages over the batch and would couple samples, which breaks the per-sample privacy analysis.

For every MoE run we also save the **expert usage share** and the **routing entropy** (1 = routing spread evenly, 0 = one expert does everything) to show whether the experts specialise or collapse.

## Recipe, so the tracks cannot drift apart

Optimizer, learning rate, balancing, batch size and epochs come from `shared/best_recipe.json`, written by Track Paper stage P2. The MoE therefore gets exactly the recipe that was chosen for the DNN (slightly conservative for the MoE, since it was not tuned for it).

## Stages

| Stage | What it runs |
|---|---|
| **M1 `central`** | Centralised MoE, both protocols, fixed and validation-tuned threshold. |
| **M2 `fedavg`** | FedAvg MoE, 3 IID clients, same rounds and local epochs as Track Paper P4. |
| **M3 `dp`** | FedAvg MoE + Opacus, **exactly** Track Paper P5's grid (including the epsilon-matched 0.44 run). |
| **M4 `noniid`** | 6 clients, domain-skewed split (k-means traffic clusters + Dirichlet 0.3). Compares FedAvg DNN, FedAvg MoE, **domain-aware MoE** (embedding averaged globally; gate and experts averaged within clusters; personalised evaluation) and **local-only MoE** (lower bound). All at threshold 0.5. |
| **M5 `noniid_dp`** | The M4 comparison with DP (noise 1.5, clip 1.0, delta 1e-4). |

Domain-aware aggregation lives in `aggregation.py`. **Limitation:** the design noises the routing descriptors and charges them to the privacy budget; that step is *not* implemented, so in M5 the domain-aware rows are labelled "weights only; descriptors un-noised".

## What was taken from the design document, and what was left out

Taken: the embedding + gate + top-2 experts structure, load-balancing loss and gate noise, domain-aware aggregation by parameter group, ghost-clipping-style full-batch expert masking, Opacus DP, the non-IID evaluation idea and the metrics (per-class detection, expert utilisation entropy, epsilon vs F1).

Left out on purpose — these belong to the **application write-up**, not to WUSTL: the FastAPI reverse-proxy gateway, Flower orchestration, the 24 API-traffic features at 3 window scales, the 4-class output (WUSTL is binary), and the latency / bytes-per-round / peak-memory measurements.

Mapping to the document's evaluation plan: macro-F1 and detection rate → accuracy/precision/recall/F1/AUC; expert utilisation entropy → `expert_share`, `routing_entropy`; epsilon vs F1 → M3/M5; the four baselines → M1 (centralised MoE), M4 FedAvg MoE, M4 FedAvg DNN, M4 local-only.
