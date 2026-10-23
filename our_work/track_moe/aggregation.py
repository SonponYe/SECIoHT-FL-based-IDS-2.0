"""Aggregation strategies for the MoE (plug into shared.fl.run_fl as `aggregator`).

domain_aggregator  Domain-aware aggregation from the design document:
    * embedding            -> averaged across ALL clients (weight = sample count)
    * gating network       -> averaged within each domain cluster
    * expert i             -> averaged within the cluster over clients whose routing share to expert i
                              is >= 5%, weight = samples x share
    Clusters come from each client's descriptor (mean gate-probability vector) via average-linkage
    clustering on cosine distance, recomputed every `recluster` rounds and cached in between.
    NOTE: the design adds clipped Gaussian noise to descriptors and charges it to the privacy
    budget; that step is NOT implemented here (descriptors are un-noised) -- stated in the README.
local_aggregator   No communication: every client keeps its own trained weights (lower bound).
"""
import numpy as np
import torch

from shared.fl import wavg


def moe_groups(model, K=4):
    names = [k for k, _ in model.named_parameters()]
    g = {"emb": [k for k in names if k.startswith("emb.")], "gate": [k for k in names if k.startswith("gate.")]}
    for i in range(K):
        g[f"e{i}"] = [k for k in names if k.startswith(f"experts.{i}.")]
    return g


def local_aggregator(rnd, clients, states, ns, ctx):
    return [dict(s) for s in states], {}


def domain_aggregator(cut=0.02, recluster=2, share_thr=0.05, K=4):
    from scipy.cluster.hierarchy import fcluster, linkage
    cache = {"clusters": None, "groups": None}

    def agg(rnd, clients, states, ns, ctx):
        if cache["groups"] is None:
            cache["groups"] = moe_groups(ctx["make_model"](), K)
        g = cache["groups"]
        stats = [c.base.routing_stats(torch.tensor(c.X, dtype=torch.float32)) for c in clients]
        desc = np.stack([s[0] for s in stats])
        share = np.stack([s[1] for s in stats])
        if cache["clusters"] is None or rnd % recluster == 1:
            cache["clusters"] = (fcluster(linkage(desc, "average", metric="cosine"), t=cut, criterion="distance") - 1
                                 if len(clients) > 1 else np.zeros(len(clients), dtype=int))
        clusters = cache["clusters"]
        emb = wavg(states, ns, g["emb"])
        new = [dict(s) for s in states]
        for cl in np.unique(clusters):
            mem = [i for i in range(len(clients)) if clusters[i] == cl]
            gate = wavg([states[i] for i in mem], [ns[i] for i in mem], g["gate"])
            for i in mem:
                new[i].update(gate)
            for e in range(K):
                act = [i for i in mem if share[i, e] >= share_thr]
                if act:
                    agg_e = wavg([states[i] for i in act], [ns[i] * share[i, e] for i in act], g[f"e{e}"])
                    for i in act:
                        new[i].update(agg_e)
        for s_ in new:
            s_.update(emb)
        return new, dict(n_clusters=int(len(np.unique(clusters))))

    return agg
