"""S1d - splitting training data across simulated federated clients."""
import numpy as np


def iid_parts(X, y, seed, n=3):
    """Random equal-sized shards. Returns [(X_i, y_i), ...]."""
    idx = np.random.RandomState(seed).permutation(len(X))
    return [(X[i], y[i]) for i in np.array_split(idx, n)]


def partition_noniid(Xtr, ytr, Xte, yte, n_clients=6, alpha=0.3, n_domains=3, seed=0):
    """Domain-skewed split: k-means traffic clusters distributed with Dirichlet(alpha).

    The same per-domain proportions are applied to the test set so each client is scored on data
    from its own distribution.  Returns (train_idx_lists, test_idx_lists)."""
    from sklearn.cluster import KMeans
    km = KMeans(n_domains, n_init=5, random_state=seed).fit(Xtr)
    dtr, dte = km.labels_, km.predict(Xte)
    rng = np.random.RandomState(seed)
    for _ in range(200):
        tr = [[] for _ in range(n_clients)]
        te = [[] for _ in range(n_clients)]
        for d in range(n_domains):
            q = rng.dirichlet(alpha * np.ones(n_clients))
            for arr, out in ((dtr, tr), (dte, te)):
                idx = rng.permutation(np.where(arr == d)[0])
                cuts = (np.cumsum(q) * len(idx)).astype(int)[:-1]
                for c, part in enumerate(np.split(idx, cuts)):
                    out[c].extend(part.tolist())
        if all(len(tr[c]) >= 300 and len(te[c]) >= 50 and len(set(ytr[tr[c]])) == 2 and len(set(yte[te[c]])) == 2
               for c in range(n_clients)):
            return [np.array(t) for t in tr], [np.array(t) for t in te]
    raise RuntimeError("could not build a valid non-IID partition")
