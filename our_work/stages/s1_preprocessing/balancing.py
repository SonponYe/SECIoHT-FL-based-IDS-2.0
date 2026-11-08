"""S1b - handling the ~7:1 class imbalance."""
import numpy as np


def class_weights(y):
    """Inverse-frequency weights [w_normal, w_attack] for a weighted loss."""
    counts = np.bincount(y, minlength=2).astype(float)
    return len(y) / (2 * np.maximum(counts, 1))


def smote_resample(X, y, seed):
    """Oversample the attack class with synthetic points (SMOTE)."""
    from imblearn.over_sampling import SMOTE
    return SMOTE(random_state=seed).fit_resample(X, y)


def undersample(X, y, seed):
    """Drop random majority-class rows until both classes are equal (the 2023 paper's method)."""
    rng = np.random.RandomState(seed)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.concatenate([rng.choice(idx0, n, replace=False), rng.choice(idx1, n, replace=False)])
    keep = rng.permutation(keep)
    return X[keep], y[keep]
