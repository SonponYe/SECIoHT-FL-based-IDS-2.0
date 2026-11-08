"""S5b - the DP settings run by EVERY approach (identical, so comparisons are fair)."""

DP_DELTA = 1e-4          # the 2025 paper's delta
PAPER_EPS = 0.44         # the 2025 paper's reported epsilon


def dp_grid(grid, recipe):
    """List of DP configurations.

    Clip values are crossed with the paper's plain SGD, because Adam rescales gradients and makes
    the clip norm almost irrelevant; the recipe optimizer is added at clip 1.0.  The epsilon-matched
    runs let Opacus choose the noise to hit the paper's epsilon 0.44 at delta 1e-4.
    grid: "full" (noise 1.5/0.5 x clip 1e-4/0.1/1.0) or "small" (noise 1.5 x clip 1e-4/1.0)."""
    sgd = dict(opt="sgd", lr=0.01)
    rec = dict(opt=recipe["opt"], lr=recipe["lr"])
    noises, clips = ((1.5, 0.5), (1e-4, 0.1, 1.0)) if grid == "full" else ((1.5,), (1e-4, 1.0))
    out = []

    def add(name, o, noise, clip, target=None):
        out.append(dict(setting=name, opt=o["opt"], lr=o["lr"], noise=noise, clip=clip, target_eps=target,
                        delta=DP_DELTA))

    for n in noises:
        for c in clips:
            add(f"DP noise={n} clip={c:g} opt=sgd", sgd, n, c)
    if rec["opt"] != "sgd":
        for n in noises:
            add(f"DP noise={n} clip=1 opt={rec['opt']}", rec, n, 1.0)
    add(f"DP eps-matched {PAPER_EPS} clip=1 opt=sgd", sgd, None, 1.0, PAPER_EPS)
    if rec["opt"] != "sgd":
        add(f"DP eps-matched {PAPER_EPS} clip=1 opt={rec['opt']}", rec, None, 1.0, PAPER_EPS)
    return out
