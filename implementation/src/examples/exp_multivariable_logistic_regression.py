import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt

from matplotlib.ticker import LogFormatterMathtext

from algorithms.datasets.libsvm_loader import load_mushrooms, load_w8a
from algorithms import GDA, GD, Nesterov


# -----------------------------
# 1) Label normalization
# -----------------------------
def to_binary01(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel().astype(np.float64)
    uniq = np.unique(y)
    if uniq.size != 2:
        raise ValueError(f"Expected binary labels, got {uniq}")

    s = set(uniq.tolist())
    if s == {1.0, 2.0}:
        return (y - 1.0).astype(np.int64)
    if s == {-1.0, 1.0}:
        return ((y + 1.0) * 0.5).astype(np.int64)
    if s == {0.0, 1.0}:
        return y.astype(np.int64)

    lo, hi = uniq[0], uniq[1]
    return (y == hi).astype(np.int64)


def to_pm1(b01: np.ndarray) -> np.ndarray:
    b01 = np.asarray(b01).astype(np.float64)
    return 2.0 * b01 - 1.0


# -----------------------------
# 2) Smoothness L
#   L = 0.25 * max_eig(X^T X / n) = 0.25 * ||X||_2^2 / n
# -----------------------------
def logistic_smoothness_L(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    s = np.linalg.norm(A, 2)  # spectral norm
    return 0.25 * (s * s) / n


# -----------------------------
# 3) Objective
#   f(w) = mean log(1+exp(-y a^T w)) + (l2/2)||w||^2
#   y in {-1,+1}
# -----------------------------
def make_objective(A: np.ndarray, y_pm1: np.ndarray, l2: float):
    A_ag = anp.array(np.asarray(A, dtype=np.float64))
    y_ag = anp.array(np.asarray(y_pm1, dtype=np.float64))

    def f(w: anp.ndarray) -> anp.float64:
        t = A_ag @ w
        loss = anp.mean(anp.logaddexp(0.0, -y_ag * t))
        reg = 0.5 * l2 * anp.dot(w, w)
        return loss + reg

    return f


# -----------------------------
# 4) Accuracy without exp overflow
# -----------------------------
def accuracy(A: np.ndarray, b01: np.ndarray, w: np.ndarray) -> float:
    logits = np.asarray(A, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
    pred01 = (logits >= 0.0).astype(np.int64)
    return float(np.mean(pred01 == b01))


# -----------------------------
# 5) Convergence iteration
# -----------------------------
def first_k_reach_eps(curve: np.ndarray, eps: float) -> int:
    idx = np.where(curve <= eps)[0]
    return int(idx[0]) if idx.size > 0 else -1


def format_vector(x: np.ndarray, max_len: int = 12) -> str:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size <= max_len:
        return np.array2string(x, precision=4, suppress_small=False)
    head = x[:max_len]
    return (
        np.array2string(head, precision=4, suppress_small=False) + f", ... (d={x.size})"
    )


# -----------------------------
# 6) Plot helper: match paper y-axis ticks
# -----------------------------
def apply_log_axis(ax):
    ticks = [1e1, 1e-2, 1e-5, 1e-8, 1e-11, 1e-14]
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 1e1)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(False, which="minor")


# -----------------------------
# 7) Run one dataset summary
# -----------------------------
def run_dataset(dataset: str, it_max=None, eps_for_conv: float = 1e-11):
    ds = dataset.lower().strip()
    if ds == "mushrooms":
        A, y_raw = load_mushrooms()
        if it_max is None:
            it_max = 4000
    elif ds == "w8a":
        A, y_raw = load_w8a(train=True)
        if it_max is None:
            it_max = 8000
    else:
        raise ValueError("dataset must be 'mushrooms' or 'w8a'")

    A = np.asarray(A, dtype=np.float64)
    b01 = to_binary01(y_raw)
    y_pm1 = to_pm1(b01)

    n, d = A.shape

    # notebook L and l2
    L = logistic_smoothness_L(A)
    l2 = L / (10.0 * n)
    f = make_objective(A, y_pm1, l2)

    print(f"\nDataset={dataset}  n={n}  d={d}  it_max={it_max}")
    print(f"L (logistic smoothness) = {L}")
    print(f"l2 (reg) = {l2}")

    w0 = np.zeros(d, dtype=np.float64)

    # GD with lr = 1/L
    gd = GD(function=f, bounds=None, constraints=[])
    res_gd = gd.solve(x0=w0, L=L, max_iter=int(it_max), stop_if_stationary=False)
    vals_gd = np.array([float(f(w)) for w in res_gd.history], dtype=np.float64)

    # Nesterov with lr = 1/L
    nest = Nesterov(function=f, bounds=None, constraints=[])
    res_ne = nest.solve(x0=w0, L=L, max_iter=int(it_max), stop_if_stationary=False)
    vals_ne = np.array([float(f(w)) for w in res_ne.history], dtype=np.float64)

    # GDA variants: lr0=1000, sigma=1/L, k in {0.75,0.85,0.95}
    gda_runs = []
    for kappa in (0.75, 0.85, 0.95):
        gda = GDA(function=f, bounds=None, constraints=[])
        res = gda.solve(
            x0=w0,
            lambda_0=1000.0,
            sigma=1.0 / L,
            kappa=kappa,
            max_iter=int(it_max),
            stop_if_stationary=False,
        )
        vals = np.array([float(f(w)) for w in res.history], dtype=np.float64)
        gda_runs.append((kappa, res, vals))

    # best found f*
    f_star = float(
        np.min(np.concatenate([vals_gd, vals_ne] + [vals for _, _, vals in gda_runs]))
    )

    # clamp to paper floor for plotting + convergence
    floor = 1e-14
    gd_curve = np.maximum(vals_gd - f_star, floor)
    ne_curve = np.maximum(vals_ne - f_star, floor)
    gda_curves = [
        (kappa, np.maximum(vals - f_star, floor)) for kappa, _, vals in gda_runs
    ]

    # ----------------SUMMARY ----------------
    print("\n=== SUMMARY ===")
    print(f"Dataset: {dataset}")
    print(f"f* (best found) = {f_star:.16e}")
    print(f"eps (for k_conv) = {eps_for_conv:.1e}")
    print("k_conv = smallest k with f(x^k)-f* <= eps\n")

    k_gd = first_k_reach_eps(gd_curve, eps_for_conv)
    k_ne = first_k_reach_eps(ne_curve, eps_for_conv)

    print("GD:")
    print(f"  k_conv = {k_gd}")
    print(f"  f(x*)  = {float(vals_gd[-1]):.16e}")
    print(f"  x*     = {format_vector(res_gd.x_opt)}\n")

    print("Nesterov:")
    print(f"  k_conv = {k_ne}")
    print(f"  f(x*)  = {float(vals_ne[-1]):.16e}")
    print(f"  x*     = {format_vector(res_ne.x_opt)}\n")

    for (kappa, res, vals), (_, curve) in zip(gda_runs, gda_curves):
        k_gda = first_k_reach_eps(curve, eps_for_conv)
        print(f"GDA (k={kappa}):")
        print(f"  k_conv = {k_gda}")
        print(f"  f(x*)  = {float(vals[-1]):.16e}")
        print(f"  x*     = {format_vector(res.x_opt)}\n")

    # extra: accuracy on same file
    print("Acc on full dataset file:")
    print("  GD      :", accuracy(A, b01, res_gd.x_opt))
    print("  Nesterov:", accuracy(A, b01, res_ne.x_opt))
    for kappa, res, _ in gda_runs:
        print(f"  GDA(k={kappa}):", accuracy(A, b01, res.x_opt))

    # ---------------- PLOT ----------------
    plt.figure(figsize=(7.6, 4.8), dpi=130)
    ax = plt.gca()
    apply_log_axis(ax)

    ax.plot(gd_curve, label="GD", linewidth=2.0)
    ax.plot(
        ne_curve,
        label="Nesterov",
        linewidth=2.0,
        marker="o",
        markevery=250,
        markersize=4,
    )
    for kappa, curve in gda_curves:
        ax.plot(
            curve,
            label=f"GDA (k={kappa})",
            linewidth=2.0,
            marker="*",
            markevery=250,
            markersize=5,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$f(w^k) - f_{\ast}$")
    ax.set_title(f"Example 5.2 logistic regression: {dataset}")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def main():
    run_dataset("mushrooms", it_max=4000, eps_for_conv=1e-11)
    # run_dataset("w8a", it_max=8000, eps_for_conv=1e-11)


if __name__ == "__main__":
    main()
