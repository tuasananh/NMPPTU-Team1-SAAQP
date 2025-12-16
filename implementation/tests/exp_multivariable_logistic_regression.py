import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt

from algorithms.datasets.libsvm_loader import load_mushrooms, load_w8a
from algorithms import GDA, GD, Nesterov


# ----- 1. Logistic loss + regularization (numerically stable) -----


def make_regularized_logistic_loss(A: np.ndarray, b: np.ndarray):
    """
    Build the regularized logistic regression objective in a numerically
    stable way.

    We use the standard binary logistic loss with labels in {-1, 1}:

        l_i(x) = log(1 + exp(-y_i * a_i^T x))

    where y_i = 2 * b_i - 1, b_i in {0, 1}.

    The overall objective is:

        J_bar(x) = sum_i l_i(x) + (1 / (2N)) * ||x||^2

    A: shape (N, d)
    b: shape (N,), labels in {0, 1}
    """
    A_ag = anp.array(A)
    b_ag = anp.array(b)
    N = A_ag.shape[0]

    # Convert labels to {-1, 1}
    y = 2.0 * b_ag - 1.0

    def objective(x: anp.ndarray) -> anp.float64:
        # t_i = a_i^T x
        t = A_ag @ x  # shape (N,)
        yz = y * t    # shape (N,)

        # logistic loss l_i = log(1 + exp(-yz_i))
        # logaddexp is numerically stable:
        # log(1 + exp(-yz)) = logaddexp(0, -yz)
        loss_vec = anp.logaddexp(0.0, -yz)
        ce = anp.sum(loss_vec)

        # L2 regularization
        reg = 0.5 / N * anp.dot(x, x)

        return ce + reg

    return objective

def estimate_L(A: np.ndarray, N: int) -> float:
    """
    Estimate a Lipschitz constant of the gradient of the objective.

    For the (un-averaged) logistic loss:

        J(x) = sum_i log(1 + exp(-y_i * a_i^T x))

    the Hessian satisfies:

        ∇^2 J(x) <= (1/4) * sum_i a_i a_i^T   (in PSD sense)

    so a valid Lipschitz constant for ∇J is:

        L_J <= (1/4) * ||A||_2^2

    The regularization term (1 / (2N)) * ||x||^2 adds (1 / N) * I to
    the Hessian, so overall:

        L <= (1/4) * ||A||_2^2 + 1 / N

    We also apply a small safety factor to be conservative.
    """
    A_norm = np.linalg.norm(A, 2)  # spectral norm
    L_est = 0.25 * (A_norm ** 2) + 1.0 / N
    # Safety factor to avoid underestimation
    return 2.0 * L_est


def accuracy(A: np.ndarray, b: np.ndarray, x: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute classification accuracy of a logistic regression model.

    We use probability p_i = sigma(a_i^T x) and threshold at 'threshold'.
    """
    logits = A @ x
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= threshold).astype(int)
    return float(np.mean(y_pred == b))


# ----- 2. Run GDA, GD, Nesterov and collect objective values -----

def _load_dataset(which: str = "mushrooms", seed: int = 0, test_ratio: float = 0.2):
    which = which.lower().strip()
    if which == "mushrooms":
        A, b = load_mushrooms()
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.int64)

        # Optional: split train/test để báo accuracy rõ ràng
        rng = np.random.default_rng(seed)
        perm = rng.permutation(A.shape[0])
        n_test = int(round(test_ratio * A.shape[0]))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        A_train, b_train = A[train_idx], b[train_idx]
        A_test, b_test = A[test_idx], b[test_idx]
        name = f"Mushrooms (split {1.0 - test_ratio:.0%}/{test_ratio:.0%})"
        return name, A_train, b_train, A_test, b_test

    if which == "w8a":
        A_train, b_train = load_w8a(train=True)
        A_test, b_test = load_w8a(train=False)

        A_train = np.asarray(A_train, dtype=np.float64)
        b_train = np.asarray(b_train, dtype=np.int64)
        A_test = np.asarray(A_test, dtype=np.float64)
        b_test = np.asarray(b_test, dtype=np.int64)

        name = "W8a (train/test)"
        return name, A_train, b_train, A_test, b_test

    raise ValueError("which must be 'mushrooms' or 'w8a'")


def run_experiment(which: str = "mushrooms"):
    # --- Load ONE dataset consistently ---
    name, A_train, b_train, A_test, b_test = _load_dataset(which=which, seed=0, test_ratio=0.2)
    N, d = A_train.shape
    print(f"{name}: train {N} samples, {d} features; test {A_test.shape[0]} samples")

    # --- Build objective on TRAIN only ---
    f = make_regularized_logistic_loss(A_train, b_train)
    L = estimate_L(A_train, N)
    print(f"Using L = {L:.6f} for GD and Nesterov")

    x0 = np.zeros(d, dtype=np.float64)
    max_iter = 200

    # --- GDA ---
    gda = GDA(function=f, bounds=None, constraints=[])
    res_gda = gda.solve(
        x0=x0,
        lambda_0=1.0,
        sigma=0.1,
        kappa=0.5,
        max_iter=max_iter,
        stop_if_stationary=False,  # plot fair: run full max_iter
    )
    vals_gda = [float(f(x)) for x in res_gda.history]

    # --- GD ---
    gd = GD(function=f, bounds=None, constraints=[])
    res_gd = gd.solve(
        x0=x0,
        L=L,
        max_iter=max_iter,
        stop_if_stationary=False,
    )
    vals_gd = [float(f(x)) for x in res_gd.history]

    # --- Nesterov ---
    nest = Nesterov(function=f, bounds=None, constraints=[])
    res_nest = nest.solve(
        x0=x0,
        L=L,
        max_iter=max_iter,
        stop_if_stationary=False,
    )
    vals_nest = [float(f(x)) for x in res_nest.history]

    # --- Report (train + test accuracy, consistent) ---
    print("GDA       final objective:", vals_gda[-1])
    print("GD        final objective:", vals_gd[-1])
    print("Nesterov  final objective:", vals_nest[-1])

    print("GDA       train acc:", accuracy(A_train, b_train, res_gda.x_opt))
    print("GD        train acc:", accuracy(A_train, b_train, res_gd.x_opt))
    print("Nesterov  train acc:", accuracy(A_train, b_train, res_nest.x_opt))

    print("GDA       test  acc:", accuracy(A_test, b_test, res_gda.x_opt))
    print("GD        test  acc:", accuracy(A_test, b_test, res_gd.x_opt))
    print("Nesterov  test  acc:", accuracy(A_test, b_test, res_nest.x_opt))

    # --- Plot ---
    plt.figure()
    plt.plot(range(len(vals_gda)), vals_gda, label="GDA")
    plt.plot(range(len(vals_gd)), vals_gd, label="GD (1/L)")
    plt.plot(range(len(vals_nest)), vals_nest, label="Nesterov (1/L)")
    plt.xlabel("Iteration k")
    plt.ylabel("Objective value J_bar(x_k) [train]")
    plt.title(f"Logistic regression: {name} | GDA vs GD vs Nesterov")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment("mushrooms")
    run_experiment("w8a")
