import time
import numpy as np
import autograd.numpy as anp

from algorithms.gd import GD
from algorithms.gda import GDA
from algorithms.utils import Projector
from algorithms.utils import Bounds, Constraint, ConstraintType


# ---------- Gradient mapping ----------
def gradient_mapping_norm(x, grad_fx, step, projector):
    y = projector(x - step * grad_fx)
    return np.linalg.norm((x - y) / step)


# ---------- Experiment ----------
def run_experiment():
    print("-" * 120)
    print(
        f"{'n':<5} | "
        f"{'GDA f(x*)':<12} {'||G||':<10} {'Iter':<8} {'Time(s)':<10} | "
        f"{'GD f(x*)':<12} {'||G||':<10} {'Iter':<8} {'Time(s)':<10}"
    )
    print("-" * 120)

    n_values = [10, 20, 50, 100, 200, 500]
    np.random.seed(42)

    tol = 1e-6
    max_iter = 5000

    for n in n_values:
        # ----- Problem data -----
        e = anp.arange(1, n + 1, dtype=anp.float64)
        a = anp.random.uniform(0.1, 1.0, size=n)

        beta = 0.741271
        alpha = 3 * (beta ** 1.5) * anp.sqrt(n) + 1
        L = 4 * (beta ** 1.5) * anp.sqrt(n) + 3 * alpha

        def objective(x):
            xtx = anp.dot(x, x)
            return (
                anp.dot(a, x)
                + alpha * xtx
                + beta / anp.sqrt(1 + beta * xtx) * anp.dot(e, x)
            )

        # ----- Constraints: x >= 1 -----
        bounds = Bounds(lb=np.ones(n), ub=np.inf)
        constraints = []

        projector = Projector(bounds=bounds, constraints=constraints, tol=1e-9)

        x0 = np.ones(n) * 2.0

        # ---------- GDA ----------
        gda = GDA(objective, bounds, constraints)
        start = time.time()
        res_gda = gda.solve(
            x0=x0,
            lambda_0=5.0 / L,
            sigma=0.1,
            kappa=0.5,
            max_iter=max_iter,
            stop_if_stationary=False,  # VERY IMPORTANT
            tol=tol,
        )
        time_gda = time.time() - start

        gnorm_gda = gradient_mapping_norm(
            res_gda.x_opt,
            gda.gradient(res_gda.x_opt),
            5.0 / L,
            projector,
        )

        # ---------- GD ----------
        gd = GD(objective, bounds, constraints)
        start = time.time()
        res_gd = gd.solve(
            x0=x0,
            step_size=1.0 / L,
            max_iter=max_iter,
            stop_if_stationary=False,  # VERY IMPORTANT
            tol=tol,
        )
        time_gd = time.time() - start

        gnorm_gd = gradient_mapping_norm(
            res_gd.x_opt,
            gd.gradient(res_gd.x_opt),
            1.0 / L,
            projector,
        )

        print(
            f"{n:<5} | "
            f"{res_gda.fun_opt:<12.4f} {gnorm_gda:<10.2e} {len(res_gda.history):<8} {time_gda:<10.4f} | "
            f"{res_gd.fun_opt:<12.4f} {gnorm_gd:<10.2e} {len(res_gd.history):<8} {time_gd:<10.4f}"
        )

    print("-" * 120)


if __name__ == "__main__":
    run_experiment()
