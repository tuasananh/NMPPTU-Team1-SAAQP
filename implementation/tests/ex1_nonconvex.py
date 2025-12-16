import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, SRC_DIR)
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint

from algorithms import GDA
from algorithms.utils import Projector


# ==================================================
# Objective function (Example 1)
# ==================================================
def f(x: np.ndarray) -> np.float64:
    x1, x2 = x
    return (x1**2 + x2**2 + 3.0) / (1.0 + 2.0*x1 + 8.0*x2)


# ==================================================
# Constraint: x1^2 + 2 x1 x2 >= 4
# ==================================================
def nonlinear_constraint_fun(x):
    x1, x2 = x
    return x1**2 + 2.0*x1*x2


# ==================================================
# Build projector
# ==================================================
bounds = Bounds([0.0, 0.0], [np.inf, np.inf])

nonlinear_constraint = NonlinearConstraint(
    nonlinear_constraint_fun,
    lb=4.0,
    ub=np.inf
)

projector = Projector(
    bounds=bounds,
    constraints=[nonlinear_constraint]
)


# ==================================================
# Run GDA
# ==================================================
if __name__ == "__main__":
    x0 = np.array([1.0, 2.0])

    solver = GDA(
        function=f,
        projector=projector
    )

    result = solver.solve(
        x0=x0,
        lambda_0=1.0,
        sigma=0.1,
        kappa=0.5,
        max_iter=1000,
        tol=1e-8
    )

    print("==== Example 1: Fractional Pseudoconvex ====")
    print("x* =", result.x_opt)
    print("f(x*) =", result.fun_opt)
    print("Converged =", result.success)
    print("Iterations =", len(result.history))
