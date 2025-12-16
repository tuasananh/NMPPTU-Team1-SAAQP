import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint

from algorithms.gd import GD
from algorithms.utils import Projector


# ==================================================
# Objective function (Example 1)
# ==================================================
def f(x):
    x1, x2 = x
    return (x1**2 + x2**2 + 3.0) / (1.0 + 2.0*x1 + 8.0*x2)


# ==================================================
# Constraint
# ==================================================
def g_fun(x):
    x1, x2 = x
    return x1**2 + 2.0*x1*x2


bounds = Bounds([0.0, 0.0], [np.inf, np.inf])

constraint = NonlinearConstraint(
    g_fun,
    lb=4.0,
    ub=np.inf
)

projector = Projector(bounds=bounds, constraints=[constraint])


# ==================================================
# Run GD
# ==================================================
if __name__ == "__main__":

    x0 = np.array([1.0, 2.0])

    solver = GD(
        function=f,
        projector=projector
    )

    result = solver.solve(
        x0=x0,
        step_size=0.05,     # 🔥 cố tình chọn nhỏ
        max_iter=100,
        tol=1e-8
    )

    print("==== Example 1 with GD ====")
    print("x =", result.x_opt)
    print("f(x) =", result.fun_opt)
    print("Iterations =", len(result.history))
