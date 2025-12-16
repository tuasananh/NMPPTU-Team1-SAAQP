import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

import autograd.numpy as anp
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint

from algorithms.gd import GD
from algorithms.utils import Projector


# ==================================================
# Objective function (Example 2)
# ==================================================
def f(x):
    x1, x2, x3, x4 = x
    return (anp.exp(anp.abs(x2 - 3.0)) - 30.0) / (
        x1**2 + x3**2 + 2.0*x4**2 + 4.0
    )


# ==================================================
# Constraints
# ==================================================
def g1_fun(x):
    x1, _, x3, x4 = x
    return (x1 + x3)**3 + 2.0*x4**2

def g2_fun(x):
    return (x[1] - 1.0)**2

A = np.array([[2.0, 4.0, 1.0, 0.0]])

constraints = [
    NonlinearConstraint(g1_fun, lb=-np.inf, ub=10.0),
    NonlinearConstraint(g2_fun, lb=0.0, ub=1.0),
    LinearConstraint(A, lb=-1.0, ub=-1.0),
]

bounds = Bounds([-10]*4, [10]*4)

projector = Projector(bounds=bounds, constraints=constraints)


# ==================================================
# Run GD
# ==================================================
if __name__ == "__main__":

    x0 = np.array([-1.0, 0.4, -0.5, 0.001])

    solver = GD(
        function=f,
        projector=projector
    )

    result = solver.solve(
        x0=x0,
        step_size=0.01,
        max_iter=50,
        tol=1e-8
    )

    print("==== Example 2 with GD ====")
    print("x =", result.x_opt)
    print("f(x) =", result.fun_opt)
    print("Iterations =", len(result.history))
