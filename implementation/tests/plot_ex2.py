import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint

from algorithms import GDA
from algorithms.utils import Projector


def f(x):
    x1, x2, x3, x4 = x
    numerator = anp.exp(anp.abs(x2 - 3.0)) - 30.0
    denominator = x1**2 + x3**2 + 2.0*x4**2 + 4.0
    return numerator / denominator


def g1_fun(x):
    x1, _, x3, x4 = x
    return (x1 + x3)**3 + 2.0*x4**2

def g2_fun(x):
    return (x[1] - 1.0)**2

A = np.array([[2.0, 4.0, 1.0, 0.0]])
constraints = [
    NonlinearConstraint(g1_fun, lb=-np.inf, ub=10.0),
    NonlinearConstraint(g2_fun, lb=0.0, ub=1.0),
    LinearConstraint(A, lb=-1.0, ub=-1.0)
]

bounds = Bounds(
    [-10, -10, -10, -10],
    [ 10,  10,  10,  10]
)

projector = Projector(bounds=bounds, constraints=constraints)

initial_points = [
    np.array([-0.5, 0.4, -0.5, 0.0]),
    np.array([-0.7, 0.0, -0.35, 0.75]),
    np.array([-1.05, 0.3, -0.75, 0.6]),
]

# ==================================================
# Plot
# ==================================================
plt.figure(figsize=(9, 6))

for idx, x0 in enumerate(initial_points):
    solver = GDA(function=f, projector=projector)

    result = solver.solve(
        x0=x0,
        lambda_0=0.1,
        sigma=0.05,
        kappa=0.7,
        max_iter=30,
        tol=1e-8
    )

    history = np.array(result.history)

    plt.plot(history[:, 0], label=f"$x_1(t)$, start {idx+1}")
    plt.plot(history[:, 1], label=f"$x_2(t)$, start {idx+1}", linestyle="--")
    plt.plot(history[:, 2], label=f"$x_3(t)$, start {idx+1}", linestyle="-.")
    plt.plot(history[:, 3], label=f"$x_4(t)$, start {idx+1}", linestyle=":")


plt.xlabel("Iteration")
plt.ylabel("$x(t)$")
plt.title("Computational results for Example 2")
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
