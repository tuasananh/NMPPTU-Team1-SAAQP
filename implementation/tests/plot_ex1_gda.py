import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, NonlinearConstraint

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from algorithms import GDA
from algorithms.utils import Projector

def f(x: np.ndarray) -> np.float64:
    x1, x2 = x
    return (x1**2 + x2**2 + 3.0) / (1.0 + 2.0*x1 + 8.0*x2)


def constraint_fun(x):
    x1, x2 = x
    return x1**2 + 2.0*x1*x2


bounds = Bounds([0.0, 0.0], [np.inf, np.inf])

nonlinear_constraint = NonlinearConstraint(
    constraint_fun,
    lb=4.0,
    ub=np.inf
)

projector = Projector(
    bounds=bounds,
    constraints=[nonlinear_constraint]
)



x0_list = [
    np.array([1.0, 2.0]),
    np.array([2.0, 0.8]),
    np.array([1.5, 1.8]),
]

labels = [
    r"$x^0 = (1.0,\,2.0)$",
    r"$x^0 = (2.0,\,0.8)$",
    r"$x^0 = (1.5,\,1.8)$",
]

MAX_PLOT_ITER = 100


results = []

print("===== GDA results for Example 1 =====")

for i, x0 in enumerate(x0_list):
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

    results.append(result)

    print(f"\n--- Run {i+1} ---")
    print("Initial point:", x0)
    print("x* =", result.x_opt)
    print("f(x*) =", result.fun_opt)
    print("Total iterations =", len(result.history))



plt.figure(figsize=(7, 5))

for result, label in zip(results, labels):
    xs = np.array(result.history)
    xs_plot = xs[:MAX_PLOT_ITER]
    t = np.arange(len(xs_plot))

    plt.plot(
        t,
        xs_plot[:, 0],
        linewidth=2,
        label=label + r", $x_1(t)$"
    )
    plt.plot(
        t,
        xs_plot[:, 1],
        linestyle="--",
        linewidth=2,
        label=label + r", $x_2(t)$"
    )

plt.xlabel("Iteration t")
plt.ylabel(r"$x(t)$")
plt.title("Computational results for Example 1")
plt.legend(fontsize=9)
plt.grid(True)
plt.tight_layout()

plt.savefig("fig1_computational_results_example1.pdf")
plt.show()
plt.close()