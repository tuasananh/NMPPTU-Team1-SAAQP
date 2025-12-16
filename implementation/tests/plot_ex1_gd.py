import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, NonlinearConstraint
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

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

# ==================================================
# Initial points (same as GDA)
# ==================================================
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

MAX_PLOT_ITER = 700

results = []

print("===== GD results for Example 1 =====")

# ==================================================
# Run GD
# ==================================================
for i, x0 in enumerate(x0_list):
    solver = GD(
        function=f,
        projector=projector
    )

    result = solver.solve(
        x0=x0,
        step_size=0.05,   
        max_iter=1000,
        tol=1e-8
    )

    results.append(result)

    print(f"\n--- Run {i+1} ---")
    print("Initial point:", x0)
    print("x =", result.x_opt)
    print("f(x) =", result.fun_opt)
    print("Iterations =", len(result.history))


# ==================================================
# Plot (FORMAT GIỐNG HỆT GDA)
# ==================================================
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

plt.xlabel("Iteration $t$")
plt.ylabel(r"$x(t)$")
plt.title("Computational results for Example 1 (GD)")
plt.legend(fontsize=9)
plt.grid(True)
plt.tight_layout()

plt.savefig("fig1_gd_example1.pdf")
plt.show()
plt.close()
