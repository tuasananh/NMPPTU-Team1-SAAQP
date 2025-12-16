import sys
import os

# ==================================================
# Path setup
# ==================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

# ==================================================
# Imports
# ==================================================
import autograd.numpy as anp   # ⚠️ BẮT BUỘC
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint

from algorithms.gd import GD
from algorithms.utils import Projector

# ==================================================
# Objective function (Example 2)
# ==================================================
def f(x):
    x1, x2, x3, x4 = x
    numerator = anp.exp(anp.abs(x2 - 3.0)) - 30.0
    denominator = x1**2 + x3**2 + 2.0 * x4**2 + 4.0
    return numerator / denominator


# ==================================================
# Constraints
# ==================================================
def g1_fun(x):
    x1, _, x3, x4 = x
    return (x1 + x3)**3 + 2.0 * x4**2


def g2_fun(x):
    return (x[1] - 1.0)**2


A = np.array([[2.0, 4.0, 1.0, 0.0]])
linear_constraint = LinearConstraint(A, lb=-1.0, ub=-1.0)

constraints = [
    NonlinearConstraint(g1_fun, lb=-np.inf, ub=10.0),
    NonlinearConstraint(g2_fun, lb=0.0, ub=1.0),
    linear_constraint,
]

bounds = Bounds(
    [-10.0, -10.0, -10.0, -10.0],
    [ 10.0,  10.0,  10.0,  10.0]
)

projector = Projector(bounds=bounds, constraints=constraints)

# ==================================================
# Initial points
# ==================================================
initial_points = [
    np.array([ 0.5,  1.8, -2.7,  0.5]),
    np.array([-2.5,  0.2,  1.7,  0.3]),
    np.array([ 1.0,  1.2, -5.8,  0.9]),
]
labels = [
    r"$x^0_1$",
    r"$x^0_2$",
    r"$x^0_3$",
]

MAX_PLOT_ITER = 1500

# ==================================================
# Run GD and store results
# ==================================================
results = []

print("===== Computational results for Example 2 (GD) =====")

for i, x0 in enumerate(initial_points):
    print(f"\n--- Run {i+1}, x0 = {x0} ---")

    solver = GD(
        function=f,
        projector=projector
    )

    result = solver.solve(
        x0=x0,
        step_size=0.02,      # 🔧 nhỏ để tránh dao động
        max_iter=3000,
        tol=1e-8
    )

    results.append(result)

    print("x* =", result.x_opt)
    print("f(x*) =", result.fun_opt)
    print("Iterations =", len(result.history))


# ==================================================
# Plot
# ==================================================
plt.figure(figsize=(8, 5))

for result, label in zip(results, labels):
    xs = np.array(result.history[:MAX_PLOT_ITER])
    t = np.arange(len(xs))

    for i in range(4):
        plt.plot(
            t,
            xs[:, i],
            linewidth=2,
            label=label + rf", $x_{i+1}(t)$"
        )

plt.xlabel("Iteration t")
plt.ylabel(r"$x(t)$")
plt.title("Computational results for Example 2 (GD)")
plt.grid(True)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

plt.savefig("fig2_computational_results_example2_gd.pdf")
plt.show()
plt.close()
