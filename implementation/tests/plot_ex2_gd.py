import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)
import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint

from algorithms.gd import GD
from algorithms.utils import Projector

def f(x):
    x1, x2, x3, x4 = x
    numerator = anp.exp(anp.abs(x2 - 3.0)) - 30.0
    denominator = x1**2 + x3**2 + 2.0 * x4**2 + 4.0
    return numerator / denominator
def g1_fun(x):
    x1, _, x3, x4 = x
    return (x1 + x3)**3 + 2.0 * x4**2

def g2_fun(x):
    return (x[1] - 1.0)**2

A = np.array([[2.0, 4.0, 1.0, 0.0]])

constraints = [
    NonlinearConstraint(g1_fun, lb=-np.inf, ub=10.0),
    NonlinearConstraint(g2_fun, lb=0.0, ub=1.0),
    LinearConstraint(A, lb=-1.0, ub=-1.0),
]

bounds = Bounds(
    [-10.0, -10.0, -10.0, -10.0],
    [ 10.0,  10.0,  10.0,  10.0],
)

projector = Projector(bounds=bounds, constraints=constraints)
initial_points = [
    np.array([ 0.5,  1.8, -2.7,  0.5]),
    np.array([-2.5,  0.2,  1.7,  0.3]),
    np.array([ 1.0,  1.2, -5.8,  0.9]),
]
print("===== Computational results for Example 2 (GD) =====")

results = []

for i, x0 in enumerate(initial_points):
    print(f"\n--- Run {i+1}, x0 = {x0} ---")

    solver = GD(function=f, projector=projector)

    result = solver.solve(
        x0=x0,
        step_size=0.02,     # GD cần bước nhỏ để ổn định
        max_iter=3000,
        tol=1e-8,
    )

    results.append(result)

    print("x* =", result.x_opt)
    print("f(x*) =", result.fun_opt)
    print("Iterations =", len(result.history))

plt.figure(figsize=(9, 6))

colors = ["tab:blue", "tab:orange", "tab:green"]
linestyles = ["-", "--", "-.", ":"]   # x1, x2, x3, x4

MAX_PLOT_ITER = 1600

for idx, (result, color) in enumerate(zip(results, colors)):
    history = np.array(result.history[:MAX_PLOT_ITER])
    t = np.arange(len(history))

    for j in range(4):
        plt.plot(
            t,
            history[:, j],
            color=color,
            linestyle=linestyles[j],
            linewidth=2,
            label=rf"$x_{j+1}(t)$, start {idx+1}",
        )

plt.xlabel("Iteration $t$")
plt.ylabel(r"$x(t)$")
plt.title("Computational results for Example 2 (GD)")
plt.grid(True)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

plt.savefig("fig2_computational_results_example2_gd.pdf")
plt.show()
plt.close()
