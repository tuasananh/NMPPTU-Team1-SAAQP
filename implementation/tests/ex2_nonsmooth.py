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
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint

from algorithms import GDA
from algorithms.utils import Projector

# ==================================================
# Objective function (Example 2 – paper)
# f(x) = (exp(|x2 - 3|) - 30) / (x1^2 + x3^2 + 2x4^2 + 4)
# ==================================================
def f(x):
    x1, x2, x3, x4 = x
    numerator = anp.exp(anp.abs(x2 - 3.0)) - 30.0
    denominator = x1**2 + x3**2 + 2.0*x4**2 + 4.0
    return numerator / denominator

# ==================================================
# Constraints
# ==================================================

# g1(x) = (x1 + x3)^3 + 2x4^2 ≤ 10
def g1_fun(x):
    x1, _, x3, x4 = x
    return (x1 + x3)**3 + 2.0*x4**2

# g2(x) = (x2 - 1)^2 ≤ 1
def g2_fun(x):
    return (x[1] - 1.0)**2

# Linear: 2x1 + 4x2 + x3 = -1
A = np.array([[2.0, 4.0, 1.0, 0.0]])
linear_constraint = LinearConstraint(A, lb=-1.0, ub=-1.0)

constraints = [
    NonlinearConstraint(g1_fun, lb=-np.inf, ub=10.0),
    NonlinearConstraint(g2_fun, lb=0.0, ub=1.0),
    linear_constraint
]

# ==================================================
# Bounds (loose, đúng tinh thần bài báo)
# ==================================================
bounds = Bounds(
    [-10.0, -10.0, -10.0, -10.0],
    [ 10.0,  10.0,  10.0,  10.0]
)

projector = Projector(bounds=bounds, constraints=constraints)

# ==================================================
# MULTI–START INITIAL POINTS
# ==================================================
initial_points = [
    np.array([-0.5, 1.0, -0.5, 0.1]),
    np.array([-1.0, 0.5, -0.5, 0.0]),
    np.array([-1.2, 0.4, -0.6, 0.0]),   # gần nghiệm bài báo
    np.array([-2.0, 1.2, -1.0, 0.1])
]

# ==================================================
# Run GDA
# ==================================================
if __name__ == "__main__":

    best_result = None

    for i, x0 in enumerate(initial_points):
        print(f"\n--- Run {i+1}, x0 = {x0} ---")

        solver = GDA(
            function=f,
            projector=projector
        )

        result = solver.solve(
            x0=x0,
            lambda_0=0.1,     # 🔧 nhỏ hơn → ổn định hơn
            sigma=0.05,
            kappa=0.7,
            max_iter=3000,
            tol=1e-8
        )

        print("x* =", result.x_opt)
        print("f(x*) =", result.fun_opt)
        print("Iterations =", len(result.history))

        if best_result is None or result.fun_opt < best_result.fun_opt:
            best_result = result

    print("\n================ BEST RESULT ================")
    print("x* =", best_result.x_opt)
    print("f(x*) =", best_result.fun_opt)
    print("Iterations =", len(best_result.history))
