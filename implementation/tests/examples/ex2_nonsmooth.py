import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint

from algorithms import GDA
from algorithms.utils import Projector


# ==================================================
# Objective function (Example 2 - paper)
# ==================================================
def f(x: np.ndarray) -> np.float64:
    x1, x2, x3, x4 = x
    numerator = np.exp(np.abs(x2 - 3.0)) - 30.0
    denominator = x1**2 + x3**2 + 2.0*x4**2 + 4.0
    return numerator / denominator


# ==================================================
# Constraints
# ==================================================

# g1(x) = (x1 + x3)^3 + 2 x4^2 <= 10
def g1_fun(x):
    x1, _, x3, x4 = x
    return (x1 + x3)**3 + 2.0*x4**2


# g2(x) = (x2 - 1)^2 <= 1
def g2_fun(x):
    return (x[1] - 1.0)**2


# Linear constraint: 2x1 + 4x2 + x3 = -1
A = np.array([[2.0, 4.0, 1.0, 0.0]])
linear_constraint = LinearConstraint(A, lb=-1.0, ub=-1.0)

nonlinear_constraint_1 = NonlinearConstraint(g1_fun, lb=-np.inf, ub=10.0)
nonlinear_constraint_2 = NonlinearConstraint(g2_fun, lb=0.0, ub=1.0)

constraints = [
    nonlinear_constraint_1,
    nonlinear_constraint_2,
    linear_constraint
]


# ==================================================
# Bounds (no explicit bounds in paper → loose bounds)
# ==================================================
bounds = Bounds(
    [-10.0, -10.0, -10.0, -10.0],
    [10.0, 10.0, 10.0, 10.0]
)


projector = Projector(
    bounds=bounds,
    constraints=constraints
)


# ==================================================
# Run GDA
# ==================================================
if __name__ == "__main__":

    # Initial point (must satisfy constraints approximately)
    x0 = np.array([-0.5, 1.0, -0.5, 0.1])

    solver = GDA(
        function=f,
        projector=projector
    )

    result = solver.solve(
        x0=x0,
        lambda_0=1.0,
        sigma=0.1,
        kappa=0.5,
        max_iter=2000,
        tol=1e-8
    )

    print("==== Example 2 (Paper) ====")
    print("x* =", result.x_opt)
    print("f(x*) =", result.fun_opt)
    print("Converged =", result.success)
    print("Iterations =", len(result.history))
