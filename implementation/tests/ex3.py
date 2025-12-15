import sys
import os
import time
import numpy as np
import autograd.numpy as anp

from algorithms.gd import GD
from algorithms.gda import GDA
from algorithms.utils import Bounds, Constraint, Constraints, ConstraintType

def run_experiment():

    print(f"{'-'*110}")
    print(f"{'n':<5} | {'Algorithm GDA (proposed)':<45} | {'Algorithm GD':<45}")
    print(f"{'':<5} | {'f(x*)':<12} {'#Iter':<10} {'Time':<15} | {'f(x*)':<12} {'#Iter':<10} {'Time':<15}")
    print(f"{'-'*110}")
    n_values = [10, 20, 50, 100, 200, 500]
    np.random.seed(42)

    for n in n_values:

        e = np.arange(1, n + 1, dtype=np.float64)
        a = np.random.uniform(0.1, 1.0, size=n)
        beta = 0.741271
        alpha = 3 * (beta**1.5) * np.sqrt(n) + 1
        L = 4 * (beta**1.5) * np.sqrt(n) + 3 * alpha

        def objective(x):
            xt_x = anp.dot(x, x)
            at_x = anp.dot(a, x)
            et_x = anp.dot(e, x)
            term3 = (beta / anp.sqrt(1 + beta * xt_x)) * et_x
            return at_x + alpha * xt_x + term3

        bounds = Bounds(lb=1e-6, ub=np.inf)
        def constraint_func(x):
            return anp.sum(anp.log(x))

        constraints = [Constraint(lhs=constraint_func, type=ConstraintType.NON_NEGATIVE)]
        x0 = np.ones(n) * 2.0
        gda_solver = GDA(objective, bounds, constraints)
        start_time = time.time()
        res_gda = gda_solver.solve(
            x0=x0,
            lambda_0=5.0/L,
            sigma=0.1,
            kappa=0.5,
            max_iter=1000,
            stop_if_stationary=True,
            tol=1e-6
        )

        time_gda = time.time() - start_time
        gd_solver = GD(objective, bounds, constraints)
        start_time = time.time()
        res_gd = gd_solver.solve(
            x0=x0,
            step_size=1.0/L,
            max_iter=1000,
            stop_if_stationary=True,
            tol=1e-6

        )
        time_gd = time.time() - start_time
        print(f"{n:<5} | "
              f"{res_gda.fun_opt:<12.4f} {len(res_gda.history):<10} {time_gda:<15.4f} | "
              f"{res_gd.fun_opt:<12.4f} {len(res_gd.history):<10} {time_gd:<15.4f}")
    print(f"{'-'*110}")

if __name__ == "__main__":
    run_experiment()