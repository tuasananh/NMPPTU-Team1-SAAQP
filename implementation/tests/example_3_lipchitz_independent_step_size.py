import time
import numpy as np
import autograd.numpy as anp
from autograd import grad
from scipy.optimize import brentq, NonlinearConstraint, Bounds

from algorithms.gd import GD
from algorithms.gda import GDA
from algorithms.utils import Projector

def true_euclidean_projector(x: np.ndarray) -> np.ndarray:
    """
    Computes the true Euclidean projection of vector x onto the set C:
    C = {y in R^n_{++} | product(y_i) >= 1}
    
    This method uses the KKT conditions and root-finding (Brent's method) 
    and is robust even when x contains negative or zero components.

    Args:
        x: A 1D numpy array representing the vector to project.

    Returns:
        A 1D numpy array representing the projected vector P_C(x).
    """
    epsilon = 1e-12 # Use a slightly smaller epsilon for robust log/sqrt
    
    # --- Step 1: Simple Bounds Check and Early Exit ---
    # First, project onto the R^n_{++} orthant.
    y_bounds = np.maximum(x, epsilon)
    
    # Check if this bounded vector already satisfies the product constraint (is in C).
    # log(product(y)) >= 0  <=>  sum(log(y)) >= 0
    if np.sum(np.log(y_bounds)) >= 0:
        return y_bounds
        
    # --- Step 2: Iterative Solution for Active Constraint ---
    # The constraint is active: P_C(x) lies on the boundary product(y_i) = 1.
    
    def get_y(lam):
        """Calculates y_i(lambda) from the KKT stationarity condition."""
        # y_i = (x_i + sqrt(x_i^2 + 4*lambda)) / 2
        # This solution automatically ensures y_i > 0 for lambda > 0.
        val = (x + np.sqrt(x**2 + 4 * lam)) / 2
        # Use epsilon for the final projection to ensure strict positivity R^n_{++}
        return np.maximum(val, epsilon) 
        
    def objective(lam):
        """The dual objective function to find lambda: sum(log(y_i)) - log(1) = 0."""
        y = get_y(lam)
        # Note: We seek the root where sum(log(y)) = 0
        return np.sum(np.log(y))
        
    # --- Step 3: Find Bracket for brentq ---
    lam_min = 0.0 # The lower bound of lambda (when constraint is non-active)
    lam_max = 1.0
    
    # Find a lam_max such that objective(lam_max) > 0 (i.e., product(y) > 1)
    while objective(lam_max) < 0:
        lam_max *= 2
        # Prevent indefinite loop for extremely difficult cases
        if lam_max > 1e12: 
            break
            
    # --- Step 4: Find the Optimal Lambda ---
    try:
        # brentq is highly efficient for finding the root of a single-variable function
        # The objective function is guaranteed to be monotonic, which is ideal for brentq.
        lam_opt = brentq(objective, lam_min, lam_max)
        return get_y(lam_opt)
    except ValueError as e:
        # This usually means the bracket failed. Return the simple bounds projection as a safe fallback.
        print(f"Brent's method failed: {e}. Falling back to bounds projection.")
        return y_bounds


def run_experiment():

    print(f"{'-'*110}")
    print(f"{'n':<5} | {'Algorithm GDA (proposed)':<45} | {'Algorithm GD':<45}")
    print(f"{'':<5} | {'f(x*)':<12} {'#Iter':<10} {'Time':<15} | {'f(x*)':<12} {'#Iter':<10} {'Time':<15}")
    print(f"{'-'*110}")
    # n_values = [10, 20, 50, 100, 200, 500]
    np.random.seed(42)
    n_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    for n in n_values:

        e = np.arange(1, n + 1, dtype=np.float64)
        a = np.random.uniform(1e-12, 1.0, size=n)    
        beta = 0.741271
        alpha = 3 * (beta**1.5) * np.sqrt(n + 1)
        # L = 4 * (beta**1.5) * np.sqrt(n) + 3 * alpha
        # L = 10
        
        def objective(x):
            xt_x = anp.dot(x, x)
            at_x = anp.dot(a, x)
            et_x = anp.dot(e, x)
            term3 = (beta / anp.sqrt(1 + beta * xt_x)) * et_x
            return at_x + alpha * xt_x + term3

        bounds = Bounds(lb=1e-6, ub=np.inf)
        def constraint_func(x):
            x_safe = anp.maximum(x, 1e-8)
            return anp.sum(anp.log(x_safe))

        constraints = [NonlinearConstraint(fun=constraint_func, lb=0.0, ub=np.inf, jac=grad(constraint_func))]
        x0 = np.random.rand(n)
        
        projector = Projector(bounds=bounds, constraints=constraints)
        
        gda_solver = GDA(objective, projector=true_euclidean_projector)
        start_time = time.time()
        res_gda = gda_solver.solve(
            x0=x0,
            lambda_0=0.5,
            sigma=0.1,
            kappa=0.5,
            max_iter=1000,
            stop_if_stationary=True,
            tol=1e-9,
        )

        time_gda = time.time() - start_time
        gd_solver = GD(objective, projector=true_euclidean_projector)
        start_time = time.time()
        res_gd = gd_solver.solve(
            x0=x0,
            step_size=0.1,
            max_iter=1000,
            stop_if_stationary=True,
            tol=1e-9,
        )
        time_gd = time.time() - start_time
        print(f"{n:<5} | "
              f"{res_gda.fun_opt:<12.4f} {len(res_gda.history):<10} {time_gda:<15.4f} | "
              f"{res_gd.fun_opt:<12.4f} {len(res_gd.history):<10} {time_gd:<15.4f}")
    print(f"{'-'*110}")

if __name__ == "__main__":
    run_experiment()