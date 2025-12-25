from typing import List, Optional
import numpy as np

from autograd import grad

from .utils import (
    ScalarFunction,
    VectorFunction,
    OptimizationResult,
)

from tqdm import tqdm


class Nesterov:
    """
    Projected Nesterov's accelerated gradient method for minimizing
    a differentiable function over a convex set.

    The method solves:
        minimize   f(x)
        subject to x in C

    where C is a convex set described by bounds and constraints.

    This implementation uses the classical Nesterov acceleration scheme
    with a constant step size 1 / L and projection onto C:

        x_{k+1} = P_C( y_k - (1/L) * grad f(y_k) )
        t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
        y_{k+1} = x_{k+1} + (t_k - 1) / t_{k+1} * (x_{k+1} - x_k)
    """

    def __init__(self, function: ScalarFunction, projector: VectorFunction):
        """
        Initialize the Nesterov optimizer.

        Args:
            function: Objective function to minimize.
            bounds: Bounds describing the feasible set (can be None for R^n).
            constraints: List of constraints describing the feasible set.
            tol: Tolerance used in the stopping criterion.
            projector_max_iter: Maximum number of iterations for the projection subproblem.
        """
        self.function = function
        self.projector = projector
        self.gradient: VectorFunction = grad(function)

    def solve(
        self,
        x0: np.ndarray,
        step_size: float,
        max_iter: int = 1000,
        stop_if_stationary: bool = True,
        tol: Optional[float] = 1e-8,
        with_x_history: bool = True,
        with_f_history: bool = False,
        with_lr_history: bool = False,
        with_tqdm: bool = False,
    ) -> OptimizationResult:
        """
        Solve the optimization problem using projected Nesterov's accelerated gradient.

        Args:
            x0: Initial point.
            L: Lipschitz constant of the gradient ∇f (L > 0). The step
               size used is 1 / L.
            max_iter: Maximum number of iterations.
            stop_if_stationary: If True, stop when successive iterates
                become stationary (x_{k+1} ≈ x_k).
            tol: Optional override of the default tolerance.
            projector_max_iter: Optional override for the projector's
                maximum number of iterations.

        Returns:
            OptimizationResult: The result of the optimization, containing:
                - x_opt: Final iterate.
                - fun_opt: Objective value at x_opt.
                - success: True if converged before reaching max_iter.
                - message: Text description of termination.
                - history: List of iterates x_k.
        """

        projector = self.projector
        
        # Initialize
        x_k = projector(x0)
        y_k = x_k.copy()
        t_k = 1.0
        
        xs: List[np.ndarray] = []
        fs = []
        lrs = []
        success = False

        iter_range = tqdm(range(max_iter), leave=False, desc="Nesterov") if with_tqdm else range(max_iter)
        for _ in iter_range:
            if with_x_history:
                xs.append(x_k.copy()) # Copy to ensure history doesn't point to updated array
            if with_f_history:
                fs.append(self.function(x_k))
            if with_lr_history:
                lrs.append(step_size)

            # 1. Gradient step (using lookahead y_k)
            grad = self.gradient(y_k)
            x_k1 = projector(y_k - step_size * grad)

            # 2. Check stopping criterion (only needs x)
            if stop_if_stationary and np.allclose(x_k, x_k1, atol=tol, rtol=tol):
                x_k = x_k1 # Ensure we return the latest
                success = True
                break

            # 3. Nesterov momentum update
            # We can overwrite y_k immediately because old y_k is done
            t_k1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            beta = (t_k - 1.0) / t_k1
            
            y_k = x_k1 + beta * (x_k1 - x_k)
            
            # Update pointers
            x_k = x_k1
            t_k = t_k1

        # Calculate function value ONLY ONCE at the end
        f_opt = self.function(x_k)

        return OptimizationResult(
            x_opt=x_k,
            f_opt=f_opt,
            success=success,
            x_history=xs,
            f_history=fs,
            lr_history=lrs,
        )
