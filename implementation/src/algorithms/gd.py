from typing import List, Optional
import numpy as np

from autograd import grad

from .utils import ScalarFunction, VectorFunction, Bounds, Constraints, Projector
from .gda import OptimizationResult


class GD:
    """
    Projected gradient descent with constant step size 1 / L.

    It solves:
        minimize   f(x)
        subject to x in C

    with updates:
        x_{k+1} = P_C( x_k - (1/L) * grad f(x_k) ).
    """

    def __init__(
        self,
        function: ScalarFunction,
        bounds: Bounds,
        constraints: Constraints,
        tol: float = 1e-9,
        projector_max_iter: int = 1000,
    ):
        """
        Initialize the GD optimizer.

        Args:
            function: Objective function to minimize.
            bounds: Bounds describing the feasible set (can be None for R^n).
            constraints: List of constraints describing the feasible set.
            tol: Tolerance used in the stopping criterion.
            projector_max_iter: Maximum number of iterations for the projection subproblem.
        """
        self.function = function
        self.bounds = bounds
        self.constraints = constraints
        self.gradient: VectorFunction = grad(function)
        self.tol = tol
        self.projector_max_iter = projector_max_iter

    def solve(
        self,
        x0: np.ndarray,
        L: float,
        max_iter: int = 1000,
        stop_if_stationary: bool = True,
        tol: Optional[float] = None,
        projector_max_iter: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run projected gradient descent with step size 1 / L.

        Args:
            x0: Initial point.
            L: Lipschitz constant of the gradient ∇f (L > 0).
            max_iter: Maximum number of iterations.
            stop_if_stationary: If True, stop when x_{k+1} is close to x_k.
            tol: Optional override of the default tolerance.
            projector_max_iter: Optional override for the projector's
                maximum number of iterations.

        Returns:
            OptimizationResult with the final iterate, objective value
            and the history of iterates.
        """
        if L <= 0:
            raise ValueError("L (Lipschitz constant) must be positive.")

        tol = tol if tol is not None else self.tol
        projector_max_iter = (
            projector_max_iter
            if projector_max_iter is not None
            else self.projector_max_iter
        )

        projector = Projector(
            bounds=self.bounds,
            constraints=self.constraints,
            tol=tol,
            max_iter=projector_max_iter,
        )

        # Project the initial point onto the feasible set
        x_k = projector(x0)
        xs: List[np.ndarray] = []
        f_x_k = self.function(x_k)

        converged = False
        for _ in range(max_iter):
            xs.append(x_k.copy())
            grad_f_x_k = self.gradient(x_k)
            x_k1 = projector(x_k - (1.0 / L) * grad_f_x_k)
            f_x_k1 = self.function(x_k1)

            if stop_if_stationary and np.allclose(x_k, x_k1, atol=tol, rtol=tol):
                x_k = x_k1
                f_x_k = f_x_k1
                converged = True
                break

            x_k = x_k1
            f_x_k = f_x_k1

        xs.append(x_k.copy())

        return OptimizationResult(
            x_opt=x_k,
            fun_opt=f_x_k,
            success=converged,
            history=xs,
        )
