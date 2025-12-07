from typing import Any, Callable, List, Optional
import numpy as np

from autograd import grad

from .gda import OptimizationResult
from .utils import Bounds, Constraints, Projector

StochasticScalarFunction = Callable[[np.ndarray, Any], np.float64]
Sampler = Callable[[], Any]


class SGDA:
    """
    Stochastic projected gradient descent with adaptive step size
    (corresponding to Algorithm 3 in the paper).

    Problem form:
        minimize   E_ξ[ f_ξ(x) ]
        subject to x in C

    where C is a convex set represented by bounds and constraints.
    """

    def __init__(
        self,
        stochastic_function: StochasticScalarFunction,
        bounds: Bounds,
        constraints: Constraints,
        sampler: Sampler,
        tol: float = 1e-9,
        projector_max_iter: int = 1000,
    ):
        """
        Initialize the SGDA optimizer.

        Args:
            stochastic_function: Function f_ξ(x). It receives (x, xi) and
                returns a scalar value f_ξ(x).
            bounds: Variable bounds (as scipy.optimize.Bounds).
            constraints: List of constraints defining the feasible set.
            sampler: Function without arguments that returns a random ξ
                each time it is called.
            tol: Tolerance used in the stopping criterion.
            projector_max_iter: Maximum iterations for the projection subproblem.
        """
        self.stochastic_function = stochastic_function
        self.bounds = bounds
        self.constraints = constraints
        self.sampler = sampler
        # Gradient with respect to x; ξ is treated as a constant parameter.
        self.gradient = grad(stochastic_function)  # argnum=0 by default
        self.tol = tol
        self.projector_max_iter = projector_max_iter

    def solve(
        self,
        x0: np.ndarray,
        lambda_0: float = 1.0,
        sigma: float = 0.1,
        kappa: float = 0.5,
        max_iter: int = 1000,
        stop_if_stationary: bool = True,
        tol: Optional[float] = None,
        projector_max_iter: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run SGDA starting from x0.

        Args:
            x0: Initial point.
            lambda_0: Initial step size (λ_0 > 0).
            sigma: Armijo parameter (0 < sigma < 1) for sufficient decrease.
            kappa: Step size reduction factor (0 < kappa < 1). When the
                Armijo condition fails, the new step size is kappa * lambda_k.
            max_iter: Maximum number of iterations.
            stop_if_stationary: If True, stop when x_{k+1} is close to x_k.
            tol: Optional override of the default tolerance.
            projector_max_iter: Optional override of the projector max iterations.

        Returns:
            OptimizationResult: Contains the final point, function value
            estimate, success flag, termination message and history of iterates.
        """
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

        x_k = projector(x0)
        lambda_k = lambda_0
        xs: List[np.ndarray] = []

        f_x_k: Optional[np.float64] = None

        for _ in range(max_iter):
            xs.append(x_k)

            xi_k = self.sampler()
    
            grad_f_x_k = self.gradient(x_k, xi_k)
            f_x_k = self.stochastic_function(x_k, xi_k)

            x_k1 = projector(x_k - lambda_k * grad_f_x_k)
            f_x_k1 = self.stochastic_function(x_k1, xi_k)

            if f_x_k1 > f_x_k - sigma * np.dot(grad_f_x_k, x_k - x_k1):
                lambda_k *= kappa

            if stop_if_stationary and np.allclose(
                x_k, x_k1, atol=tol, rtol=tol
            ):
                x_k = x_k1
                f_x_k = f_x_k1
                break

            x_k = x_k1
            f_x_k = f_x_k1

        if f_x_k is None:
            xi_last = self.sampler()
            f_x_k = self.stochastic_function(x_k, xi_last)

        return OptimizationResult(
            x_opt=x_k,
            fun_opt=f_x_k,
            success=len(xs) < max_iter,
            history=xs,
        )
