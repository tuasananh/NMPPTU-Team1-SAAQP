from typing import List, Optional
import numpy as np
from .utils import ScalarFunction, VectorFunction, Bounds, Constraints, Projector
from autograd import grad


class OptimizationResult:
    """
    Result of the optimization process.

    Attributes:
        x_opt (np.ndarray): The optimal solution found.
        fun_opt (np.float64): The value of the objective function at the optimal solution.
        success (bool): Whether the optimization converged successfully.
        message (str): Description of the cause of the termination.
        history (List[np.ndarray]): List of solution vectors at each iteration.
    """

    def __init__(
        self,
        x_opt: np.ndarray,
        fun_opt: np.float64,
        success: bool,
        history: List[np.ndarray],
    ):
        self.x_opt = x_opt
        self.fun_opt = fun_opt
        self.success = success
        self.message = "Solution converged" if success else "Maximum iterations reached"
        self.history = history


class GDA:
    """
    This class implements a projected gradient descent method to minimize a scalar function
    subject to bounds and constraints.

    Args:
        function (ScalarFunction): The objective function to minimize.
        bounds (Bounds): The bounds on the variables.
        constraints (Constraints): A list of constraints.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-9.
    """

    def __init__(
        self,
        function: ScalarFunction,
        bounds: Bounds,
        constraints: Constraints,
        tol: float = 1e-9,
        projector_max_iter=1000,
    ):
        self.function = function
        self.bounds = bounds
        self.constraints = constraints
        self.gradient: VectorFunction = grad(function)
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
        Solve the optimization problem using Projected Gradient Descent.

        Args:
            x0 (np.ndarray): Initial guess for the solution.
            lambda_0 (float, optional): Initial step size. Defaults to 1.0.
            sigma (float, optional): Parameter for the Armijo condition (sufficient decrease). Defaults to 0.1.
            kappa (float, optional): Step size reduction factor. Defaults to 0.5.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
            stop_if_stationary (bool, optional): Whether to stop if the solution does not change. Defaults to True.

        Returns:
            OptimizationResult: The result of the optimization.
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
        f_x_k = self.function(x_k)
        xs = []
        for _ in range(max_iter):
            xs.append(x_k)
            grad_f_x_k = self.gradient(x_k)
            x_k1 = projector(x_k - lambda_k * grad_f_x_k)
            f_x_k1 = self.function(x_k1)
            if f_x_k1 > f_x_k - sigma * np.dot(grad_f_x_k, x_k - x_k1):
                lambda_k *= kappa
            if stop_if_stationary and np.allclose(x_k, x_k1, atol=tol, rtol=tol):
                break
            x_k = x_k1
            f_x_k = f_x_k1

        return OptimizationResult(
            x_opt=x_k,
            fun_opt=f_x_k,
            success=len(xs) < max_iter,
            history=xs,
        )
