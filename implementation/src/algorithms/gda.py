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
    Projected gradient method with Armijo-type backtracking.

    It solves:
        minimize   f(x)
        subject to x in C

    with:
        x_{k+1} = P_C(x_k - lambda_k * grad f(x_k))
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
        f_x_k = self.function(x_k)
        lambda_k = lambda_0

        xs: List[np.ndarray] = []
        converged = False

        for _ in range(max_iter):
            xs.append(x_k.copy())
            g = self.gradient(x_k)

            lam = lambda_k
            while True:
                x_k1 = projector(x_k - lam * g)

                # nếu bước chiếu không đổi gì nữa -> coi như hội tụ
                if np.allclose(x_k, x_k1, atol=tol, rtol=tol):
                    f_x_k1 = f_x_k
                    converged = True
                    break

                f_x_k1 = self.function(x_k1)

                if f_x_k1 <= f_x_k - sigma * np.dot(g, (x_k - x_k1)):
                    break

                lam *= kappa
                if lam < 1e-20:
                    x_k1 = x_k
                    f_x_k1 = f_x_k
                    converged = True
                    break

            lambda_k = lam

            x_k = x_k1
            f_x_k = f_x_k1

            if converged and stop_if_stationary:
                break

        if len(xs) == 0 or not np.allclose(xs[-1], x_k, atol=tol, rtol=tol):
            xs.append(x_k.copy())

        return OptimizationResult(
            x_opt=x_k,
            fun_opt=f_x_k,
            success=converged,
            history=xs,
        )
