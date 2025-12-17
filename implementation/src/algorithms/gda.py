from typing import List, Optional
import numpy as np
from autograd import grad

from .utils import ScalarFunction, VectorFunction, Bounds, Constraints, Projector


class OptimizationResult:
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
    Notebook-style GDA (Example 5.2 style):

    For x_k, g_k = ∇f(x_k), lr = λ_k:
      x_trial = P(x_k - lr * g_k)
      e = <g_k, x_k - x_trial>
      dl = f(x_trial) - f(x_k) + sigma * e

      if dl > 0: lr <- kappa * lr   (shrink ONCE)
      x_{k+1} = P(x_k - lr * g_k)   (step uses possibly-shrunk lr)
    """

    def __init__(
        self,
        function: ScalarFunction,
        bounds: Bounds,
        constraints: Constraints,
        tol: float = 1e-9,
        projector_max_iter: int = 1000,
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
        kappa: float = 0.75,
        max_iter: int = 1000,
        stop_if_stationary: bool = True,
        tol: Optional[float] = None,
        projector_max_iter: Optional[int] = None,
    ) -> OptimizationResult:
        tol = self.tol if tol is None else tol
        projector_max_iter = self.projector_max_iter if projector_max_iter is None else projector_max_iter

        projector = Projector(
            bounds=self.bounds,
            constraints=self.constraints,
            tol=tol,
            max_iter=projector_max_iter,
        )

        x_k = projector(x0)
        f_k = self.function(x_k)
        lam = float(lambda_0)

        xs: List[np.ndarray] = [x_k.copy()]
        converged = False

        for _ in range(max_iter):
            g = self.gradient(x_k)

            # 1) thử bước với lr hiện tại
            x_trial = projector(x_k - lam * g)
            f_trial = self.function(x_trial)
            e = float(np.dot(g, (x_k - x_trial)))   # <g, x - x_new>

            # f(x_new) - f(x) + sigma*e <= 0
            dl = f_trial - f_k + sigma * e

            # 2) nếu fail -> giảm lr 1 lần rồi bước bằng lr mới
            if dl > 0:
                lam = kappa * lam
                x_trial = projector(x_k - lam * g)
                f_trial = self.function(x_trial)

            # 3) cập nhật
            if stop_if_stationary and np.allclose(x_trial, x_k, atol=tol, rtol=tol):
                converged = True
                x_k = x_trial
                f_k = f_trial
                xs.append(x_k.copy())
                break

            x_k = x_trial
            f_k = f_trial
            xs.append(x_k.copy())

        return OptimizationResult(
            x_opt=x_k,
            fun_opt=f_k,
            success=converged,
            history=xs,
        )
