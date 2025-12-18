import numpy as np
from .utils import ScalarFunction, VectorFunction, OptimizationResult
from autograd import grad
from typing import Optional


class GDA:
    def __init__(
        self,
        function: ScalarFunction,
        projector: VectorFunction,
    ):
        self.function = function
        self.projector = projector
        self.gradient: VectorFunction = grad(function)

    def solve(
        self,
        x0: np.ndarray,
        lambda_0: float = 1.0,
        sigma: float = 0.1,
        kappa: float = 0.75,
        max_iter: int = 1000,
        stop_if_stationary: bool = True,
        tol: Optional[float] = 1e-8,
    ) -> OptimizationResult:
        projector = self.projector
        x_k = projector(x0)
        lambda_k = lambda_0
        f_x_k = self.function(x_k)
        xs = []
        success = False
        for _ in range(max_iter):
            xs.append(x_k.copy())
            grad_f_x_k = self.gradient(x_k)
            x_k1 = projector(x_k - lambda_k * grad_f_x_k)
            f_x_k1 = self.function(x_k1)
            if f_x_k1 > f_x_k - sigma * np.dot(grad_f_x_k, x_k - x_k1):
                lambda_k *= kappa
            if np.allclose(x_k, x_k1, atol=tol, rtol=tol):
                success = True
                if stop_if_stationary:
                    break

            x_k = x_k1
            f_x_k = f_x_k1

        return OptimizationResult(
            x_opt=x_k,
            fun_opt=f_x_k,
            success=success,
            history=xs,
        )


__all__ = ["GDA"]
