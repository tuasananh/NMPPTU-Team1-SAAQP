from typing import Optional
import numpy as np
from .utils import ScalarFunction, VectorFunction, OptimizationResult
from autograd import grad


class GD:
    def __init__(
        self,
        function: ScalarFunction,
        projector: VectorFunction,
    ):
        self.function = function
        self.gradient: VectorFunction = grad(function)
        self.projector = projector

    def solve(
        self,
        x0: np.ndarray,
        step_size: float = 0.1,
        max_iter: int = 1000,
        stop_if_stationary: bool = True,
        tol: Optional[float] = 1e-8,
    ) -> OptimizationResult:
        projector = self.projector
        x_k = projector(x0)
        xs = []
        success = False
        for _ in range(max_iter):
            xs.append(x_k.copy())
            grad_f_x_k = self.gradient(x_k)
            x_k1 = projector(x_k - step_size * grad_f_x_k)
            if np.allclose(x_k, x_k1, atol=tol, rtol=tol):
                success = True
                if stop_if_stationary:
                    break

            x_k = x_k1
        f_x_k = self.function(x_k)

        return OptimizationResult(
            x_opt=x_k,
            fun_opt=f_x_k,
            success=success,
            history=xs,
        )


__all__ = [
    "GD",
]
