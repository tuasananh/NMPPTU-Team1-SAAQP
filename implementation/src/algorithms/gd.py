from typing import Optional
import numpy as np
from .utils import ScalarFunction, VectorFunction, OptimizationResult
from autograd import grad
from tqdm import tqdm

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
        with_x_history: bool = True,
        with_f_history: bool = False,
        with_lr_history: bool = False,
        with_tqdm = False,
    ) -> OptimizationResult:
        projector = self.projector
        x_k = projector(x0)
        xs = []
        ys = []
        lrs = []
        success = False
        iter_range = tqdm(range(max_iter), leave=False, desc="GD") if with_tqdm else range(max_iter)
        for _ in iter_range:
            if with_x_history:
                xs.append(x_k.copy())
            if with_f_history:
                ys.append(self.function(x_k))
            if with_lr_history:
                lrs.append(step_size)
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
            f_opt=f_x_k,
            success=success,
            x_history=xs,
            f_history=ys,
            lr_history=lrs,
        )

__all__ = [
    "GD",
]
