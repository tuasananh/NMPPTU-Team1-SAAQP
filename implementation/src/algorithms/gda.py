import numpy as np
from .utils import ScalarFunction, VectorFunction, OptimizationResult
from autograd import grad
from typing import Optional
from tqdm import tqdm

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
        with_x_history: bool = True,
        with_f_history: bool = False,
        with_lr_history: bool = False,
        with_tqdm = False,
    ) -> OptimizationResult:
        projector = self.projector
        x_k = projector(x0)
        lambda_k = lambda_0
        f_x_k = self.function(x_k)
        xs = []
        fs = []
        lrs = []
        success = False
        iter_range = tqdm(range(max_iter), leave=False, desc="GDA") if with_tqdm else range(max_iter)
        for _ in iter_range:
            if with_x_history:
                xs.append(x_k.copy())
            if with_f_history:
                fs.append(f_x_k.copy())
            if with_lr_history:
                lrs.append(lambda_k)
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
            f_opt=f_x_k,
            success=success,
            x_history=xs,
            f_history=fs,
            lr_history=lrs,
        )

__all__ = ["GDA"]
