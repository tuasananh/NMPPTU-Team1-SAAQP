from typing import List, Optional, Union
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

class Projector:
    def __init__(
        self,
        bounds: Optional[Bounds] = None,
        constraints: List[Union[LinearConstraint, NonlinearConstraint]] = [],
    ):
        self.bounds = bounds
        self.constraints = constraints

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Objective: 0.5 * ||y - x||^2
        def objective(y):
            return 0.5 * np.sum(np.square(y - x))

        # Gradient: y - x
        def objective_grad(y):
            return y - x

        x0 = x.copy()
        x0 = np.clip(x0, self.bounds.lb, self.bounds.ub) if self.bounds is not None else x0
        
        res = minimize(
            objective,
            x0=x0,
            jac=objective_grad,
            method="SLSQP",
            bounds=self.bounds,
            constraints=self.constraints,
            options={"maxiter": 1000},
        )

        if not res.success:
            # Fallback or warning
            print(f"Projection warning: {res.message} {res.x}")
            
        return res.x