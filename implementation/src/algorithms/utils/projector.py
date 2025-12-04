from typing import List, Optional
from scipy.optimize import Bounds, minimize
from .constraint import Constraints
from autograd import grad
from .typing import Vector, Scalar
import numpy as np

class Projector:
    def __init__(self, bounds: Optional[Bounds] = None, constraints: Constraints = []): 
        """Create a projector for a convex set defined by bounds and constraints

        Args:
            bounds (Bounds): The bounds defining the convex set
            constraints (List[Constraint]): The constraints defining the convex set
        """
        self.bounds = bounds
        self.constraints = [constraint.to_scipy_constraint() for constraint in constraints]
        
    def __call__(self, x: Vector) -> Vector:
        def objective(y: Vector) -> Scalar:
            d = y - x
            return 0.5 * np.dot(d, d)
        def objective_grad(y: Vector) -> Vector:
            return y - x

        res = minimize(
            objective, 
            x0=x,
            jac=objective_grad,
            method="SLSQP",
            bounds=self.bounds,
            constraints=self.constraints,
        )

        assert res.success, f"Projection failed: {res.message}"

        return res.x