from .typing import ScalarFunction, VectorFunction
from typing import List
from enum import Enum
from autograd import grad

class ConstraintType(Enum):
    EQUAL_ZERO = 'eq'      # for '=0'
    NON_NEGATIVE = 'ineq'  # for '>=0'

class Constraint: 
    def __init__(self, lhs: ScalarFunction, type: ConstraintType):
        """Create a constraint: lhs = (>=) 0

        Args:
            lhs (ScalarFunction): The left-hand side function of the constraint
            type (ConstraintType): The type of the constraint (=0 or >=0)
        """
        self.type = type
        self.lhs = lhs
        self.grad = grad(lhs)
    
    def to_scipy_constraint(self):
        cons = {
            "type": self.type.value,
            "fun": self.lhs,
            "jac": self.grad
        }
        return cons

Constraints = List[Constraint]