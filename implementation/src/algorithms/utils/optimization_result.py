from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
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

    x_opt: np.ndarray
    f_opt: np.float64
    success: bool
    x_history: List[np.ndarray] 
    f_history: List[np.ndarray]
    lr_history: List[float]
    
__all__ = ["OptimizationResult"]
