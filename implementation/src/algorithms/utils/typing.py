import numpy as np
import numpy.typing as npt
from typing import Any, Callable

Vector = npt.NDArray[np.float64]
Scalar = np.float64
ScalarFunction = Callable[[Vector], Scalar]
VectorFunction = Callable[[Vector], Vector]
StochasticScalarFunction = Callable[[np.ndarray, Any], np.float64]
Sampler = Callable[[], Any]

__all__ = [
    "Vector",
    "Scalar",
    "ScalarFunction",
    "VectorFunction",
    "StochasticScalarFunction",
    "Sampler",
]
