import numpy as np
import numpy.typing as npt
from typing import Callable

Vector = npt.NDArray[np.float64]
Scalar = np.float64
ScalarFunction = Callable[[Vector], Scalar]
VectorFunction = Callable[[Vector], Vector]
