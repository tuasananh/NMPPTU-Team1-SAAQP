import numpy as np
from pathlib import Path
from sklearn.datasets import load_svmlight_file


# Path of this file:
# .../implementation/src/algorithms/datasets/libsvm_loader.py
THIS_FILE = Path(__file__).resolve()

# .../implementation/src
SRC_DIR = THIS_FILE.parents[2]

# .../implementation
IMPLEMENTATION_DIR = SRC_DIR.parent

# .../implementation/data/libsvm
DATA_DIR = IMPLEMENTATION_DIR / "data" / "libsvm"


def _load_libsvm(relative_path: str):
    """
    Load a LIBSVM dataset stored under implementation/data/libsvm.

    relative_path examples:
        "mushrooms/mushrooms"
        "w8a/w8a"
        "w8a/w8a.t"
    """
    path = DATA_DIR / relative_path

    if not path.exists():
        raise FileNotFoundError(f"LIBSVM file not found: {path}")

    X, y = load_svmlight_file(str(path))
    X = X.toarray().astype(np.float64)

    # Ensure labels are in {-1, +1}
    uniq = np.unique(y)
    if set(uniq) == {0.0, 1.0}:
        y = np.where(y == 0.0, -1.0, 1.0)
    else:
        y = y.astype(np.float64)

    return X, y


def load_mushrooms():
    """
    Load the Mushrooms dataset from:

        implementation/data/libsvm/mushrooms/mushrooms
    """
    return _load_libsvm("mushrooms/mushrooms")


def load_w8a(train: bool = True):
    """
    Load the W8a dataset from:

        implementation/data/libsvm/w8a/w8a   (train=True)
        implementation/data/libsvm/w8a/w8a.t (train=False)
    """
    if train:
        return _load_libsvm("w8a/w8a")
    else:
        return _load_libsvm("w8a/w8a.t")
