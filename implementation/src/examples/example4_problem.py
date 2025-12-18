from __future__ import annotations
import numpy as np
import math

def make_example4(n: int, q: np.ndarray | None = None, b: float = 16.0, block: int = 10, radius_sq: float = 20.0):
    assert n % block == 0

    if q is None:
        q = np.ones(n, dtype=float)
    q = np.array(q, dtype=float)
    q2 = q * q

    a = np.ones(n, dtype=float)
    a[n // 2:] = 3.0

    def S(x: np.ndarray) -> float:
        # -ln(-f(x)) = sum x_i^2 / q_i^2
        return float(np.sum((x * x) / q2))

    def f(x: np.ndarray) -> float:
        return -math.exp(-S(x))

    def grad_f(x: np.ndarray) -> np.ndarray:
        # ∇f(x) = 2*exp(-S(x)) * (x / q^2)
        s = S(x)
        e = math.exp(-s)
        return (2.0 * e) * (x / q2)

    # inequality constraints g_i(x) = ||block_i||^2 - 20
    m = n // block

    def g_list():
        gs = []
        for j in range(m):
            sl = slice(j * block, (j + 1) * block)
            def g(x, sl=sl):
                xb = x[sl]
                return float(xb @ xb - radius_sq)
            gs.append(g)
        return gs

    def grad_g_list():
        ggs = []
        for j in range(m):
            sl = slice(j * block, (j + 1) * block)
            def gg(x, sl=sl):
                out = np.zeros_like(x, dtype=float)
                out[sl] = 2.0 * x[sl]
                return out
            ggs.append(gg)
        return ggs

    A = a.reshape(1, -1)          # p=1
    bb = np.array([b], dtype=float)

    return {
        "n": n,
        "q": q,
        "a": a,
        "A": A,
        "b": bb,
        "S": S,
        "f": f,
        "grad_f": grad_f,
        "g_list": g_list(),
        "grad_g_list": grad_g_list(),
    }
