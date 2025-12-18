from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RNNParams:
    dt: float = 1e-2
    it_max: int = 1000
    tol_ineq: float = 1e-10
    tol_eq: float = 1e-10
    tol_sign: float = 1e-10


def _sign_subgrad(s: float, tol: float) -> float:
    if s > tol:
        return 1.0
    if s < -tol:
        return -1.0
    return 0.0


def _c_adjusted(g_vals: np.ndarray, r: np.ndarray, tol_ineq: float, tol_eq: float) -> float:
    if np.any(g_vals > tol_ineq):
        return 0.0
    if np.any(np.abs(r) > tol_eq):
        return 0.0
    return 1.0


def rnn_solve_paper(
    x0: np.ndarray,
    grad_f,
    g_list,
    grad_g_list,
    A: np.ndarray,
    b: np.ndarray,
    params: RNNParams = RNNParams(),
):
    x = np.array(x0, dtype=float).copy()
    n = x.size
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    p = int(A.shape[0])
    assert A.shape[1] == n
    assert b.shape == (p,)
    assert len(g_list) == len(grad_g_list)

    hist = [x.copy()]

    for _ in range(params.it_max):
        g_vals = np.array([gi(x) for gi in g_list], dtype=float)  # (m,)
        r = A @ x - b                                            # (p,)

        c = _c_adjusted(g_vals, r, params.tol_ineq, params.tol_eq)

        subP = np.zeros(n, dtype=float)
        for gv, ggi in zip(g_vals, grad_g_list):
            if gv > params.tol_ineq:
                subP += ggi(x)

        subL1 = np.zeros(n, dtype=float)
        for i in range(p):
            si = _sign_subgrad(float(r[i]), params.tol_sign)
            subL1 += si * A[i]

        v = c * grad_f(x) + subP + subL1
        x = x - params.dt * v
        hist.append(x.copy())

    return x, hist
