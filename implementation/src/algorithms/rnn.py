from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class RNNParams:
    dt: float = 1e-2
    it_max: int = 1000
    tol_ineq: float = 1e-10          # dùng cho "g_i(x) <= 0"
    tol_eq_abs: float = 1e-10        # dùng cho "|Ax-b| <= 0" trong c(x)
    tol_sign: float = 1e-10          # dùng cho sign trong ∂||Ax-b||_1

def _sign_subgrad(s: float, tol: float) -> float:
    # chọn phần tử trong [-1,1] tại 0 là 0
    if s > tol:
        return 1.0
    if s < -tol:
        return -1.0
    return 0.0

def _c_adjusted(g_vals: np.ndarray, Ax_minus_b: np.ndarray, tol_ineq: float, tol_eq_abs: float) -> float:
    # paper: c(x) = Π_i c_i, c_i ∈ 1 - Ψ1(J_i(x))
    # với J = (g_1,...,g_m, |A1x-b1|,...,|Apx-bp|)
    # chọn: Ψ1(s)=1 nếu s>tol, Ψ1(s)=0 nếu s<=tol  => 1-Ψ1 = 0 khi vi phạm, 1 khi không vi phạm
    if np.any(g_vals > tol_ineq):
        return 0.0
    if np.any(np.abs(Ax_minus_b) > tol_eq_abs):
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
    p = int(A.shape[0])
    assert A.shape[1] == n
    assert b.shape == (p,)
    assert len(g_list) == len(grad_g_list)

    hist = [x.copy()]

    for _ in range(params.it_max):
        g_vals = np.array([gi(x) for gi in g_list], dtype=float)   # (m,)
        r = A @ x - b                                             # (p,)

        # c(x)
        c = _c_adjusted(g_vals, r, params.tol_ineq, params.tol_eq_abs)

        # ∂P(x): chọn phần tử: cộng ∇g_i nếu g_i(x) > 0, ngược lại 0
        subP = np.zeros(n, dtype=float)
        for gv, ggi in zip(g_vals, grad_g_list):
            if gv > params.tol_ineq:
                subP += ggi(x)

        # ∂||Ax-b||_1 = Σ (2Ψ1(A_i x - b_i) - 1) A_i^T
        # chọn phần tử tại 0 là 0 tương đương "sign subgradient"
        subL1 = np.zeros(n, dtype=float)
        for i in range(p):
            si = _sign_subgrad(float(r[i]), params.tol_sign)
            subL1 += si * A[i]

        v = c * grad_f(x) + subP + subL1
        x = x - params.dt * v
        hist.append(x.copy())

    return x, hist
