# src/examples/example4_problem.py
from __future__ import annotations
import numpy as np
import autograd.numpy as anp
import math


def make_example4(
    n: int,
    q: np.ndarray | None = None,
    b: float = 16.0,
    block: int = 10,
    radius_sq: float = 20.0,
):
    assert n % block == 0
    r = math.sqrt(radius_sq)

    if q is None:
        q = np.ones(n, dtype=float)
    q = np.array(q, dtype=float)
    q2 = anp.array(q * q)

    a = np.ones(n, dtype=float)
    a[n // 2 :] = 3.0
    A = a.reshape(1, -1)
    bb = np.array([b], dtype=float)

    # objective (autograd-friendly)
    def f(x):
        x = anp.array(x)
        return -anp.exp(-anp.sum((x * x) / q2))

    # for reporting: S = -ln(-f) = sum x_i^2 / q_i^2
    def S(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum((x * x) / (q * q)))

    # inequalities g_i(x)=||block||^2 - 20
    m = n // block

    g_list = []
    grad_g_list = []

    for j in range(m):
        sl = slice(j * block, (j + 1) * block)

        def g(x, sl=sl):
            xb = x[sl]
            return float(xb @ xb - radius_sq)

        def gg(x, sl=sl):
            out = np.zeros_like(x, dtype=float)
            out[sl] = 2.0 * x[sl]
            return out

        g_list.append(g)
        grad_g_list.append(gg)

    # projection onto block balls
    def proj_blocks(v: np.ndarray) -> np.ndarray:
        x = v.copy()
        for j in range(m):
            sl = slice(j * block, (j + 1) * block)
            nb = np.linalg.norm(x[sl])
            if nb > r:
                x[sl] *= (r / nb)
        return x

    # projection onto C = {a^T x = b} ∩ (product of balls)
    def project_C(y: np.ndarray, tol: float = 1e-12, max_iter: int = 80) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        def eval_dot(nu: float):
            x = proj_blocks(y - nu * a)
            return x, float(a @ x)

        x0, d0 = eval_dot(0.0)
        g0 = d0 - b
        if abs(g0) <= tol:
            return x0

        _, d_plus = eval_dot(+1.0)
        _, d_minus = eval_dot(-1.0)
        decreasing = d_plus < d_minus

        step = 1.0
        direction = (+1.0 if g0 > 0 else -1.0) if decreasing else (-1.0 if g0 > 0 else +1.0)

        lo = 0.0
        g_lo = g0
        hi = direction * step
        x_hi, d_hi = eval_dot(hi)
        g_hi = d_hi - b

        for _ in range(80):
            if g_lo * g_hi <= 0:
                break
            step *= 2.0
            hi = direction * step
            x_hi, d_hi = eval_dot(hi)
            g_hi = d_hi - b
        else:
            # symmetric fallback
            step = 1.0
            lo, hi = -step, +step
            x_lo, d_lo = eval_dot(lo)
            x_hi, d_hi = eval_dot(hi)
            g_lo, g_hi = d_lo - b, d_hi - b
            for _ in range(80):
                if g_lo * g_hi <= 0:
                    break
                step *= 2.0
                lo, hi = -step, +step
                x_lo, d_lo = eval_dot(lo)
                x_hi, d_hi = eval_dot(hi)
                g_lo, g_hi = d_lo - b, d_hi - b
            else:
                raise RuntimeError("Projection failed to bracket nu.")

        if lo > hi:
            lo, hi = hi, lo
            g_lo, g_hi = g_hi, g_lo

        x_mid = None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            x_mid, d_mid = eval_dot(mid)
            g_mid = d_mid - b
            if abs(g_mid) <= tol:
                return x_mid
            if g_lo * g_mid <= 0:
                hi = mid
                g_hi = g_mid
            else:
                lo = mid
                g_lo = g_mid

        return x_mid

    return {
        "n": n,
        "q": q,
        "a": a,
        "A": A,
        "b": bb,
        "f": f,
        "S": S,
        "g_list": g_list,
        "grad_g_list": grad_g_list,
        "project": project_C,
    }
