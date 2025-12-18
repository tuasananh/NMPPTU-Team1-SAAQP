from __future__ import annotations

import inspect
import time
import numpy as np
from scipy.optimize import Bounds

from examples.example4_problem import make_example4
from algorithms.gda import GDA


class _EqAxb:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)

    def to_scipy_constraint(self):
        # SciPy SLSQP: 'eq' means fun(x) == 0
        return {
            "type": "eq",
            "fun": lambda x: self.A @ x - self.b,
            "jac": lambda x: self.A,
        }


class _BlockBallIneq:
    def __init__(self, sl: slice, radius_sq: float):
        self.sl = sl
        self.radius_sq = float(radius_sq)

    def to_scipy_constraint(self):
        # SciPy SLSQP: 'ineq' means fun(x) >= 0
        # original: sum(x_block^2) - radius_sq <= 0
        # => radius_sq - sum(x_block^2) >= 0
        def fun(x):
            xb = x[self.sl]
            return self.radius_sq - float(xb @ xb)

        def jac(x):
            g = np.zeros_like(x, dtype=float)
            g[self.sl] = -2.0 * x[self.sl]
            return g

        return {"type": "ineq", "fun": fun, "jac": jac}


def main():
    ns = [10, 20, 50, 100, 300, 400, 600]
    rng = np.random.default_rng(1)

    sigma = 0.1
    lambda0 = 1.0
    kappa = 0.5
    it_max = 10

    block = 10
    radius_sq = 20.0

    for n in ns:
        prob = make_example4(n=n, q=np.ones(n))
        x0 = 0.1 * rng.standard_normal(n)   # scale nhỏ để tránh vanishing gradient
        x0 = prob["project"](x0)            # đưa về feasible để RNN có c(x)=1 ngay


        # Bounds (unbounded)
        lb = -1e30 * np.ones(n, dtype=float)
        ub = +1e30 * np.ones(n, dtype=float)
        bounds = Bounds(lb, ub)

        # Constraints list (objects with to_scipy_constraint)
        cons = [_EqAxb(prob["A"], prob["b"])]
        m = n // block
        for j in range(m):
            sl = slice(j * block, (j + 1) * block)
            cons.append(_BlockBallIneq(sl, radius_sq))

        gda = GDA(function=prob["f"], bounds=bounds, constraints=cons)

        for name, val in [
            ("sigma", sigma),
            ("lambda0", lambda0),
            ("kappa", kappa),
            ("it_max", it_max),
            ("max_iter", it_max),
        ]:
            if hasattr(gda, name):
                try:
                    setattr(gda, name, val)
                except Exception:
                    pass

        solve_sig = inspect.signature(gda.solve)
        cand = {
            "sigma": sigma,
            "lambda0": lambda0,
            "kappa": kappa,
            "it_max": it_max,
            "max_iter": it_max,
        }
        solve_kwargs = {k: v for k, v in cand.items() if k in solve_sig.parameters}

        t0 = time.perf_counter()
        res = gda.solve(x0, **solve_kwargs)
        t1 = time.perf_counter()

        x_star = res.x_opt
        val = prob["S"](x_star)  # = -ln(-f)

        print(f"n={n:4d}  GDA it={it_max:3d}  -ln(-f)={val:.6f}  time={t1-t0:.4f}s")


if __name__ == "__main__":
    main()
