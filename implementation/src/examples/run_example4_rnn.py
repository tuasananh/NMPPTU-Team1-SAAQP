from __future__ import annotations

import time
import numpy as np
from autograd import grad

from examples.example4_problem import make_example4
from algorithms.rnn import rnn_solve_paper, RNNParams


def _get_grad_f(prob):
    if "grad_f" in prob and prob["grad_f"] is not None:
        return prob["grad_f"]
    gf = grad(prob["f"])
    return lambda x: np.asarray(gf(x), dtype=float)


def main():
    ns = [10, 20, 50, 100, 300, 400, 600]
    rng = np.random.default_rng(1)

    for n in ns:
        prob = make_example4(n=n, q=np.ones(n))
        x0 = 0.1 * rng.standard_normal(n)   # scale nhỏ để tránh vanishing gradient
        x0 = prob["project"](x0)            # đưa về feasible để RNN có c(x)=1 ngay


        grad_f = _get_grad_f(prob)

        t0 = time.perf_counter()
        x_star, _ = rnn_solve_paper(
            x0=x0,
            grad_f=grad_f,
            g_list=prob["g_list"],
            grad_g_list=prob["grad_g_list"],
            A=prob["A"],
            b=prob["b"],
            params=RNNParams(dt=1e-2, it_max=1000),
        )
        t1 = time.perf_counter()

        print(f"n={n:4d}  RNN it=1000  -ln(-f)={prob['S'](x_star):.6f}  time={t1-t0:.4f}s")


if __name__ == "__main__":
    main()
