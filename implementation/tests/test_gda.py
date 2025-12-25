from algorithms import GDA, Bounds
import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint
from autograd import grad
from algorithms.utils import Projector


def test_simple_r_to_r():
    # Minimize function f(x) = x^2 - 2x + 3 subjected to bounds [0, 3]
    def f(x):
        return x[0] ** 2 - 2 * x[0] + 3

    bounds = Bounds([0.0], [3.0])
    
    def projector(x):
        # Simple projection onto bounds
        return np.clip(x, bounds.lb, bounds.ub)

    gda = GDA(function=f, projector=projector)

    solution = gda.solve(x0=np.random.rand(1))
    assert solution.success
    assert np.allclose(solution.x_opt, 1.0)


def test_example_1():
    def f(x):
        x1, x2 = x[0], x[1]
        numerator = x1**2 + x2**2 + 3
        denominator = 1 + 2 * x1 + 8 * x2
        return numerator / denominator

    def g2(x):
        x1, x2 = x[0], x[1]
        return x1**2 + 2 * x1 * x2 - 4
    

    bounds = Bounds([0.0, 0.0], [np.inf, np.inf])
    constraints = [NonlinearConstraint(fun=g2, lb=0.0, ub=np.inf, jac=grad(g2))]
    projector = Projector(bounds=bounds, constraints=constraints)
    gda = GDA(function=f, projector=projector)
    solution = gda.solve(x0=np.array([2.0, 2.0]))
    assert solution.success
    assert np.allclose(solution.x_opt, np.array([0.8916, 1.7973]), atol=1e-4)


def test_gemini_1():
    def f(x):
        x1, x2, x3 = x[0], x[1], x[2]
        numerator = x1**2 + 2 * x2**2 + 3 * x3**2
        denominator = 1 + x1 + 2 * x2 + x3
        return numerator / denominator

    bounds = Bounds([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])
    constraints = [LinearConstraint([[2, 1, 3]], lb=-np.inf, ub=6.0)]
    projector = Projector(bounds=bounds, constraints=constraints)
    gda = GDA(function=f, projector=projector)
    solution = gda.solve(x0=np.array([1.0, 1.0, 1.0]))

    assert solution.success
    assert np.allclose(solution.x_opt, np.zeros(3))
