import numpy as np
from scipy.optimize import Bounds
import autograd.numpy as anp
from algorithms.utils import Projector
from scipy.optimize import LinearConstraint, NonlinearConstraint
from autograd import grad


def test_simple_bounds():
    # Original test case
    bounds = Bounds([-1, -2], [1, 2])
    projector = Projector(bounds=bounds)

    x = np.array([2.0, -3.0])
    projected_x = projector(x)

    assert np.allclose(projected_x, np.array([1.0, -2.0]))


def test_scipy_bounds_object():
    # Test using scipy.optimize.Bounds object
    bounds = Bounds([-1, -2], [1, 2])
    projector = Projector(bounds=bounds)

    x = np.array([0.5, 0.0])  # Inside
    assert np.allclose(projector(x), x)

    x = np.array([1.5, 2.5])  # Outside
    assert np.allclose(projector(x), np.array([1.0, 2.0]))


def test_linear_equality_constraint():
    # Project onto line x + y = 1
    # Constraint: x + y - 1 = 0

    def eq_constraint(x):
        return x[0] + x[1] - 1.0

    # Use large bounds effectively unbounded
    bounds = Bounds([-100, -100], [100, 100])

    constraints = [LinearConstraint([[1, 1]], lb=1.0, ub=1.0)]

    projector = Projector(bounds=bounds, constraints=constraints)

    # Project (2, 2) -> (0.5, 0.5)
    x = np.array([2.0, 2.0])
    proj_x = projector(x)

    assert np.allclose(proj_x, np.array([0.5, 0.5]))
    assert np.isclose(eq_constraint(proj_x), 0, atol=1e-5)


def test_linear_inequality_constraint():
    # Project onto half-space x + y >= 1
    # Scipy 'ineq' means fun(x) >= 0

    def ineq_constraint(x):
        return x[0] + x[1] - 1.0

    bounds = Bounds([-100, -100], [100, 100])
    constraints = [LinearConstraint([[1, 1]], lb=1.0, ub=np.inf)]

    projector = Projector(bounds=bounds, constraints=constraints)

    # Point satisfying constraint
    x_in = np.array([2.0, 2.0])  # 2+2-1 = 3 >= 0
    assert np.allclose(projector(x_in), x_in)

    # Point violating constraint
    x_out = np.array([0.0, 0.0])  # 0+0-1 = -1 < 0
    # Should project to boundary x+y=1 -> (0.5, 0.5)
    assert np.allclose(projector(x_out), np.array([0.5, 0.5]))


def test_nonlinear_constraint_circle():
    # Project onto unit circle x^2 + y^2 = 1
    # Use autograd.numpy for automatic gradient calculation

    def circle_constraint(x):
        return anp.sum(x**2) - 1.0

    bounds = Bounds([-2, -2], [2, 2])
    constraints = [NonlinearConstraint(fun=circle_constraint, lb=0.0, ub=0.0, jac=grad(circle_constraint))]

    projector = Projector(bounds=bounds, constraints=constraints)

    # (2, 0) -> (1, 0)
    x = np.array([2.0, 0.0])
    assert np.allclose(projector(x), np.array([1.0, 0.0]), atol=1e-4)

    # (0, 2) -> (0, 1)
    x2 = np.array([0.0, 2.0])
    assert np.allclose(projector(x2), np.array([0.0, 1.0]), atol=1e-4)


def test_bounds_and_constraints_interaction():
    # Box [0, 2] x [0, 2]
    # Constraint x + y = 3
    # Intersection is line segment from (1, 2) to (2, 1)

    bounds = Bounds([0, 0], [2, 2])

    def line_constraint(x):
        return x[0] + x[1] - 3.0

    constraints = [LinearConstraint([[1, 1]], lb=3.0, ub=3.0)]

    projector = Projector(bounds=bounds, constraints=constraints)

    # Point (0, 0).
    # Projection onto line x+y=3 is (1.5, 1.5).
    # (1.5, 1.5) is inside bounds.
    x = np.array([0.0, 0.0])
    assert np.allclose(projector(x), np.array([1.5, 1.5]))

    # Point (3, 3).
    # Projection onto line x+y=3 is (1.5, 1.5).
    x2 = np.array([3.0, 3.0])
    assert np.allclose(projector(x2), np.array([1.5, 1.5]))


def test_multiple_constraints():
    # x >= 0, y >= 0 (via bounds)
    # x + y <= 1 (ineq: 1 - x - y >= 0)
    # x - y = 0 (eq: x = y)
    # Feasible set: segment on line y=x from (0,0) to (0.5, 0.5)

    bounds = Bounds([0, 0], [2, 2])

    constraints = [
        LinearConstraint([[ 1, 1]], lb=-np.inf, ub=1.0),
        LinearConstraint([[ 1, -1]], lb=0.0, ub=0.0),
    ]

    projector = Projector(bounds=bounds, constraints=constraints)

    # Point (1, 0).
    # Closest point on y=x is (0.5, 0.5).
    # (0.5, 0.5) satisfies x+y = 1 <= 1.
    x = np.array([1.0, 0.0])
    assert np.allclose(projector(x), np.array([0.5, 0.5]), atol=1e-4)

    # Point (2, 2).
    # Closest point on y=x is (2, 2).
    # But must satisfy x+y <= 1.
    # Intersection of y=x and x+y<=1 is line segment (0,0) to (0.5, 0.5).
    # Closest point in feasible set to (2,2) is (0.5, 0.5).
    x2 = np.array([2.0, 2.0])
    assert np.allclose(projector(x2), np.array([0.5, 0.5]), atol=1e-4)
