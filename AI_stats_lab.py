import numpy as np

# -------------------------------------------------
# Sparse 4 by 4 Joint PMF
# -------------------------------------------------

# The full PMF table stored as a 2D list for quick lookup
# Rows = x (0..3), Cols = y (0..3)
_PMF = [
    [0.10, 0.05, 0.00, 0.00],   # x=0
    [0.15, 0.20, 0.05, 0.00],   # x=1
    [0.00, 0.10, 0.15, 0.05],   # x=2
    [0.00, 0.00, 0.05, 0.10],   # x=3
]

def joint_pmf(x, y):
    """
    Joint PMF table:
             y=0   y=1   y=2   y=3
    x=0      0.10  0.05  0.00  0.00
    x=1      0.15  0.20  0.05  0.00
    x=2      0.00  0.10  0.15  0.05
    x=3      0.00  0.00  0.05  0.10
    """
    if x not in range(4) or y not in range(4):
        return 0.0
    return _PMF[x][y]

def marginal_px(x):
    """
    Compute PX(x) by summing joint_pmf(x, y) over y = 0,1,2,3.
    """
    return sum(joint_pmf(x, y) for y in range(4))

def marginal_py(y):
    """
    Compute PY(y) by summing joint_pmf(x, y) over x = 0,1,2,3.
    """
    return sum(joint_pmf(x, y) for x in range(4))

def conditional_pmf_x_given_y(x, y):
    """
    Compute P(X=x | Y=y) = joint_pmf(x, y) / PY(y).
    If PY(y) is zero, return 0.
    """
    py = marginal_py(y)
    if py == 0:
        return 0.0
    return joint_pmf(x, y) / py

def conditional_distribution_x_given_y(y):
    """
    Return conditional distribution of X given Y=y as a dictionary.
    """
    return {x: conditional_pmf_x_given_y(x, y) for x in range(4)}

def probability_sum_greater_than_3():
    """
    Compute P(X + Y > 3).
    Sum joint_pmf(x, y) for all (x, y) where x + y > 3.
    """
    return sum(
        joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
        if x + y > 3
    )

def independence_check():
    """
    Return True if X and Y are independent.
    X and Y are independent iff joint_pmf(x,y) == PX(x) * PY(y) for all x, y.
    """
    return all(
        np.isclose(joint_pmf(x, y), marginal_px(x) * marginal_py(y))
        for x in range(4)
        for y in range(4)
    )

# -------------------------------------------------
# Expectation, Covariance, and Correlation
# -------------------------------------------------

def expected_x():
    """
    Compute E[X] = sum over x of x * PX(x).
    """
    return sum(x * marginal_px(x) for x in range(4))

def expected_y():
    """
    Compute E[Y] = sum over y of y * PY(y).
    """
    return sum(y * marginal_py(y) for y in range(4))

def expected_xy():
    """
    Compute E[XY] = sum over all (x,y) of x*y * joint_pmf(x,y).
    """
    return sum(
        x * y * joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
    )

def variance_x():
    """
    Compute Var(X) = E[X^2] - (E[X])^2.
    """
    ex2 = sum(x**2 * marginal_px(x) for x in range(4))
    return ex2 - expected_x()**2

def variance_y():
    """
    Compute Var(Y) = E[Y^2] - (E[Y])^2.
    """
    ey2 = sum(y**2 * marginal_py(y) for y in range(4))
    return ey2 - expected_y()**2

def covariance_xy():
    """
    Compute Cov(X,Y) = E[XY] - E[X]*E[Y].
    """
    return expected_xy() - expected_x() * expected_y()

def correlation_xy():
    """
    Compute rho_XY = Cov(X,Y) / sqrt(Var(X) * Var(Y)).
    """
    return covariance_xy() / np.sqrt(variance_x() * variance_y())

def variance_sum():
    """
    Compute Var(X+Y) directly:
    Var(X+Y) = E[(X+Y)^2] - (E[X+Y])^2
    """
    ex_plus_y = expected_x() + expected_y()
    e_sq = sum(
        (x + y)**2 * joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
    )
    return e_sq - ex_plus_y**2

def variance_identity_check():
    """
    Verify: Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y).
    Return True if identity holds (within floating-point tolerance).
    """
    lhs = variance_sum()
    rhs = variance_x() + variance_y() + 2 * covariance_xy()
    return np.isclose(lhs, rhs)


