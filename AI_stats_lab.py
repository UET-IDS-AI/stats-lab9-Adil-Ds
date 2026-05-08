import numpy as np

_PMF = [
    [0.10, 0.05, 0.00, 0.00],
    [0.15, 0.20, 0.05, 0.00],
    [0.00, 0.10, 0.15, 0.05],
    [0.00, 0.00, 0.05, 0.10],
]

def joint_pmf(x, y):
    if x not in range(4) or y not in range(4):
        return 0.0
    return _PMF[x][y]

def marginal_px(x):
    return sum(joint_pmf(x, y) for y in range(4))

def marginal_py(y):
    return sum(joint_pmf(x, y) for x in range(4))

def conditional_pmf_x_given_y(x, y):
    py = marginal_py(y)
    if py == 0:
        return 0.0
    return joint_pmf(x, y) / py

def conditional_distribution_x_given_y(y):
    return {x: conditional_pmf_x_given_y(x, y) for x in range(4)}

def probability_sum_greater_than_3():
    return sum(
        joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
        if x + y > 3
    )

def independence_check():
    return all(
        np.isclose(joint_pmf(x, y), marginal_px(x) * marginal_py(y))
        for x in range(4)
        for y in range(4)
    )

def expected_x():
    return sum(x * marginal_px(x) for x in range(4))

def expected_y():
    return sum(y * marginal_py(y) for y in range(4))

def expected_xy():
    return sum(
        x * y * joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
    )

def variance_x():
    ex2 = sum(x**2 * marginal_px(x) for x in range(4))
    return ex2 - expected_x()**2

def variance_y():
    ey2 = sum(y**2 * marginal_py(y) for y in range(4))
    return ey2 - expected_y()**2

def covariance_xy():
    return expected_xy() - expected_x() * expected_y()

def correlation_xy():
    return covariance_xy() / np.sqrt(variance_x() * variance_y())

def variance_sum():
    ex_plus_y = expected_x() + expected_y()
    e_sq = sum(
        (x + y)**2 * joint_pmf(x, y)
        for x in range(4)
        for y in range(4)
    )
    return e_sq - ex_plus_y**2

def variance_identity_check():
    lhs = variance_sum()
    rhs = variance_x() + variance_y() + 2 * covariance_xy()
    return bool(np.isclose(lhs, rhs))
