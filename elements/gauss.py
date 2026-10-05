import math
import numpy as np


# =============================================================================
# 1D LINE GAUSS RULES
# =============================================================================

def gauss_line_2():
    """
    Two-point Gauss-Legendre quadrature on the reference line [-1, 1].

    Common use
    ----------
    Line2 boundary element.

    Returns
    -------
    points : ndarray, shape (2, 1)
        Natural coordinates [xi].
    weights : ndarray, shape (2,)
        Integration weights.
    """
    a = 1.0 / math.sqrt(3.0)

    points = np.array([
        [-a],
        [ a],
    ], dtype=float)

    weights = np.array([
        1.0,
        1.0,
    ], dtype=float)

    return points, weights


def gauss_line_3():
    """
    Three-point Gauss-Legendre quadrature on the reference line [-1, 1].

    Common use
    ----------
    Line3 boundary element.

    Returns
    -------
    points : ndarray, shape (3, 1)
        Natural coordinates [xi].
    weights : ndarray, shape (3,)
        Integration weights.
    """
    a = math.sqrt(3.0 / 5.0)

    points = np.array([
        [-a],
        [0.0],
        [ a],
    ], dtype=float)

    weights = np.array([
        5.0 / 9.0,
        8.0 / 9.0,
        5.0 / 9.0,
    ], dtype=float)

    return points, weights


# =============================================================================
# QUADRILATERAL GAUSS RULES
# =============================================================================

def gauss_quad_2x2():
    """
    2x2 Gauss-Legendre quadrature for quadrilateral elements.

    Common use
    ----------
    Quad4
    """
    a = 1.0 / math.sqrt(3.0)

    points = np.array([
        [-a, -a],
        [ a, -a],
        [-a,  a],
        [ a,  a],
    ], dtype=float)

    weights = np.array([
        1.0,
        1.0,
        1.0,
        1.0,
    ], dtype=float)

    return points, weights


# =============================================================================
# TRIANGULAR GAUSS RULES
# =============================================================================

def gauss_tri_3():
    """
    Three-point Gaussian quadrature for the reference triangle.

    Common use
    ----------
    Tri6

    Reference triangle
    ------------------
        (xi, eta) = (0,0), (1,0), (0,1)

    Notes
    -----
    The reference triangle has area 1/2, so the weights sum to 1/2.
    """
    points = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ], dtype=float)

    weights = np.array([
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
    ], dtype=float)

    return points, weights


# =============================================================================
# FUTURE GAUSS RULES
# =============================================================================
#
# Possible future additions:
#
#   gauss_line_4()
#   gauss_line_5()      # candidate rule for future higher-order line elements
#
#   gauss_quad_3x3()
#
#   gauss_tri_1()
#   gauss_tri_6()
#   gauss_tri_7()
#   ...                 # select the rule needed by future Tri15 formulation
#
# Important:
#   Number of element nodes and number of Gauss points are separate choices.
#   A future Line5 does not automatically mean that exactly 5 Gauss points
#   must be used, and a future Tri15 does not automatically use 15 points.
# =============================================================================


if __name__ == "__main__":
    for name, rule in (
        ("Line2", gauss_line_2),
        ("Line3", gauss_line_3),
        ("Quad4", gauss_quad_2x2),
        ("Tri6", gauss_tri_3),
    ):
        points, weights = rule()
        print(f"{name}:")
        print("points =")
        print(points)
        print("weights =", weights)
        print("sum(weights) =", np.sum(weights))
        print()
