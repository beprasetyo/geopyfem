import numpy as np

from elements.gauss import (
    gauss_line_2,
    gauss_line_3,
    gauss_quad_2x2,
    gauss_tri_3,
)


# =============================================================================
# LINE2
# =============================================================================

def line2_shape(xi):
    """
    Linear 2-node line shape functions.

    Gmsh local-node convention:
        node 0 = xi = -1
        node 1 = xi = +1
    """
    x = float(np.asarray(xi).reshape(-1)[0])

    return np.array([
        0.5 * (1.0 - x),
        0.5 * (1.0 + x),
    ])


def line2_gradshape(xi):
    """
    Derivatives of Line2 shape functions with respect to xi.

    Returns
    -------
    dN : ndarray, shape (1, 2)
    """
    return np.array([
        [-0.5, 0.5],
    ])


# =============================================================================
# LINE3
# =============================================================================

def line3_shape(xi):
    """
    Quadratic 3-node line shape functions.

    Gmsh Line3 local-node convention:
        node 0 = xi = -1   first endpoint
        node 1 = xi = +1   second endpoint
        node 2 = xi =  0   midside node

    Natural layout:
        0 -------- 2 -------- 1
       -1          0         +1
    """
    x = float(np.asarray(xi).reshape(-1)[0])

    return np.array([
        0.5 * x * (x - 1.0),
        0.5 * x * (x + 1.0),
        1.0 - x**2,
    ])


def line3_gradshape(xi):
    """
    Derivatives of Line3 shape functions with respect to xi.

    Returns
    -------
    dN : ndarray, shape (1, 3)
    """
    x = float(np.asarray(xi).reshape(-1)[0])

    return np.array([[
        x - 0.5,
        x + 0.5,
        -2.0 * x,
    ]])


# =============================================================================
# QUAD4
# =============================================================================

def quad4_shape(xi):
    """
    Quad4 shape functions.

    Local natural coordinates:
        node 0 = (-1, -1)
        node 1 = ( 1, -1)
        node 2 = ( 1,  1)
        node 3 = (-1,  1)
    """
    x, y = tuple(xi)

    N = [
        (1.0-x)*(1.0-y),
        (1.0+x)*(1.0-y),
        (1.0+x)*(1.0+y),
        (1.0-x)*(1.0+y),
    ]

    return 0.25 * np.array(N)


def quad4_gradshape(xi):
    """
    Gradient of Quad4 shape functions with respect to natural coordinates.

    Returns
    -------
    dN : ndarray, shape (2, 4)
    """
    x, y = tuple(xi)

    dN = [
        [-(1.0-y),  (1.0-y),  (1.0+y), -(1.0+y)],
        [-(1.0-x), -(1.0+x),  (1.0+x),  (1.0-x)],
    ]

    return 0.25 * np.array(dN)


# =============================================================================
# TRI6
# =============================================================================

def tri6_shape(xi):
    """
    Quadratic 6-node triangular shape functions.

    Reference triangle:
        node 0 = (0, 0)
        node 1 = (1, 0)
        node 2 = (0, 1)

    Gmsh Tri6 local-node convention:
        node 3 = midside node on edge 0-1
        node 4 = midside node on edge 1-2
        node 5 = midside node on edge 2-0
    """
    x, y = tuple(xi)

    L1 = 1.0 - x - y
    L2 = x
    L3 = y

    return np.array([
        L1 * (2.0 * L1 - 1.0),
        L2 * (2.0 * L2 - 1.0),
        L3 * (2.0 * L3 - 1.0),
        4.0 * L1 * L2,
        4.0 * L2 * L3,
        4.0 * L3 * L1,
    ])


def tri6_gradshape(xi):
    """
    Gradient of Tri6 shape functions with respect to natural coordinates.

    Returns
    -------
    dN : ndarray, shape (2, 6)
    """
    x, y = tuple(xi)

    L1 = 1.0 - x - y

    dN_dxi = [
        1.0 - 4.0 * L1,
        4.0 * x - 1.0,
        0.0,
        4.0 * (L1 - x),
        4.0 * y,
        -4.0 * y,
    ]

    dN_deta = [
        1.0 - 4.0 * L1,
        0.0,
        4.0 * y - 1.0,
        -4.0 * x,
        4.0 * x,
        4.0 * (L1 - y),
    ]

    return np.array([
        dN_dxi,
        dN_deta,
    ])


# =============================================================================
# ELEMENT REGISTRY
# =============================================================================

ELEMENT_TYPES = {
    # -------------------------------------------------------------------------
    # Boundary elements
    # -------------------------------------------------------------------------
    "Line2": {
        "dimension": 1,
        "nnode": 2,
        "ndof_node": 2,
        "shape": line2_shape,
        "gradshape": line2_gradshape,
        "gauss": gauss_line_2,

        # Not used by the current 2-D domain mesh checker/postprocessor.
        "corner_nodes": None,
        "reverse_order": None,
        "plot_triangles": [],
    },

    "Line3": {
        "dimension": 1,
        "nnode": 3,
        "ndof_node": 2,
        "shape": line3_shape,
        "gradshape": line3_gradshape,
        "gauss": gauss_line_3,

        # Boundary orientation is not currently normalized because global
        # traction components tx, ty are independent of edge direction.
        "corner_nodes": None,
        "reverse_order": None,
        "plot_triangles": [],
    },

    # -------------------------------------------------------------------------
    # 2-D continuum elements
    # -------------------------------------------------------------------------
    "Quad4": {
        "dimension": 2,
        "nnode": 4,
        "ndof_node": 2,
        "shape": quad4_shape,
        "gradshape": quad4_gradshape,
        "gauss": gauss_quad_2x2,

        "corner_nodes": [0, 1, 2, 3],
        "reverse_order": [0, 3, 2, 1],

        "plot_triangles": [
            [0, 1, 2],
            [0, 2, 3],
        ],
    },

    "Tri6": {
        "dimension": 2,
        "nnode": 6,
        "ndof_node": 2,
        "shape": tri6_shape,
        "gradshape": tri6_gradshape,
        "gauss": gauss_tri_3,

        "corner_nodes": [0, 1, 2],

        # Gmsh Tri6:
        # corners  = 0, 1, 2
        # midsides = 3:(0-1), 4:(1-2), 5:(2-0)
        "reverse_order": [0, 2, 1, 5, 4, 3],

        "plot_triangles": [
            [0, 3, 5],
            [3, 1, 4],
            [5, 4, 2],
            [3, 4, 5],
        ],
    },

    # =========================================================================
    # FUTURE HIGHER-ORDER ELEMENTS
    # =========================================================================
    #
    # Tri15 (complete fourth-order triangle) will require:
    #
    #   tri15_shape()
    #   tri15_gradshape()
    #   an appropriate triangle Gauss rule from gauss.py
    #   verified Gmsh local-node ordering
    #   verified reverse_order
    #   plot_triangles
    #
    # Its boundary will normally be a 5-node fourth-order line element,
    # so a future Line5 entry should also be implemented.
    #
    # "Line5": {
    #     "dimension": 1,
    #     "nnode": 5,
    #     "ndof_node": 2,
    #     "shape": line5_shape,
    #     "gradshape": line5_gradshape,
    #     "gauss": gauss_line_...,
    #     "corner_nodes": None,
    #     "reverse_order": None,
    #     "plot_triangles": [],
    # },
    #
    # "Tri15": {
    #     "dimension": 2,
    #     "nnode": 15,
    #     "ndof_node": 2,
    #     "shape": tri15_shape,
    #     "gradshape": tri15_gradshape,
    #     "gauss": gauss_tri_...,
    #     "corner_nodes": [0, 1, 2],
    #     "reverse_order": [...],
    #     "plot_triangles": [...],
    # },
    #
    # When Tri15/Line5 are activated, gmsh_parser.py must also contain the
    # corresponding Gmsh element-type IDs and node counts.
    # =========================================================================
}


def get_element_type(element_type):
    """
    Return the definition of an implemented element type.
    """
    if element_type not in ELEMENT_TYPES:
        supported = ", ".join(ELEMENT_TYPES.keys())

        raise NotImplementedError(
            f'Element type "{element_type}" is not implemented. '
            f"Currently supported: {supported}"
        )

    return ELEMENT_TYPES[element_type]


def supported_element_types():
    """
    Return implemented element type names.
    """
    return tuple(ELEMENT_TYPES.keys())


if __name__ == "__main__":
    print("Supported element types:")
    print(supported_element_types())

    # Shape-function sanity checks.
    test_cases = [
        ("Line2", np.array([0.2])),
        ("Line3", np.array([0.2])),
        ("Quad4", np.array([0.2, -0.1])),
        ("Tri6", np.array([0.2, 0.3])),
    ]

    for element_name, point in test_cases:
        info = get_element_type(element_name)
        N = info["shape"](point)
        dN = info["gradshape"](point)

        print(f"\n{element_name} at {point}:")
        print("sum(N) =", np.sum(N))
        print("sum(grad rows) =", np.sum(dN, axis=1))
