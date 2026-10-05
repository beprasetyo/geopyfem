import numpy as np

from elements.elemtype import get_element_type


def _signed_polygon_area(xy):
    """
    Signed polygon area.

    Positive area means counter-clockwise node order.
    Negative area means clockwise node order.
    """
    area = 0.0
    n = len(xy)

    for i in range(n):
        j = (i + 1) % n

        area += (
            xy[i, 0] * xy[j, 1]
            - xy[j, 0] * xy[i, 1]
        )

    return 0.5 * area


def normalize_element_orientation(
    nodes,
    mesh_elements,
    area_tolerance=1.0e-14,
):
    """
    Normalize supported 2D element connectivity to counter-clockwise order.

    Element-specific information such as corner-node positions and the
    reverse permutation is obtained from elements/elemtype.py.

    The element dictionaries are modified in place.

    Returns
    -------
    reoriented : int
        Total number of elements whose orientation was reversed.

    reoriented_by_type : dict
        Number of reversed elements for each element type.
    """
    reoriented = 0
    reoriented_by_type = {}

    for element in mesh_elements:
        element_info = get_element_type(element["type"])

        corner_nodes = element_info.get("corner_nodes")
        reverse_order = element_info.get("reverse_order")

        # Some future element types may not require polygon orientation
        # normalization. In that case these entries can be None.
        if corner_nodes is None or reverse_order is None:
            continue

        c = list(element["connectivity"])

        corner_connectivity = [
            c[local_index]
            for local_index in corner_nodes
        ]

        xy = nodes[corner_connectivity, :]
        signed_area = _signed_polygon_area(xy)

        if abs(signed_area) <= area_tolerance:
            raise ValueError(
                f'Element {element["id"]} ({element["type"]}) has nearly '
                f'zero signed area ({signed_area:.6e}). '
                'The element may be degenerate or self-intersecting.'
            )

        if signed_area < 0.0:
            if len(reverse_order) != len(c):
                raise ValueError(
                    f'Element type "{element["type"]}" has an invalid '
                    'reverse_order definition in elemtype.py.'
                )

            c = [
                c[local_index]
                for local_index in reverse_order
            ]

            element["connectivity"] = c

            reoriented += 1
            reoriented_by_type[element["type"]] = (
                reoriented_by_type.get(element["type"], 0) + 1
            )

    return reoriented, reoriented_by_type


def check_element_jacobians(
    nodes,
    mesh_elements,
    detj_tolerance=1.0e-12,
):
    """
    Check det(J) at the integration points of every supported domain element.

    The gradient shape function and Gauss rule are obtained from
    elements/elemtype.py, so mesh_checker.py does not hard-code Quad4
    integration points or Quad4 shape derivatives.

    Returns
    -------
    minimum_detj : float or None
        Smallest Jacobian determinant found in the mesh.

    minimum_by_type : dict
        Smallest det(J) for each element type.
    """
    minimum_detj = float("inf")
    minimum_by_type = {}

    for element in mesh_elements:
        element_info = get_element_type(element["type"])

        gradshape = element_info["gradshape"]
        gauss_rule = element_info["gauss"]

        gauss_points, _ = gauss_rule()

        c = element["connectivity"]
        xIe = nodes[c, :]

        for q in gauss_points:
            dN = gradshape(q)

            J = np.dot(dN, xIe)
            detJ = np.linalg.det(J)

            minimum_detj = min(minimum_detj, detJ)

            current_min = minimum_by_type.get(
                element["type"],
                float("inf"),
            )

            minimum_by_type[element["type"]] = min(
                current_min,
                detJ,
            )

            if detJ <= detj_tolerance:
                raise ValueError(
                    f'Element {element["id"]} ({element["type"]}) has '
                    f'invalid Jacobian det(J)={detJ:.6e} at natural '
                    f'coordinate {tuple(q)}. '
                    'The element may be inverted, strongly distorted, '
                    'or have invalid connectivity.'
                )

    if minimum_detj == float("inf"):
        minimum_detj = None

    return minimum_detj, minimum_by_type


def check_and_normalize_mesh(
    nodes,
    mesh_elements,
    area_tolerance=1.0e-14,
    detj_tolerance=1.0e-12,
):
    """
    General mesh checker for element types registered in elemtype.py.

    Steps
    -----
    1. Normalize element orientation.
    2. Check Jacobian determinant at the element integration points.

    Returns
    -------
    report : dict
        Mesh-checking summary.
    """
    reoriented, reoriented_by_type = normalize_element_orientation(
        nodes,
        mesh_elements,
        area_tolerance=area_tolerance,
    )

    minimum_detj, minimum_by_type = check_element_jacobians(
        nodes,
        mesh_elements,
        detj_tolerance=detj_tolerance,
    )

    return {
        "reoriented_elements": reoriented,
        "reoriented_by_type": reoriented_by_type,
        "minimum_detJ": minimum_detj,
        "minimum_detJ_by_type": minimum_by_type,
    }


# Backward-compatible alias for the previous main.py.
# It can be removed later after all code uses check_and_normalize_mesh().
def check_and_normalize_quad4_mesh(
    nodes,
    mesh_elements,
    area_tolerance=1.0e-14,
    detj_tolerance=1.0e-12,
):
    return check_and_normalize_mesh(
        nodes,
        mesh_elements,
        area_tolerance=area_tolerance,
        detj_tolerance=detj_tolerance,
    )
