import numpy as np


def build_B_matrix(
    element_coordinates,
    gradshape,
    natural_coordinates,
):
    """
    Build the small-strain 2-D strain-displacement matrix B.

    This is the SINGLE B-matrix implementation used by GeoPyFEM for both:

        - stiffness assembly
        - strain/stress post-processing

    Parameters
    ----------
    element_coordinates : ndarray, shape (nnode, 2)
        Global x-y coordinates of the element nodes.

    gradshape : callable
        Element shape-function gradient with respect to natural coordinates.

        It must return:

            dN/dxi

        in an array with shape:

            (2, nnode)

        for a 2-D continuum element.

    natural_coordinates : array-like
        Natural coordinates of the integration/Gauss point.

        Examples
        --------
        Quad4:
            [xi, eta]

        Tri6:
            [xi, eta]

    Returns
    -------
    B : ndarray, shape (3, 2*nnode)
        Small-strain engineering-strain matrix:

            strain = B @ ue

        with strain ordering:

            [epsilon_xx,
             epsilon_yy,
             gamma_xy]

        where:

            gamma_xy = 2 * epsilon_xy

    detJ : float
        Determinant of the element Jacobian.

    dN_global : ndarray, shape (2, nnode)
        Shape-function gradients with respect to global coordinates:

            [dN/dx,
             dN/dy]

    Notes
    -----
    The kinematic transformation is:

        dN/d(x,y) = inv(J) @ dN/d(xi,eta)

    and the B matrix is:

        [dN1/dx      0      dN2/dx      0      ...]
        [     0 dN1/dy           0 dN2/dy      ...]
        [dN1/dy dN1/dx      dN2/dy dN2/dx      ...]

    Keeping this routine in one module prevents stiffness assembly and
    post-processing from silently using different strain-displacement
    formulations as GeoPyFEM develops toward nonlinear constitutive models.
    """
    element_coordinates = np.asarray(
        element_coordinates,
        dtype=float,
    )

    dN = np.asarray(
        gradshape(natural_coordinates),
        dtype=float,
    )

    if element_coordinates.ndim != 2 or element_coordinates.shape[1] != 2:
        raise ValueError(
            "element_coordinates must have shape (nnode, 2)."
        )

    if dN.ndim != 2 or dN.shape[0] != 2:
        raise ValueError(
            "gradshape() for a 2-D continuum element must return "
            "an array with shape (2, nnode)."
        )

    if dN.shape[1] != element_coordinates.shape[0]:
        raise ValueError(
            "Number of shape-function gradients does not match "
            "the number of element nodes."
        )

    # ------------------------------------------------------------------
    # Natural coordinates -> global coordinates
    # ------------------------------------------------------------------
    J = np.dot(
        dN,
        element_coordinates,
    )

    detJ = np.linalg.det(J)

    if detJ <= 0.0:
        raise ValueError(
            "Non-positive element Jacobian in GeoPyFEM kinematics: "
            f"det(J)={detJ:.6e}."
        )

    dN_global = np.dot(
        np.linalg.inv(J),
        dN,
    )

    # ------------------------------------------------------------------
    # Strain-displacement matrix
    # ------------------------------------------------------------------
    nnode = element_coordinates.shape[0]

    B = np.zeros((
        3,
        2 * nnode,
    ))

    B[0, 0::2] = dN_global[0, :]
    B[1, 1::2] = dN_global[1, :]

    B[2, 0::2] = dN_global[1, :]
    B[2, 1::2] = dN_global[0, :]

    return B, detJ, dN_global
