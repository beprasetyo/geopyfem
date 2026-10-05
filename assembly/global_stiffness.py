import numpy as np

from elements.elemtype import get_element_type
from elements.kinematics import build_B_matrix


def assemble_global_stiffness(
    nodes,
    mesh_elements,
    material_models,
    active_element_ids=None,
    ndof_node=2,
):
    """
    Assemble the global stiffness matrix.

    Parameters
    ----------
    nodes : ndarray, shape (nnode, 2)
        Global nodal coordinates.
    mesh_elements : list of dict
        Domain elements returned by gmsh_parser.py.
    material_models : dict
        Constitutive model objects indexed by material region.
        Each model must provide:

            tangent_matrix(...)

    ndof_node : int, optional
        Number of global displacement DOFs per node.
        Current 2-D continuum solver uses 2.

    Returns
    -------
    K : ndarray
        Global stiffness matrix.

    Notes
    -----
    GeoPyFEM element formulation decides:
        - number of nodes
        - gradshape()
        - Gauss integration rule

    The material model decides:
        - constitutive/tangent matrix C

    This keeps global assembly independent of a specific element type and
    independent of a specific constitutive model.

    For future nonlinear/plastic analyses this function can be extended with
    Gauss-point state variables and an iteration state. The current version
    assembles a stiffness matrix from the tangent returned by each material
    model without updating history variables.
    """
    num_nodes = len(nodes)

    K = np.zeros((
        ndof_node * num_nodes,
        ndof_node * num_nodes,
    ))

    active_element_ids = (
        None
        if active_element_ids is None
        else {
            int(element_id)
            for element_id in active_element_ids
        }
    )

    for element in mesh_elements:
        element_id = int(
            element["id"]
        )

        if (
            active_element_ids is not None
            and element_id not in active_element_ids
        ):
            continue
        c = element["connectivity"]

        # ---------------------------------------------------------------
        # Element formulation
        # ---------------------------------------------------------------
        element_info = get_element_type(element["type"])

        nnode = element_info["nnode"]
        element_ndof_node = element_info["ndof_node"]
        gradshape = element_info["gradshape"]
        gauss_rule = element_info["gauss"]

        if element_ndof_node != ndof_node:
            raise NotImplementedError(
                f'Element "{element["type"]}" uses '
                f"{element_ndof_node} DOFs/node, while global assembly "
                f"is configured for {ndof_node} DOFs/node."
            )

        if len(c) != nnode:
            raise ValueError(
                f'Element {element["id"]} is {element["type"]} but its '
                f'connectivity contains {len(c)} nodes instead of {nnode}.'
            )

        # ---------------------------------------------------------------
        # Material model
        # ---------------------------------------------------------------
        material_region = element["material"]

        if material_region not in material_models:
            raise ValueError(
                f'Element {element["id"]} uses material region '
                f'"{material_region}", but no material model is assigned '
                "to that region."
            )

        material_model = material_models[material_region]

        # For the current linear/static implementation C is constant for
        # the element. A future nonlinear material can make this dependent
        # on a Gauss-point state.
        C = material_model.tangent_matrix()

        # ---------------------------------------------------------------
        # Element stiffness
        # ---------------------------------------------------------------
        gauss_points, gauss_weights = gauss_rule()

        xIe = nodes[c, :]

        ndof_element = ndof_node * nnode

        Ke = np.zeros((
            ndof_element,
            ndof_element,
        ))

        for q, w in zip(gauss_points, gauss_weights):
            # -----------------------------------------------------------
            # Shared GeoPyFEM kinematics
            #
            # The same B-matrix routine is used by stiffness assembly
            # and post-processing.
            # -----------------------------------------------------------
            B, detJ, _ = build_B_matrix(
                element_coordinates=xIe,
                gradshape=gradshape,
                natural_coordinates=q,
            )

            Ke += (
                np.dot(
                    np.dot(B.T, C),
                    B,
                )
                * detJ
                * w
            )

        # ---------------------------------------------------------------
        # Ke -> global K
        # ---------------------------------------------------------------
        for i, I in enumerate(c):
            for j, Jnode in enumerate(c):
                K[2*I, 2*Jnode] += Ke[2*i, 2*j]
                K[2*I+1, 2*Jnode] += Ke[2*i+1, 2*j]
                K[2*I+1, 2*Jnode+1] += Ke[2*i+1, 2*j+1]
                K[2*I, 2*Jnode+1] += Ke[2*i, 2*j+1]

    return K
