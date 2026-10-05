import numpy as np

from elements.elemtype import get_element_type
from elements.kinematics import build_B_matrix


def assemble_internal_force(
    nodes,
    mesh_elements,
    gauss_states,
    active_element_ids=None,
    ndof_node=2,
):
    """
    Assemble the global internal-force vector from TOTAL Gauss-point stresses.

    Parameters
    ----------
    nodes : ndarray, shape (nnode, 2)
        Global nodal coordinates.

    mesh_elements : list of dict
        GeoPyFEM domain elements.

    gauss_states : dict
        Material-state map indexed as:

            gauss_states[element_id][gauss_point_index]

        This can be either:

            state_manager.committed

        or, during a future Newton iteration:

            state_manager.trial

    ndof_node : int, optional
        Current 2-D continuum formulation uses ux and uy.

    Returns
    -------
    f_internal : ndarray
        Global internal-force vector.

    Notes
    -----
    Effective-stress formulation
    ----------------------------
    Constitutive models store effective stress sigma'.  Pore pressure is a
    separate state variable and equilibrium uses total stress:

        sigma_total = sigma_effective - u I

    For one 2-D element:

        f_int,e = integral(B^T sigma_total dOmega)

    and numerically:

        f_int,e =
            sum_gp(
                B_gp^T
                sigma_gp
                detJ_gp
                w_gp
            )

    The stress vector work-conjugate to the current 2-D B matrix is:

        sigma_2d =
            [sigma_xx,
             sigma_yy,
             tau_xy]

    Although GeoPyFEM stores a complete 3-D plane-strain stress state including
    sigma_zz, sigma_zz does not enter this 2-D internal-force expression
    because the corresponding virtual strain epsilon_zz is zero.

    Future nonlinear use
    --------------------
    This routine deliberately does NOT perform a constitutive update.
    It only integrates stresses that already exist in the supplied
    Gauss-point states.

    During Newton-Raphson:

        constitutive update
            -> trial Gauss-point stress
            -> assemble_internal_force(..., state_manager.trial)
    """
    if ndof_node != 2:
        raise NotImplementedError(
            "GeoPyFEM internal-force assembly currently supports "
            "2 displacement DOFs per node."
        )

    num_nodes = len(nodes)

    f_internal = np.zeros(
        ndof_node * num_nodes,
        dtype=float,
    )

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
        element_info = get_element_type(
            element["type"]
        )

        if element_info["dimension"] != 2:
            continue

        connectivity = element[
            "connectivity"
        ]

        nnode = element_info[
            "nnode"
        ]

        element_ndof_node = element_info[
            "ndof_node"
        ]

        gradshape = element_info[
            "gradshape"
        ]

        gauss_rule = element_info[
            "gauss"
        ]

        if element_ndof_node != ndof_node:
            raise NotImplementedError(
                f'Element "{element["type"]}" uses '
                f"{element_ndof_node} DOFs/node, while internal-force "
                f"assembly is configured for {ndof_node} DOFs/node."
            )

        if len(connectivity) != nnode:
            raise ValueError(
                f'Element {element["id"]} is {element["type"]}, but its '
                f"connectivity contains {len(connectivity)} nodes "
                f"instead of {nnode}."
            )

        if element_id not in gauss_states:
            raise ValueError(
                f"No Gauss-point states were supplied for element "
                f"{element_id}."
            )

        element_states = gauss_states[
            element_id
        ]

        gauss_points, gauss_weights = (
            gauss_rule()
        )

        if len(element_states) != len(
            gauss_points
        ):
            raise ValueError(
                f"Element {element_id} has {len(element_states)} stored "
                f"Gauss-point states but its integration rule has "
                f"{len(gauss_points)} points."
            )

        element_coordinates = nodes[
            connectivity,
            :
        ]

        ndof_element = (
            ndof_node * nnode
        )

        f_internal_element = np.zeros(
            ndof_element,
            dtype=float,
        )

        # ---------------------------------------------------------------
        # Gauss integration:
        #
        #       f_int,e = integral(B^T sigma dOmega)
        # ---------------------------------------------------------------
        for gp_index, (q, w) in enumerate(
            zip(
                gauss_points,
                gauss_weights,
            )
        ):
            B, detJ, _ = build_B_matrix(
                element_coordinates=element_coordinates,
                gradshape=gradshape,
                natural_coordinates=q,
            )

            state = element_states[
                gp_index
            ]

            # Constitutive models evolve EFFECTIVE stress.  Mechanical
            # equilibrium is written in TOTAL stress.  With GeoPyFEM's
            # tension-positive convention and positive compressive pore
            # pressure:
            #
            #     sigma_total = sigma_effective - u I
            sigma_total_2d = state.total_stress_2d

            f_internal_element += (
                B.T
                @ sigma_total_2d
                * detJ
                * w
            )

        # ---------------------------------------------------------------
        # Element internal force -> global internal force
        # ---------------------------------------------------------------
        for local_node, global_node in enumerate(
            connectivity
        ):
            global_dof_x = (
                ndof_node * global_node
            )

            global_dof_y = (
                global_dof_x + 1
            )

            local_dof_x = (
                ndof_node * local_node
            )

            local_dof_y = (
                local_dof_x + 1
            )

            f_internal[
                global_dof_x
            ] += f_internal_element[
                local_dof_x
            ]

            f_internal[
                global_dof_y
            ] += f_internal_element[
                local_dof_y
            ]

    return f_internal
