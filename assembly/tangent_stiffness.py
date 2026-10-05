import numpy as np

from elements.elemtype import get_element_type
from elements.kinematics import build_B_matrix


def assemble_tangent_stiffness(
    nodes,
    mesh_elements,
    gauss_tangents=None,
    active_element_ids=None,
    ndof_node=2,
    material_models=None,
    gauss_states=None,
):
    """
    Assemble the current global tangent stiffness matrix.

    Preferred constitutive-response path
    -----------------------------------
    ``gauss_tangents`` is supplied by the constitutive integration performed
    earlier in the same Newton residual evaluation:

        material.integrate(...)
            -> ConstitutiveResponse.tangent
            -> state_manager.trial_tangents
            -> this assembly routine

    Therefore tangent assembly does not need to know which material model is
    active at a Gauss point.

    Backward compatibility
    ----------------------
    ``material_models`` + ``gauss_states`` are still accepted as a legacy
    fallback for older GeoPyFEM scripts. New constitutive models should rely
    on the response/tangent path above.
    """
    if ndof_node != 2:
        raise NotImplementedError(
            "GeoPyFEM tangent assembly currently supports 2 DOFs per node."
        )

    use_response_tangents = (
        gauss_tangents is not None
    )

    if not use_response_tangents:
        if material_models is None or gauss_states is None:
            raise ValueError(
                "assemble_tangent_stiffness requires gauss_tangents from "
                "ConstitutiveResponse, or the legacy material_models + "
                "gauss_states pair."
            )

    num_nodes = len(nodes)

    K_tangent = np.zeros((
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
                f"{element_ndof_node} DOFs/node, while tangent assembly "
                f"is configured for {ndof_node} DOFs/node."
            )

        if len(connectivity) != nnode:
            raise ValueError(
                f'Element {element["id"]} is {element["type"]}, but its '
                f"connectivity contains {len(connectivity)} nodes "
                f"instead of {nnode}."
            )

        if use_response_tangents:
            if element_id not in gauss_tangents:
                raise ValueError(
                    f"No constitutive tangents were supplied for element "
                    f"{element_id}."
                )

            element_tangents = gauss_tangents[
                element_id
            ]

        else:
            if element_id not in gauss_states:
                raise ValueError(
                    f"No Gauss-point states were supplied for element "
                    f"{element_id}."
                )

            material_region = element[
                "material"
            ]

            if material_region not in material_models:
                raise ValueError(
                    f'Element {element_id} uses material region '
                    f'"{material_region}", but no material model is available.'
                )

            material = material_models[
                material_region
            ]

            element_states = gauss_states[
                element_id
            ]

        gauss_points, gauss_weights = (
            gauss_rule()
        )

        if use_response_tangents:
            if len(element_tangents) != len(gauss_points):
                raise ValueError(
                    f"Element {element_id} has {len(element_tangents)} stored "
                    f"constitutive tangents but its integration rule has "
                    f"{len(gauss_points)} points."
                )
        else:
            if len(element_states) != len(gauss_points):
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

        K_tangent_element = np.zeros((
            ndof_element,
            ndof_element,
        ))

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

            if use_response_tangents:
                tangent = element_tangents[
                    gp_index
                ]

                if tangent is None:
                    raise ValueError(
                        f"Element {element_id}, Gauss point {gp_index} has "
                        "no ConstitutiveResponse tangent. The constitutive "
                        "integration must run before tangent assembly."
                    )

                C_tangent = np.asarray(
                    tangent,
                    dtype=float,
                ).reshape(3, 3)

            else:
                state = element_states[
                    gp_index
                ]

                C_tangent = material.tangent_matrix(
                    state=state
                )

            K_tangent_element += (
                B.T
                @ C_tangent
                @ B
                * detJ
                * w
            )

        for i, I in enumerate(
            connectivity
        ):
            for j, Jnode in enumerate(
                connectivity
            ):
                K_tangent[
                    2*I,
                    2*Jnode,
                ] += K_tangent_element[
                    2*i,
                    2*j,
                ]

                K_tangent[
                    2*I + 1,
                    2*Jnode,
                ] += K_tangent_element[
                    2*i + 1,
                    2*j,
                ]

                K_tangent[
                    2*I + 1,
                    2*Jnode + 1,
                ] += K_tangent_element[
                    2*i + 1,
                    2*j + 1,
                ]

                K_tangent[
                    2*I,
                    2*Jnode + 1,
                ] += K_tangent_element[
                    2*i,
                    2*j + 1,
                ]

    return K_tangent
