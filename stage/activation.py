import numpy as np

from elements.elemtype import get_element_type
from elements.kinematics import build_B_matrix


def _element_node_ids(
    mesh_elements,
    element_ids,
):
    """Return nodes belonging to the selected 2-D domain elements."""
    element_ids = {
        int(element_id)
        for element_id in element_ids
    }

    node_ids = set()

    for element in mesh_elements:
        element_id = int(
            element["id"]
        )

        if element_id not in element_ids:
            continue

        element_info = get_element_type(
            element["type"]
        )

        if element_info["dimension"] != 2:
            continue

        node_ids.update(
            int(node_id)
            for node_id in element["connectivity"]
        )

    return node_ids


def _assemble_region_laplacian(
    nodes,
    mesh_elements,
    element_ids,
):
    """
    Assemble a scalar FE Laplacian matrix over newly activated elements.

    This is a purely kinematic auxiliary problem.  It is NOT a mechanical
    stiffness matrix and does not use material E, nu, density, or stress.

    The matrix is used to smoothly extend the already accumulated nodal
    displacement from the old active interface into nodes that did not exist
    mechanically before activation.
    """
    element_ids = {
        int(element_id)
        for element_id in element_ids
    }

    region_node_ids = sorted(
        _element_node_ids(
            mesh_elements=mesh_elements,
            element_ids=element_ids,
        )
    )

    if not region_node_ids:
        return [], np.zeros((0, 0), dtype=float)

    local_index = {
        node_id: local_id
        for local_id, node_id in enumerate(region_node_ids)
    }

    K_extension = np.zeros(
        (
            len(region_node_ids),
            len(region_node_ids),
        ),
        dtype=float,
    )

    found_element_ids = set()

    for element in mesh_elements:
        element_id = int(
            element["id"]
        )

        if element_id not in element_ids:
            continue

        element_info = get_element_type(
            element["type"]
        )

        if element_info["dimension"] != 2:
            continue

        found_element_ids.add(
            element_id
        )

        connectivity = [
            int(node_id)
            for node_id in element["connectivity"]
        ]

        coordinates = nodes[
            connectivity,
            :,
        ]

        gradshape = element_info[
            "gradshape"
        ]

        gauss_points, weights = element_info[
            "gauss"
        ]()

        Ke = np.zeros(
            (
                len(connectivity),
                len(connectivity),
            ),
            dtype=float,
        )

        for q, weight in zip(
            gauss_points,
            weights,
        ):
            _, detJ, dN_global = build_B_matrix(
                element_coordinates=coordinates,
                gradshape=gradshape,
                natural_coordinates=q,
            )

            # Scalar Laplacian / harmonic-extension operator:
            #
            #     Ke = integral(grad(N)^T grad(N) dOmega)
            #
            # It is used only to interpolate a smooth birth displacement.
            Ke += (
                dN_global.T
                @ dN_global
                * detJ
                * weight
            )

        local_connectivity = [
            local_index[node_id]
            for node_id in connectivity
        ]

        for a, A in enumerate(local_connectivity):
            for b, B in enumerate(local_connectivity):
                K_extension[A, B] += Ke[a, b]

    missing = (
        element_ids
        - found_element_ids
    )

    if missing:
        raise ValueError(
            "Could not assemble activation displacement extension for "
            f"element IDs {sorted(missing)}."
        )

    return region_node_ids, K_extension


def extend_activation_birth_fields(
    nodes,
    mesh_elements,
    activated_element_ids,
    previously_active_node_ids,
    displacement,
    displacement_reference=None,
):
    """
    Give newly activated nodes a cumulative geometric birth displacement.

    Why this is needed
    ------------------
    Before element activation, nodes belonging only to inactive elements are
    suppressed from the mechanical system.  Their displacement therefore does
    not automatically follow settlement/deformation that has already occurred
    in the existing active domain.

    At activation we want BOTH:

    1. geometric continuity / cumulative displacement, and
    2. stress-free birth of the newly installed material.

    This routine handles item (1).  It solves a harmonic extension problem on
    the newly activated region.  Nodes shared with the OLD active domain are
    Dirichlet anchors carrying their already accumulated displacement.  The
    displacement is smoothly extended from those anchors to the previously
    inactive nodes.

    The constitutive activation strain reference must be built AFTER this
    routine, using the updated displacement field:

        epsilon_ref = B @ u_birth

    so the new element still starts with zero material strain/stress.

    Parameters
    ----------
    nodes : ndarray (nnode, 2)
        Mesh coordinates.
    mesh_elements : list[dict]
        GeoPyFEM domain element dictionaries.
    activated_element_ids : iterable[int]
        Elements born in the current stage.
    previously_active_node_ids : iterable[int]
        Nodes connected to the mechanical domain immediately BEFORE birth.
    displacement : ndarray (2*nnode,)
        Internal accumulated displacement vector.
    displacement_reference : ndarray (2*nnode,), optional
        Reporting reset reference.  It is extended by the same operator so
        <ResetDisplacements> remains consistent for nodes born later.

    Returns
    -------
    result : dict
        Updated displacement/reference and activation diagnostics.
    """
    activated_element_ids = {
        int(element_id)
        for element_id in activated_element_ids
    }

    previously_active_node_ids = {
        int(node_id)
        for node_id in previously_active_node_ids
    }

    u_updated = np.asarray(
        displacement,
        dtype=float,
    ).copy()

    if displacement_reference is None:
        reference_updated = np.zeros_like(
            u_updated
        )
    else:
        reference_updated = np.asarray(
            displacement_reference,
            dtype=float,
        ).copy()

    if not activated_element_ids:
        return {
            "displacement": u_updated,
            "displacement_reference": reference_updated,
            "region_node_ids": [],
            "anchor_node_ids": [],
            "birth_node_ids": [],
            "maximum_birth_displacement": 0.0,
        }

    (
        region_node_ids,
        K_extension,
    ) = _assemble_region_laplacian(
        nodes=nodes,
        mesh_elements=mesh_elements,
        element_ids=activated_element_ids,
    )

    region_node_set = set(
        region_node_ids
    )

    anchor_node_ids = sorted(
        region_node_set
        & previously_active_node_ids
    )

    birth_node_ids = sorted(
        region_node_set
        - previously_active_node_ids
    )

    # If every activated-region node already belongs to the active mesh,
    # there is no new nodal displacement to initialize.  The activation
    # strain reference will still make the new elements stress-free.
    if not birth_node_ids:
        return {
            "displacement": u_updated,
            "displacement_reference": reference_updated,
            "region_node_ids": region_node_ids,
            "anchor_node_ids": anchor_node_ids,
            "birth_node_ids": [],
            "maximum_birth_displacement": 0.0,
        }

    if not anchor_node_ids:
        raise ValueError(
            "Newly activated elements contain previously inactive nodes but "
            "do not share any node with the existing active domain. "
            "GeoPyFEM cannot infer their cumulative placement displacement. "
            "Activate a region connected to the existing mesh, or define a "
            "future explicit placement rule for detached construction."
        )

    local_index = {
        node_id: local_id
        for local_id, node_id in enumerate(region_node_ids)
    }

    anchor_local = np.asarray(
        [
            local_index[node_id]
            for node_id in anchor_node_ids
        ],
        dtype=int,
    )

    birth_local = np.asarray(
        [
            local_index[node_id]
            for node_id in birth_node_ids
        ],
        dtype=int,
    )

    K_ff = K_extension[
        np.ix_(
            birth_local,
            birth_local,
        )
    ]

    K_fc = K_extension[
        np.ix_(
            birth_local,
            anchor_local,
        )
    ]

    # Solve displacement and reporting-reference extension together as four
    # right-hand sides: ux, uy, ux_reference, uy_reference.
    anchor_values = np.zeros(
        (
            len(anchor_node_ids),
            4,
        ),
        dtype=float,
    )

    for row, node_id in enumerate(anchor_node_ids):
        anchor_values[row, 0] = u_updated[
            2 * node_id
        ]
        anchor_values[row, 1] = u_updated[
            2 * node_id + 1
        ]
        anchor_values[row, 2] = reference_updated[
            2 * node_id
        ]
        anchor_values[row, 3] = reference_updated[
            2 * node_id + 1
        ]

    rhs = -(
        K_fc
        @ anchor_values
    )

    try:
        birth_values = np.linalg.solve(
            K_ff,
            rhs,
        )

    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "Activation displacement extension is singular. A newly "
            "activated connected component is probably not anchored to the "
            "previously active domain."
        ) from error

    for row, node_id in enumerate(birth_node_ids):
        u_updated[
            2 * node_id
        ] = birth_values[row, 0]
        u_updated[
            2 * node_id + 1
        ] = birth_values[row, 1]

        reference_updated[
            2 * node_id
        ] = birth_values[row, 2]
        reference_updated[
            2 * node_id + 1
        ] = birth_values[row, 3]

    birth_displacement_magnitude = np.sqrt(
        birth_values[:, 0] ** 2
        + birth_values[:, 1] ** 2
    )

    maximum_birth_displacement = (
        float(
            np.max(
                birth_displacement_magnitude
            )
        )
        if birth_displacement_magnitude.size
        else 0.0
    )

    return {
        "displacement": u_updated,
        "displacement_reference": reference_updated,
        "region_node_ids": region_node_ids,
        "anchor_node_ids": anchor_node_ids,
        "birth_node_ids": birth_node_ids,
        "maximum_birth_displacement": maximum_birth_displacement,
    }
