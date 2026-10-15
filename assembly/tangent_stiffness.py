from time import perf_counter

import numpy as np

from elements.kinematics_cache import get_element_kinematics
from assembly.matrix_builder import (
    GlobalMatrixBuilder,
    normalize_matrix_storage,
)


def assemble_tangent_stiffness(
    nodes,
    mesh_elements,
    gauss_tangents=None,
    active_element_ids=None,
    ndof_node=2,
    material_models=None,
    gauss_states=None,
    matrix_storage="dense",
    kinematics_cache=None,
    profiler=None,
    sparse_pattern=None,
):
    """Assemble the current global tangent stiffness matrix.

    Preferred constitutive-response path
    -----------------------------------
    ``gauss_tangents`` is supplied by the constitutive integration performed
    earlier in the SAME Newton residual evaluation.  Tangent assembly is then
    completely material-model agnostic.

    ``kinematics_cache`` optionally reuses invariant small-strain element data
    (DOF map, B, detJ, and Gauss weights) across Newton iterations.
    """
    if ndof_node != 2:
        raise NotImplementedError(
            "GeoPyFEM tangent assembly currently supports 2 DOFs per node."
        )

    profile_start = perf_counter() if profiler is not None else None
    if profiler is not None:
        profiler.increment("tangent_assemblies")

    matrix_storage = normalize_matrix_storage(matrix_storage)
    use_response_tangents = gauss_tangents is not None

    if not use_response_tangents:
        if material_models is None or gauss_states is None:
            raise ValueError(
                "assemble_tangent_stiffness requires gauss_tangents from "
                "ConstitutiveResponse, or the legacy material_models + "
                "gauss_states pair."
            )

    num_nodes = len(nodes)
    ndof_total = ndof_node * num_nodes
    builder = GlobalMatrixBuilder(
        size=ndof_total,
        matrix_storage=matrix_storage,
        profiler=profiler,
        sparse_pattern=sparse_pattern,
    )

    active_element_ids = (
        None
        if active_element_ids is None
        else {int(element_id) for element_id in active_element_ids}
    )

    for element in mesh_elements:
        element_id = int(element["id"])
        if active_element_ids is not None and element_id not in active_element_ids:
            continue

        kinematics = get_element_kinematics(
            nodes=nodes,
            element=element,
            ndof_node=ndof_node,
            kinematics_cache=kinematics_cache,
        )
        if kinematics is None:
            continue

        if use_response_tangents:
            if element_id not in gauss_tangents:
                raise ValueError(
                    f"No constitutive tangents were supplied for element "
                    f"{element_id}."
                )
            element_tangents = gauss_tangents[element_id]
            if len(element_tangents) != kinematics.number_of_gauss_points:
                raise ValueError(
                    f"Element {element_id} has {len(element_tangents)} stored "
                    f"constitutive tangents but its integration rule has "
                    f"{kinematics.number_of_gauss_points} points."
                )
        else:
            if element_id not in gauss_states:
                raise ValueError(
                    f"No Gauss-point states were supplied for element "
                    f"{element_id}."
                )

            material_region = element["material"]
            if material_region not in material_models:
                raise ValueError(
                    f'Element {element_id} uses material region '
                    f'"{material_region}", but no material model is available.'
                )

            material = material_models[material_region]
            element_states = gauss_states[element_id]
            if len(element_states) != kinematics.number_of_gauss_points:
                raise ValueError(
                    f"Element {element_id} has {len(element_states)} stored "
                    f"Gauss-point states but its integration rule has "
                    f"{kinematics.number_of_gauss_points} points."
                )

        Ke = np.zeros(
            (kinematics.number_of_dofs, kinematics.number_of_dofs),
            dtype=float,
        )

        element_integration_start = (
            perf_counter() if profiler is not None else None
        )

        for gp_index, gp in enumerate(kinematics.gauss_points):
            if use_response_tangents:
                tangent = element_tangents[gp_index]
                if tangent is None:
                    raise ValueError(
                        f"Element {element_id}, Gauss point {gp_index} has "
                        "no ConstitutiveResponse tangent. Constitutive "
                        "integration must run before tangent assembly."
                    )
                C_tangent = np.asarray(tangent, dtype=float).reshape(3, 3)
            else:
                C_tangent = np.asarray(
                    material.tangent_matrix(state=element_states[gp_index]),
                    dtype=float,
                ).reshape(3, 3)

            Ke += gp.B.T @ C_tangent @ gp.B * gp.detJ * gp.weight

        if profiler is not None:
            profiler.add_time(
                "tangent_element_integration",
                perf_counter() - element_integration_start,
            )

        builder.add(kinematics.dofs, Ke, element_id=element_id)

    matrix = builder.finalize()

    if profiler is not None:
        profiler.add_time(
            "tangent_assembly",
            perf_counter() - profile_start,
        )

    return matrix
