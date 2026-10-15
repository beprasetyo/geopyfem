from time import perf_counter

import numpy as np

from elements.kinematics_cache import get_element_kinematics
from assembly.matrix_builder import (
    GlobalMatrixBuilder,
    normalize_matrix_storage,
)


def assemble_global_stiffness(
    nodes,
    mesh_elements,
    material_models,
    active_element_ids=None,
    ndof_node=2,
    matrix_storage="dense",
    kinematics_cache=None,
    profiler=None,
    sparse_pattern=None,
):
    """Assemble the linear global stiffness matrix.

    ``kinematics_cache`` is optional for backward compatibility.  When a
    shared cache is supplied, element DOF maps, B matrices, det(J), and Gauss
    weights are reused rather than recomputed.
    """
    profile_start = perf_counter() if profiler is not None else None
    if profiler is not None:
        profiler.increment("global_stiffness_assemblies")

    matrix_storage = normalize_matrix_storage(matrix_storage)
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

        material_region = element["material"]
        if material_region not in material_models:
            raise ValueError(
                f'Element {element_id} uses material region '
                f'"{material_region}", but no material model is assigned '
                "to that region."
            )

        material_model = material_models[material_region]
        C = np.asarray(material_model.tangent_matrix(), dtype=float).reshape(3, 3)

        Ke = np.zeros(
            (kinematics.number_of_dofs, kinematics.number_of_dofs),
            dtype=float,
        )

        for gp in kinematics.gauss_points:
            Ke += gp.B.T @ C @ gp.B * gp.detJ * gp.weight

        builder.add(kinematics.dofs, Ke, element_id=element_id)

    matrix = builder.finalize()

    if profiler is not None:
        profiler.add_time(
            "global_stiffness_assembly",
            perf_counter() - profile_start,
        )

    return matrix
