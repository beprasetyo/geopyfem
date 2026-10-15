from time import perf_counter

import numpy as np

from elements.kinematics_cache import get_element_kinematics


def assemble_internal_force(
    nodes,
    mesh_elements,
    gauss_states,
    active_element_ids=None,
    ndof_node=2,
    kinematics_cache=None,
    profiler=None,
):
    """Assemble the global internal-force vector from TOTAL GP stresses.

    When ``kinematics_cache`` is supplied, invariant small-strain B matrices,
    Jacobian determinants, Gauss weights, and element DOF maps are reused.
    Constitutive models store effective stress; equilibrium uses total stress.
    """
    if ndof_node != 2:
        raise NotImplementedError(
            "GeoPyFEM internal-force assembly currently supports "
            "2 displacement DOFs per node."
        )

    profile_start = perf_counter() if profiler is not None else None
    if profiler is not None:
        profiler.increment("internal_force_assemblies")

    f_internal = np.zeros(ndof_node * len(nodes), dtype=float)

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

        if element_id not in gauss_states:
            raise ValueError(
                f"No Gauss-point states were supplied for element {element_id}."
            )

        element_states = gauss_states[element_id]
        if len(element_states) != kinematics.number_of_gauss_points:
            raise ValueError(
                f"Element {element_id} has {len(element_states)} stored "
                f"Gauss-point states but its integration rule has "
                f"{kinematics.number_of_gauss_points} points."
            )

        f_element = np.zeros(kinematics.number_of_dofs, dtype=float)

        for gp_index, gp in enumerate(kinematics.gauss_points):
            state = element_states[gp_index]
            sigma_total_2d = state.total_stress_2d
            f_element += (
                gp.B.T
                @ sigma_total_2d
                * gp.detJ
                * gp.weight
            )

        # Element DOFs are unique, so direct indexed accumulation is safe.
        f_internal[kinematics.dofs] += f_element

    if profiler is not None:
        profiler.add_time(
            "internal_force_assembly",
            perf_counter() - profile_start,
        )

    return f_internal
