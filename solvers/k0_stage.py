"""K0 initial-stage orchestration.

The stress-generation physics remains in :mod:`initial_stress.k0`.  This
module only verifies global equilibrium and packages the solver result.
"""

import numpy as np

from elements.kinematics_cache import ElementKinematicsCache

from assembly.internal_force import assemble_internal_force
from assembly.tangent_stiffness import assemble_tangent_stiffness
from initial_stress.k0 import initialize_k0_stress
from solvers.profiler import SolverProfiler, profile_section

def _solve_k0_initial_stage(
    nodes,
    mesh_elements,
    physical_group_elements,
    material_models,
    state_manager,
    fixed_dofs,
    fixed_dof_array,
    free_dofs,
    active_loads,
    gravity,
    k0_specification,
    f_external_target,
    units=None,
    active_element_ids=None,
    reaction_dof_array=None,
    matrix_storage="dense",
    kinematics_cache=None,
    profiler=None,
    sparse_pattern=None,
):
    """
    Generate and verify one direct K0 geostatic initial-stress stage.

    K0 is not solved by displacement loading.  Instead the initial total
    stress field is generated directly from vertical overburden, hydrostatic
    pore pressure, and the specified effective-stress K0 coefficients.

    The resulting stress field must already equilibrate the active gravity
    body force on all FREE DOFs.  Reactions may of course exist on constrained
    bottom/side DOFs.
    """
    if profiler is None:
        profiler = SolverProfiler()

    if kinematics_cache is None:
        kinematics_cache = ElementKinematicsCache(
            nodes=nodes,
            mesh_elements=mesh_elements,
        )

    active_element_ids = (
        None
        if active_element_ids is None
        else {
            int(element_id)
            for element_id in active_element_ids
        }
    )

    if reaction_dof_array is None:
        reaction_dof_array = np.asarray(
            fixed_dof_array,
            dtype=int,
        )
    else:
        reaction_dof_array = np.asarray(
            reaction_dof_array,
            dtype=int,
        )

    if active_loads:
        names = [
            load.get("name")
            for load in active_loads
        ]
        raise ValueError(
            "K0 initial stage cannot contain active boundary traction loads. "
            f"Active loads: {names}"
        )

    if not k0_specification:
        raise ValueError(
            "Stage procedure K0 requires a <K0>...</K0> definition."
        )

    with profile_section(
        profiler,
        "k0_initialization",
        count_name="k0_initializations",
    ):
        summary = initialize_k0_stress(
            nodes=nodes,
            mesh_elements=mesh_elements,
            material_models=material_models,
            state_manager=state_manager,
            k0_specification=k0_specification,
            gravity=gravity,
            units=units,
            active_element_ids=active_element_ids,
        )

    # The staged solver has already assembled the target external-force
    # configuration. For K0 this is self weight only because boundary
    # tractions are rejected above.
    f_external = np.asarray(
        f_external_target,
        dtype=float,
    ).copy()

    f_internal = assemble_internal_force(
        nodes=nodes,
        mesh_elements=mesh_elements,
        gauss_states=state_manager.committed,
        active_element_ids=active_element_ids,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
    )

    residual = f_external - f_internal

    residual_norm = np.linalg.norm(
        residual[free_dofs]
    )

    reference_norm = max(
        np.linalg.norm(
            f_external[free_dofs]
        ),
        1.0,
    )

    relative_residual = (
        residual_norm
        / reference_norm
    )

    tolerance = float(
        k0_specification.get(
            "equilibrium_tolerance",
            1.0e-8,
        )
    )

    reaction = np.zeros_like(
        f_internal
    )

    reaction[reaction_dof_array] = (
        f_internal[reaction_dof_array]
        - f_external[reaction_dof_array]
    )

    print(
        "K0 initial stress:",
        f"regions={summary['regions']},",
        f"max sigma_v={summary['maximum_vertical_total_stress']:.6g},",
        f"max sigma_v_eff={summary['maximum_vertical_effective_stress']:.6g}"
    )
    print(
        "K0 overburden spatial index:",
        f"elements={summary.get('overburden_index_elements', 0)},",
        f"candidate checks={summary.get('overburden_candidate_checks', 0)},",
        f"brute-force equivalent={summary.get('overburden_bruteforce_equivalent_checks', 0)},",
        f"reduction={summary.get('overburden_candidate_reduction_factor', 1.0):.2f}x"
    )
    print(
        "K0 free-DOF equilibrium:",
        f"|r_free|={residual_norm:.6e},",
        f"relative={relative_residual:.6e}"
    )

    if relative_residual > tolerance:
        raise RuntimeError(
            "Generated K0 initial stress is not in equilibrium on the free "
            f"DOFs: relative residual={relative_residual:.6e} > "
            f"tolerance={tolerance:.6e}. For a K0 column, check that the "
            "vertical side boundaries restrain ux and that gravity is vertical."
        )

    K_tangent = assemble_tangent_stiffness(
        nodes=nodes,
        mesh_elements=mesh_elements,
        material_models=material_models,
        gauss_states=state_manager.committed,
        active_element_ids=active_element_ids,
        matrix_storage=matrix_storage,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
        sparse_pattern=sparse_pattern,
    )

    return {
        "u": np.zeros(
            2 * len(nodes),
            dtype=float,
        ),
        "K": K_tangent,
        "f": f_external.copy(),
        "f_external": f_external,
        "f_internal": f_internal,
        "residual": residual,
        "reaction": reaction,
        "free_residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "fixed_dofs": fixed_dofs,
        "gauss_states": state_manager.committed,
        "state_manager": state_manager,
        "active_element_ids": (
            None
            if active_element_ids is None
            else sorted(
                active_element_ids
            )
        ),
        "converged": True,
        "iterations": 0,
        "load_steps": 0,
        "step_history": [],
        "analysis_type": "static",
        "procedure": "k0",
        "k0_summary": summary,
        "matrix_storage": matrix_storage,
        "performance_profile": profiler.snapshot(),
    }
