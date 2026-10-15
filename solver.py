"""GeoPyFEM solver dispatcher and backwards-compatible facade.

Numerical implementations live in the :mod:`solvers` package.  Keeping this
module small makes the overall execution flow easy to inspect while preserving
``from solver import solve_problem`` for main.py and existing user scripts.
"""

from solvers.common import (
    _initialize_groundwater_if_present,
    _element_displacement_vector,
    _update_static_linear_gauss_states,
    _update_trial_gauss_states,
    _nonlinear_settings,
    _element_ids_for_regions,
    _active_node_ids,
    _build_activation_reference_strains,
    _get_staged_dof_sets,
    _get_fixed_and_free_dofs,
    _relative_displacement_correction,
)
from solvers.static_linear import solve_static_linear
from solvers.nonlinear import solve_static_nonlinear, _solve_nonlinear_load_path
from solvers.safety_arc_length import _solve_strength_reduction_path
from solvers.k0_stage import _solve_k0_initial_stage
from solvers.staged import solve_staged_static
from solvers.time_dependent import solve_time_dependent


def solve_problem(
    nodes,
    mesh_elements,
    physical_groups,
    physical_group_elements,
    material_models,
    problem,
):
    """Dispatch one GeoPyFEM problem to the appropriate analysis solver."""
    analysis = problem.get("analysis", {})
    solver_settings = problem.get("solver", {})
    stages = problem.get("stages", [])

    if stages:
        return solve_staged_static(
            nodes=nodes,
            mesh_elements=mesh_elements,
            physical_groups=physical_groups,
            physical_group_elements=physical_group_elements,
            material_models=material_models,
            loads=problem.get("loads", []),
            boundaries=problem.get("boundaries", []),
            stages=stages,
            groundwater=problem.get("groundwater"),
            analysis=analysis,
            units=problem.get("general", {}).get("units", {}),
            solver_settings=solver_settings,
        )

    analysis_type = str(analysis.get("type", "static")).lower()
    procedure = str(analysis.get("procedure", "linear")).lower()

    if analysis_type == "static" and procedure == "linear":
        return solve_static_linear(
            nodes=nodes,
            mesh_elements=mesh_elements,
            physical_groups=physical_groups,
            physical_group_elements=physical_group_elements,
            material_models=material_models,
            loads=problem.get("loads", []),
            boundaries=problem.get("boundaries", []),
            gravity=analysis.get("gravity", {}),
            groundwater=problem.get("groundwater"),
            units=problem.get("general", {}).get("units", {}),
            solver_settings=solver_settings,
        )

    if analysis_type == "static" and procedure in ("plastic", "nonlinear"):
        return solve_static_nonlinear(
            nodes=nodes,
            mesh_elements=mesh_elements,
            physical_groups=physical_groups,
            physical_group_elements=physical_group_elements,
            material_models=material_models,
            loads=problem.get("loads", []),
            boundaries=problem.get("boundaries", []),
            gravity=analysis.get("gravity", {}),
            groundwater=problem.get("groundwater"),
            units=problem.get("general", {}).get("units", {}),
            analysis=analysis,
            solver_settings=solver_settings,
        )

    if analysis_type == "time_dependent":
        return solve_time_dependent(
            nodes=nodes,
            mesh_elements=mesh_elements,
            physical_groups=physical_groups,
            physical_group_elements=physical_group_elements,
            material_models=material_models,
            loads=problem.get("loads", []),
            boundaries=problem.get("boundaries", []),
            analysis=analysis,
            solver_settings=solver_settings,
        )

    raise NotImplementedError(
        f'Unsupported analysis combination: '
        f'type="{analysis_type}", procedure="{procedure}".'
    )


__all__ = [
    "solve_problem",
    "solve_static_linear",
    "solve_static_nonlinear",
    "solve_staged_static",
    "solve_time_dependent",
]
