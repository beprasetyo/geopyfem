"""One-step static linear solver.

This module owns the linear mechanical solve and its equilibrium/state update.
It has no staged-construction or safety-analysis policy.
"""

from time import perf_counter

import numpy as np

from assembly.global_stiffness import assemble_global_stiffness
from assembly.load_vector import assemble_load_vector
from assembly.boundary_condition import apply_displacement_boundary_conditions
from assembly.internal_force import assemble_internal_force
from state.state_manager import GaussPointStateManager
from elements.kinematics_cache import ElementKinematicsCache
from solvers.linear_algebra import solve_direct
from solvers.profiler import SolverProfiler
from solvers.common import (
    _initialize_groundwater_if_present,
    _update_static_linear_gauss_states,
)

def solve_static_linear(
    nodes,
    mesh_elements,
    physical_groups,
    physical_group_elements,
    material_models,
    loads,
    boundaries,
    gravity=None,
    groundwater=None,
    units=None,
    solver_settings=None,
):
    """
    Solve one-step static linear equilibrium:

        K u = f

    The assembly modules remain independent from the solver:

        assembly/global_stiffness.py
        assembly/load_vector.py
        assembly/boundary_condition.py

    The solver coordinates when those modules are called.
    """
    solver_settings = solver_settings or {}
    profiler = SolverProfiler()
    solver_start = perf_counter()

    linear_solver = str(
        solver_settings.get("linear_solver", "direct")
    ).lower()

    matrix_storage = str(
        solver_settings.get("matrix_storage", "dense")
    ).strip().lower()

    if matrix_storage not in ("dense", "sparse"):
        raise ValueError(
            "MatrixStorage must be 'dense' or 'sparse'."
        )

    # ------------------------------------------------------------------
    # Gauss-point material-state infrastructure
    # ------------------------------------------------------------------
    #
    # Even the current linear analysis owns one state per integration point.
    # This makes the solver/result interface ready for K0 initial stress and
    # history-dependent nonlinear constitutive models without redesigning the
    # result structure later.
    cache_start = perf_counter()
    kinematics_cache = ElementKinematicsCache(
        nodes=nodes,
        mesh_elements=mesh_elements,
    )
    profiler.add_time(
        "kinematics_cache_build",
        perf_counter() - cache_start,
    )
    print(
        "initialize element kinematics cache:",
        f"elements={kinematics_cache.number_of_elements},",
        f"Gauss points={kinematics_cache.number_of_gauss_points}",
    )

    state_manager = GaussPointStateManager(
        mesh_elements
    )

    print(
        "initialize Gauss-point states:",
        state_manager.number_of_states,
    )

    groundwater_summary = _initialize_groundwater_if_present(
        nodes=nodes,
        mesh_elements=mesh_elements,
        state_manager=state_manager,
        groundwater=groundwater,
    )

    print("assemble global stiffness matrix")

    K = assemble_global_stiffness(
        nodes,
        mesh_elements,
        material_models,
        matrix_storage=matrix_storage,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
    )

    print("global matrix storage:", matrix_storage)

    print("assemble global load vector")

    f = assemble_load_vector(
        nodes=nodes,
        mesh_elements=mesh_elements,
        physical_group_elements=physical_group_elements,
        material_models=material_models,
        loads=loads,
        gravity=gravity,
        units=units,
    )

    # Preserve the physical external-force vector before the displacement
    # boundary-condition routine modifies constrained entries for Ku = f.
    f_external = f.copy()

    print("assign displacement boundary conditions")

    K, f, fixed_dofs = apply_displacement_boundary_conditions(
        K,
        f,
        physical_groups,
        boundaries,
    )

    print("solve static linear system")

    if linear_solver == "direct":
        u = solve_direct(K, f, profiler=profiler)

    else:
        raise NotImplementedError(
            f'Linear solver "{linear_solver}" is not implemented. '
            'Currently supported: "direct".'
        )

    # ------------------------------------------------------------------
    # Update solved linear-elastic Gauss-point state
    # ------------------------------------------------------------------
    #
    # The solved displacement field is converted to strain through the same
    # shared B matrix used by stiffness assembly. Stress is then obtained from
    # the material model and stored in the TRIAL state.
    #
    # Because this is a one-step linear analysis, the trial state is accepted
    # immediately after the global solve.
    _update_static_linear_gauss_states(
        nodes=nodes,
        mesh_elements=mesh_elements,
        u=u,
        material_models=material_models,
        state_manager=state_manager,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
    )

    state_manager.commit()

    # ------------------------------------------------------------------
    # Internal-force assembly and equilibrium verification
    # ------------------------------------------------------------------
    #
    # The linear solution was obtained from K u = f. We now reconstruct
    # equilibrium through an independent path:
    #
    #       u -> strain -> stress -> integral(B^T stress dOmega)
    #
    # This is the same internal-force mechanism required by Newton-Raphson.
    f_internal = assemble_internal_force(
        nodes=nodes,
        mesh_elements=mesh_elements,
        gauss_states=state_manager.committed,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
    )

    residual = (
        f_external
        - f_internal
    )

    all_dofs = np.arange(
        len(u),
        dtype=int,
    )

    fixed_dof_array = np.asarray(
        fixed_dofs,
        dtype=int,
    )

    free_dofs = np.setdiff1d(
        all_dofs,
        fixed_dof_array,
        assume_unique=False,
    )

    free_residual_norm = np.linalg.norm(
        residual[
            free_dofs
        ]
    )

    reaction = np.zeros_like(
        f_internal
    )

    reaction[
        fixed_dof_array
    ] = (
        f_internal[
            fixed_dof_array
        ]
        - f_external[
            fixed_dof_array
        ]
    )

    print(
        "free-DOF equilibrium residual norm:",
        f"{free_residual_norm:.6e}",
    )

    return {
        "u": u,
        "K": K,

        # Boundary-condition-modified vector used by the linear equation solve.
        "f": f,

        # Physical equilibrium vectors.
        "f_external": f_external,
        "f_internal": f_internal,
        "residual": residual,
        "reaction": reaction,
        "free_residual_norm": free_residual_norm,

        "fixed_dofs": fixed_dofs,

        # Material state at the last converged solution.
        # gauss_states is convenient for post-processing/output.
        # state_manager is retained for future trial/commit/revert workflows.
        "gauss_states": state_manager.committed,
        "state_manager": state_manager,
        "groundwater": groundwater,
        "groundwater_summary": groundwater_summary,

        "converged": True,
        "iterations": 1,
        "analysis_type": "static",
        "procedure": "linear",
        "matrix_storage": matrix_storage,
        "linear_solver": linear_solver,
        "performance_profile": profiler.snapshot(
            total_elapsed=perf_counter() - solver_start
        ),
    }
