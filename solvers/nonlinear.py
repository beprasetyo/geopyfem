"""Static nonlinear Newton-Raphson solution procedures.

The load-path solver is intentionally independent of activation/excavation
policy.  Staged construction supplies only the current domain, start/target
external-force vectors, and the committed material state.
"""

from time import perf_counter

import numpy as np

from assembly.load_vector import assemble_load_vector
from assembly.internal_force import assemble_internal_force
from assembly.tangent_stiffness import assemble_tangent_stiffness
from assembly.matrix_builder import SparseAssemblyPattern, normalize_matrix_storage
from assembly.boundary_condition import apply_zero_displacement_constraints
from state.state_manager import GaussPointStateManager
from elements.kinematics_cache import ElementKinematicsCache
from solvers.linear_algebra import solve_direct
from solvers.profiler import SolverProfiler
from solvers.common import (
    _initialize_groundwater_if_present,
    _update_trial_gauss_states,
    _nonlinear_settings,
    _get_fixed_and_free_dofs,
    _relative_displacement_correction,
)

def _solve_nonlinear_load_path(
    nodes,
    mesh_elements,
    material_models,
    state_manager,
    fixed_dofs,
    fixed_dof_array,
    free_dofs,
    u_start,
    f_external_start,
    f_external_target,
    load_steps,
    solver_settings,
    stage_id=None,
    stage_name=None,
    active_element_ids=None,
    reaction_dof_array=None,
    load_path_kind="load_change",
    kinematics_cache=None,
    profiler=None,
    sparse_pattern=None,
):
    """
    Advance one nonlinear load path with full Newton-Raphson.

    Equilibrium
    -----------

        r(u) = f_external - f_internal(u) = 0

    Dual convergence
    ----------------

    An increment is accepted only when BOTH conditions hold:

        relative force residual <= ForceTolerance

    AND

        relative Newton displacement correction
            <= DisplacementTolerance

    Adaptive stepping
    -----------------

    In adaptive mode the load factor is no longer forced to follow a fixed
    sequence.  GeoPyFEM advances from the last accepted factor ``lambda`` by
    a trial increment ``delta_lambda``.

    Easy convergence:
        grow the next increment.

    Difficult but converged:
        reduce the next increment.

    Failed Newton solve:
        revert trial state, cut back the increment, and retry from the last
        committed solution.

    The target of the stage remains exactly lambda = 1.0.

    ``load_path_kind`` is bookkeeping only.  The same Newton/load-factor
    engine is used for normal load addition and excavation holding-force
    release.
    """
    if profiler is None:
        profiler = SolverProfiler()

    if kinematics_cache is None:
        kinematics_cache = ElementKinematicsCache(
            nodes=nodes,
            mesh_elements=mesh_elements,
        )

    settings = _nonlinear_settings(
        solver_settings=solver_settings,
        load_steps=load_steps,
    )

    max_iterations = settings[
        "max_iterations"
    ]

    force_tolerance = settings[
        "force_tolerance"
    ]

    displacement_tolerance = settings[
        "displacement_tolerance"
    ]

    matrix_storage = settings[
        "matrix_storage"
    ]

    stepping_type = settings[
        "load_stepping_type"
    ]

    minimum_increment = settings[
        "minimum_increment"
    ]

    maximum_increment = settings[
        "maximum_increment"
    ]

    cutback_factor = settings[
        "cutback_factor"
    ]

    maximum_cutbacks = settings[
        "maximum_cutbacks"
    ]

    growth_factor = settings[
        "growth_factor"
    ]

    fast_iterations = settings[
        "fast_convergence_iterations"
    ]

    slow_iterations = settings[
        "slow_convergence_iterations"
    ]

    current_increment = settings[
        "initial_increment"
    ]

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

    u = np.asarray(
        u_start,
        dtype=float,
    ).copy()

    f_external_start = np.asarray(
        f_external_start,
        dtype=float,
    ).copy()

    f_external_target = np.asarray(
        f_external_target,
        dtype=float,
    ).copy()

    load_increment_total = (
        f_external_target
        - f_external_start
    )

    step_history = []
    attempt_history = []

    # Bookkeeping:
    #
    # residual evaluation = evaluate current constitutive state, f_int,
    #                       and r = f_ext - f_int.
    #
    # Newton correction   = one successful solve K_t * delta_u = r.
    #
    # These are not the same event, so GeoPyFEM reports them separately.
    total_newton_corrections = 0
    total_residual_evaluations = 0

    cutback_count = 0
    consecutive_cutbacks = 0
    accepted_steps = 0

    current_load_factor = 0.0

    final_f_external = (
        f_external_start.copy()
    )

    final_f_internal = assemble_internal_force(
        nodes=nodes,
        mesh_elements=mesh_elements,
        gauss_states=state_manager.committed,
        active_element_ids=active_element_ids,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
    )

    final_residual = (
        final_f_external
        - final_f_internal
    )

    label = (
        "single phase"
        if stage_id is None
        else (
            f'stage {stage_id} '
            f'"{stage_name}"'
        )
    )

    print(
        f"solve nonlinear load path: {label}"
    )

    print(
        "  path kind:",
        load_path_kind,
    )

    print(
        "  global matrix storage:",
        matrix_storage,
    )

    print(
        "  convergence:",
        f"force <= {force_tolerance:.3e},",
        f"displacement <= {displacement_tolerance:.3e}"
    )

    print(
        "  load stepping:",
        stepping_type,
        f"(initial={current_increment:.6g}, "
        f"min={minimum_increment:.6g}, "
        f"max={maximum_increment:.6g}, "
        f"max_cutbacks={maximum_cutbacks})"
    )

    factor_tolerance = 1.0e-14

    # ==================================================================
    # ACCEPTED-INCREMENT LOOP
    # ==================================================================
    while (
        current_load_factor
        < 1.0 - factor_tolerance
    ):
        remaining_factor = (
            1.0
            - current_load_factor
        )

        # The final increment is clipped so the stage ends exactly at 1.0.
        attempted_increment = min(
            current_increment,
            remaining_factor,
        )

        target_load_factor = (
            current_load_factor
            + attempted_increment
        )

        # Guard against tiny floating-point overshoot.
        if (
            target_load_factor
            > 1.0
            - factor_tolerance
        ):
            target_load_factor = 1.0
            attempted_increment = (
                target_load_factor
                - current_load_factor
            )

        f_external = (
            f_external_start
            + target_load_factor
            * load_increment_total
        )

        # Every retry starts from the LAST ACCEPTED displacement.
        u_iter = u.copy()

        attempt_number = (
            len(attempt_history)
            + 1
        )

        print(
            f"  attempt {attempt_number}: "
            f"factor {current_load_factor:.6f} "
            f"-> {target_load_factor:.6f} "
            f"(dLambda={attempted_increment:.6f})"
        )

        converged = False
        failure_reason = None

        attempt_newton_corrections = 0
        attempt_residual_evaluations = 0

        # ==============================================================
        # NEWTON-RAPHSON LOOP
        # ==============================================================
        for residual_evaluation in range(
            1,
            max_iterations + 1,
        ):
            attempt_residual_evaluations += 1
            total_residual_evaluations += 1

            # Constitutive trial states always restart from the last
            # COMMITTED converged increment.
            state_manager.reset_trial()

            _update_trial_gauss_states(
                nodes=nodes,
                mesh_elements=mesh_elements,
                u=u_iter,
                material_models=material_models,
                state_manager=state_manager,
                active_element_ids=active_element_ids,
                stage_id=stage_id,
                stage_name=stage_name,
                load_factor=target_load_factor,
                increment_number=accepted_steps + 1,
                iteration_number=residual_evaluation,
                load_path_kind=load_path_kind,
                kinematics_cache=kinematics_cache,
                profiler=profiler,
            )

            f_internal = assemble_internal_force(
                nodes=nodes,
                mesh_elements=mesh_elements,
                gauss_states=state_manager.trial,
                active_element_ids=active_element_ids,
                kinematics_cache=kinematics_cache,
                profiler=profiler,
            )

            residual = (
                f_external
                - f_internal
            )

            residual_norm = np.linalg.norm(
                residual[
                    free_dofs
                ]
            )

            force_reference_norm = max(
                np.linalg.norm(
                    f_external[
                        free_dofs
                    ]
                ),
                np.linalg.norm(
                    load_increment_total[
                        free_dofs
                    ]
                ),
                1.0,
            )

            relative_force_residual = (
                residual_norm
                / force_reference_norm
            )

            force_ok = (
                relative_force_residual
                <= force_tolerance
            )

            # If the current point is already in equilibrium before any
            # Newton correction is needed, both criteria are satisfied.
            if (
                residual_evaluation == 1
                and force_ok
            ):
                displacement_correction_norm = 0.0
                relative_displacement_correction = 0.0
                displacement_ok = True

                print(
                    f"    residual evaluation {residual_evaluation}: "
                    f"R_force={relative_force_residual:.6e}, "
                    "R_disp=0.000000e+00 "
                    "(no correction required)"
                )

            else:
                # Tangent is required not only to update displacement, but
                # also to measure the Newton correction used by the second
                # convergence criterion.
                K_tangent = assemble_tangent_stiffness(
                    nodes=nodes,
                    mesh_elements=mesh_elements,
                    gauss_tangents=state_manager.trial_tangents,
                    active_element_ids=active_element_ids,
                    matrix_storage=matrix_storage,
                    kinematics_cache=kinematics_cache,
                    profiler=profiler,
                    sparse_pattern=sparse_pattern,
                )

                K_correction, r_correction = (
                    apply_zero_displacement_constraints(
                        K=K_tangent,
                        f=residual,
                        fixed_dofs=fixed_dofs,
                    )
                )

                try:
                    delta_u = solve_direct(
                        K_correction,
                        r_correction,
                        profiler=profiler,
                    )

                    attempt_newton_corrections += 1
                    total_newton_corrections += 1

                except np.linalg.LinAlgError as error:
                    failure_reason = (
                        "linear solve failed: "
                        f"{error}"
                    )

                    print(
                        f"    residual evaluation {residual_evaluation}: "
                        f"R_force={relative_force_residual:.6e}, "
                        "linear solve failed"
                    )

                    break

                (
                    displacement_correction_norm,
                    relative_displacement_correction,
                ) = _relative_displacement_correction(
                    delta_u=delta_u,
                    u_current=u_iter,
                    free_dofs=free_dofs,
                )

                displacement_ok = (
                    relative_displacement_correction
                    <= displacement_tolerance
                )

                print(
                    f"    residual evaluation {residual_evaluation}: "
                    f"R_force={relative_force_residual:.6e}, "
                    f"R_disp="
                    f"{relative_displacement_correction:.6e}"
                )

            # ----------------------------------------------------------
            # DUAL CONVERGENCE CHECK
            # ----------------------------------------------------------
            if (
                force_ok
                and displacement_ok
            ):
                # The current trial state corresponds to u_iter.  When the
                # correction criterion is satisfied, the remaining Newton
                # correction is negligible and is intentionally not applied.
                state_manager.commit()

                u = u_iter.copy()

                final_f_external = (
                    f_external.copy()
                )

                final_f_internal = (
                    f_internal.copy()
                )

                final_residual = (
                    residual.copy()
                )

                accepted_steps += 1

                step_record = {
                    "step": accepted_steps,
                    "attempt": attempt_number,
                    "load_factor": (
                        target_load_factor
                    ),
                    "load_increment": (
                        attempted_increment
                    ),
                    # Backward-compatible "iterations" now means actual
                    # Newton correction solves.
                    "iterations": attempt_newton_corrections,
                    "newton_iterations": attempt_newton_corrections,
                    "residual_evaluations": attempt_residual_evaluations,
                    "residual_norm": (
                        residual_norm
                    ),
                    "relative_force_residual": (
                        relative_force_residual
                    ),
                    "displacement_correction_norm": (
                        displacement_correction_norm
                    ),
                    "relative_displacement_correction": (
                        relative_displacement_correction
                    ),
                    # Legacy-compatible field name.
                    "relative_residual": (
                        relative_force_residual
                    ),
                    "converged": True,
                }

                step_history.append(
                    step_record
                )

                attempt_history.append({
                    **step_record,
                    "accepted": True,
                    "failure_reason": None,
                })

                converged = True

                print(
                    "    accepted: "
                    f"{attempt_newton_corrections} Newton correction solve(s), "
                    f"{attempt_residual_evaluations} residual evaluation(s)"
                )

                break

            # Not yet converged: apply the calculated Newton correction.
            u_iter += delta_u

        # ==============================================================
        # ACCEPTED INCREMENT
        # ==============================================================
        if converged:
            current_load_factor = (
                target_load_factor
            )

            consecutive_cutbacks = 0

            if (
                stepping_type
                == "adaptive"
                and current_load_factor
                < 1.0 - factor_tolerance
            ):
                if (
                    attempt_newton_corrections
                    <= fast_iterations
                ):
                    new_increment = min(
                        attempted_increment
                        * growth_factor,
                        maximum_increment,
                    )

                    if (
                        new_increment
                        > attempted_increment
                        + factor_tolerance
                    ):
                        print(
                            "    easy convergence: "
                            f"grow next dLambda "
                            f"{attempted_increment:.6f} "
                            f"-> {new_increment:.6f}"
                        )

                    current_increment = (
                        new_increment
                    )

                elif (
                    attempt_newton_corrections
                    >= slow_iterations
                ):
                    new_increment = max(
                        attempted_increment
                        * cutback_factor,
                        minimum_increment,
                    )

                    if (
                        new_increment
                        < attempted_increment
                        - factor_tolerance
                    ):
                        print(
                            "    slow convergence: "
                            f"reduce next dLambda "
                            f"{attempted_increment:.6f} "
                            f"-> {new_increment:.6f}"
                        )

                    current_increment = (
                        new_increment
                    )

                else:
                    current_increment = (
                        attempted_increment
                    )

            else:
                current_increment = (
                    attempted_increment
                )

            continue

        # ==============================================================
        # FAILED INCREMENT -> REVERT + CUTBACK
        # ==============================================================
        state_manager.revert()

        if failure_reason is None:
            failure_reason = (
                f"maximum nonlinear residual evaluations "
                f"({max_iterations}) reached"
            )

        attempt_history.append({
            "attempt": attempt_number,
            "load_factor_start": (
                current_load_factor
            ),
            "load_factor_target": (
                target_load_factor
            ),
            "load_increment": (
                attempted_increment
            ),
            "iterations": attempt_newton_corrections,
            "newton_iterations": attempt_newton_corrections,
            "residual_evaluations": attempt_residual_evaluations,
            "accepted": False,
            "failure_reason": (
                failure_reason
            ),
        })

        if stepping_type != "adaptive":
            raise RuntimeError(
                f"Newton-Raphson did not converge in {label}, "
                f"target factor={target_load_factor:.6f}: "
                f"{failure_reason}."
            )

        if (
            consecutive_cutbacks
            >= maximum_cutbacks
        ):
            raise RuntimeError(
                f"Adaptive load stepping failed in {label}. "
                f"MaximumCutbacks={maximum_cutbacks} reached. "
                f"Last converged factor={current_load_factor:.6f}, "
                f"failed target factor={target_load_factor:.6f}, "
                f"failed dLambda={attempted_increment:.6g}. "
                f"Last failure: {failure_reason}."
            )

        cutback_count += 1
        consecutive_cutbacks += 1

        reduced_increment = (
            attempted_increment
            * cutback_factor
        )

        # Minimum increment applies to normal adaptive advancement.  A final
        # clipped increment smaller than the nominal minimum is allowed only
        # when it is simply the remainder required to reach lambda = 1.
        nominal_minimum = min(
            minimum_increment,
            remaining_factor,
        )

        if (
            reduced_increment
            < nominal_minimum
            - factor_tolerance
        ):
            raise RuntimeError(
                f"Adaptive load stepping failed in {label}. "
                f"Increment {attempted_increment:.6g} failed and "
                f"cutback would give {reduced_increment:.6g}, below "
                f"the permitted minimum {nominal_minimum:.6g}. "
                f"Last failure: {failure_reason}."
            )

        current_increment = max(
            reduced_increment,
            nominal_minimum,
        )

        print(
            "    increment rejected:",
            failure_reason,
        )

        print(
            "    rollback to factor",
            f"{current_load_factor:.6f};",
            "cut back dLambda to",
            f"{current_increment:.6f}",
            f"(consecutive cutback "
            f"{consecutive_cutbacks}/{maximum_cutbacks})"
        )

    # ==================================================================
    # FINAL EQUILIBRIUM / REACTIONS
    # ==================================================================
    reaction = np.zeros_like(
        final_f_internal
    )

    reaction[
        reaction_dof_array
    ] = (
        final_f_internal[
            reaction_dof_array
        ]
        - final_f_external[
            reaction_dof_array
        ]
    )

    final_free_residual_norm = np.linalg.norm(
        final_residual[
            free_dofs
        ]
    )

    final_K_tangent = assemble_tangent_stiffness(
        nodes=nodes,
        mesh_elements=mesh_elements,
        gauss_tangents=state_manager.committed_tangents,
        active_element_ids=active_element_ids,
        matrix_storage=matrix_storage,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
        sparse_pattern=sparse_pattern,
    )

    return {
        "u": u,
        "K": final_K_tangent,
        "f": final_f_external,
        "f_external": final_f_external,
        "f_internal": final_f_internal,
        "residual": final_residual,
        "reaction": reaction,
        "free_residual_norm": (
            final_free_residual_norm
        ),
        "fixed_dofs": fixed_dofs,
        "gauss_states": (
            state_manager.committed
        ),
        "state_manager": state_manager,
        "active_element_ids": (
            None
            if active_element_ids is None
            else sorted(
                active_element_ids
            )
        ),
        "converged": True,

        # Backward-compatible alias:
        # iterations = number of Newton correction solves.
        "iterations": total_newton_corrections,
        "newton_iterations": total_newton_corrections,
        "residual_evaluations": total_residual_evaluations,

        # ``load_steps`` now means the ACTUAL number of accepted increments.
        # For fixed stepping it remains equal to the requested LoadSteps.
        "load_steps": accepted_steps,
        "requested_load_steps": int(
            load_steps
        ),
        "cutbacks": cutback_count,
        "maximum_cutbacks": maximum_cutbacks,
        "load_stepping_type": stepping_type,
        "load_path_kind": load_path_kind,
        "matrix_storage": matrix_storage,
        "linear_solver": settings["linear_solver"],
        "final_load_factor": (
            current_load_factor
        ),
        "step_history": step_history,
        "attempt_history": attempt_history,

        "force_tolerance": force_tolerance,
        "displacement_tolerance": (
            displacement_tolerance
        ),
        "performance_profile": profiler.snapshot(),

        "analysis_type": "static",
        "procedure": "nonlinear",
    }


def solve_static_nonlinear(
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
    analysis=None,
    solver_settings=None,
):
    """Backward-compatible one-phase static nonlinear analysis."""
    analysis = analysis or {}
    load_steps_value = analysis.get("load_steps", 1)
    load_steps = 1 if load_steps_value is None else int(load_steps_value)

    profiler = SolverProfiler()
    solver_start = perf_counter()

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

    matrix_storage = normalize_matrix_storage(
        (solver_settings or {}).get("matrix_storage", "dense")
    )
    sparse_pattern = None
    if matrix_storage == "sparse":
        pattern_start = perf_counter()
        sparse_pattern = SparseAssemblyPattern.from_kinematics_cache(
            size=2 * len(nodes),
            kinematics_cache=kinematics_cache,
        )
        profiler.add_time(
            "sparse_pattern_build",
            perf_counter() - pattern_start,
        )
        profiler.increment("sparse_pattern_builds")
        print(
            "initialize fixed sparse CSR assembly pattern:",
            f"ndof={sparse_pattern.size},",
            f"nnz={sparse_pattern.nnz},",
            f"element maps={sparse_pattern.number_of_element_maps}",
        )

    state_manager = GaussPointStateManager(mesh_elements)
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

    print("assemble total external load vector")
    f_external_target = assemble_load_vector(
        nodes=nodes,
        mesh_elements=mesh_elements,
        physical_group_elements=physical_group_elements,
        material_models=material_models,
        loads=loads,
        gravity=gravity,
        units=units,
    )

    fixed_dofs, fixed_dof_array, free_dofs = _get_fixed_and_free_dofs(
        nodes=nodes,
        physical_groups=physical_groups,
        boundaries=boundaries,
        verbose=True,
    )

    result = _solve_nonlinear_load_path(
        nodes=nodes,
        mesh_elements=mesh_elements,
        material_models=material_models,
        state_manager=state_manager,
        fixed_dofs=fixed_dofs,
        fixed_dof_array=fixed_dof_array,
        free_dofs=free_dofs,
        u_start=np.zeros(2 * len(nodes)),
        f_external_start=np.zeros(2 * len(nodes)),
        f_external_target=f_external_target,
        load_steps=load_steps,
        solver_settings=solver_settings,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
        sparse_pattern=sparse_pattern,
    )

    result["performance_profile"] = profiler.snapshot(
        total_elapsed=perf_counter() - solver_start
    )
    result["sparse_pattern_build_time"] = result["performance_profile"]["timings"].get(
        "sparse_pattern_build", 0.0
    )
    result["sparse_pattern_build_count"] = 0 if sparse_pattern is None else 1
    result["sparse_pattern_nnz"] = (
        None if sparse_pattern is None else sparse_pattern.nnz
    )
    result["sparse_pattern_element_maps"] = (
        0 if sparse_pattern is None else sparse_pattern.number_of_element_maps
    )
    result["groundwater"] = groundwater
    result["groundwater_summary"] = groundwater_summary

    return result
