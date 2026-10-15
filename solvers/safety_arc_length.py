"""Arc-length continuation solver for phi-c strength reduction.

Safety-analysis policy (strength reduction, arc metric, and plateau criteria)
lives in :mod:`stage.safety`; this module owns the numerical continuation and
augmented Newton iterations.
"""

from time import perf_counter

import numpy as np

from elements.kinematics_cache import ElementKinematicsCache

from assembly.internal_force import assemble_internal_force
from assembly.tangent_stiffness import assemble_tangent_stiffness
from solvers.common import _nonlinear_settings, _update_trial_gauss_states
from solvers.profiler import SolverProfiler
from solvers.linear_algebra import (
    extract_submatrix,
    solve_augmented_arc_system,
    solve_direct,
    solve_least_squares,
)
from stage.safety import (
    normalize_strength_reduction_settings,
    apply_strength_reduction,
    restore_reference_strength,
    summarize_gauss_state,
    displacement_metrics,
    characteristic_length,
    arc_metric_squared,
    predictor_direction_sign,
    plateau_diagnostics,
)

def _solve_strength_reduction_path(
    nodes,
    mesh_elements,
    material_models,
    state_manager,
    fixed_dofs,
    fixed_dof_array,
    free_dofs,
    u_start,
    f_external,
    solver_settings,
    strength_reduction_settings=None,
    stage_id=None,
    stage_name=None,
    active_element_ids=None,
    reaction_dof_array=None,
    kinematics_cache=None,
    profiler=None,
    sparse_pattern=None,
):
    """Advance one arc-length phi-c strength-reduction safety phase.

    Unlike the first GeoPyFEM safety prototype, one failed Newton target does
    NOT establish a permanent upper safety-factor bracket.  The continuation
    unknowns are displacement and SigmaMsf together.  This lets the solver
    follow a limit path on which SigmaMsf becomes nearly constant, or even
    decreases slightly, while the failure mechanism continues to deform.

    Equilibrium:

        R(u, M) = Fext - Fint(u, M) = 0

    Arc constraint (dimensionless):

        rms(Delta u_free / Lchar)^2
        + (w_M Delta M)^2
        = Delta s^2

    where M = SigmaMsf and Lchar is the model bounding-box diagonal times the
    user displacement-scale factor.
    """
    if profiler is None:
        profiler = SolverProfiler()

    if kinematics_cache is None:
        kinematics_cache = ElementKinematicsCache(
            nodes=nodes,
            mesh_elements=mesh_elements,
        )

    settings = normalize_strength_reduction_settings(
        strength_reduction_settings
    )

    newton = _nonlinear_settings(
        solver_settings=solver_settings,
        load_steps=1,
    )

    max_iterations = newton["max_iterations"]
    force_tolerance = newton["force_tolerance"]
    matrix_storage = newton["matrix_storage"]

    active_element_ids = (
        None
        if active_element_ids is None
        else {int(element_id) for element_id in active_element_ids}
    )

    fixed_dof_array = np.asarray(fixed_dof_array, dtype=int)
    free_dofs = np.asarray(free_dofs, dtype=int)

    if reaction_dof_array is None:
        reaction_dof_array = fixed_dof_array.copy()
    else:
        reaction_dof_array = np.asarray(reaction_dof_array, dtype=int)

    if free_dofs.size == 0:
        raise ValueError("Safety analysis requires at least one free DOF.")

    u = np.asarray(u_start, dtype=float).copy()
    u_safety_start = u.copy()
    u_previous_accepted = u.copy()
    f_external = np.asarray(f_external, dtype=float).copy()

    restore_reference_strength(material_models)
    current_strength_summary = apply_strength_reduction(
        material_models,
        settings["initial_factor"],
    )

    current_factor = float(settings["initial_factor"])

    model_length = characteristic_length(nodes)
    displacement_scale = (
        model_length * settings["displacement_scale_factor"]
    )
    factor_weight = float(settings["factor_weight"])

    accepted_steps = 0
    total_newton_corrections = 0
    total_residual_evaluations = 0
    cutback_count = 0
    consecutive_cutbacks = 0

    step_history = []
    attempt_history = []

    previous_arc_delta_u = None
    previous_arc_delta_factor = None

    initial_arc_radius = None
    current_arc_radius = None
    minimum_arc_radius = None
    maximum_arc_radius = None

    peak_factor = current_factor
    plateau_factor = None
    plateau_factor_range = None
    final_plateau_diagnostics = None

    label = (
        "safety phase"
        if stage_id is None
        else f'stage {stage_id} "{stage_name}"'
    )

    def evaluate_trial(u_eval, factor_eval, need_tangent=False):
        """Evaluate one uncommitted safety state from the same committed GP state."""
        apply_strength_reduction(material_models, factor_eval)
        state_manager.reset_trial()

        _update_trial_gauss_states(
            nodes=nodes,
            mesh_elements=mesh_elements,
            u=u_eval,
            material_models=material_models,
            state_manager=state_manager,
            active_element_ids=active_element_ids,
            stage_id=stage_id,
            stage_name=stage_name,
            load_factor=float(factor_eval),
            increment_number=accepted_steps + 1,
            iteration_number=None,
            load_path_kind="strength_reduction",
            kinematics_cache=kinematics_cache,
            profiler=profiler,
        )

        f_internal_eval = assemble_internal_force(
            nodes=nodes,
            mesh_elements=mesh_elements,
            gauss_states=state_manager.trial,
            active_element_ids=active_element_ids,
            kinematics_cache=kinematics_cache,
            profiler=profiler,
        )
        residual_eval = f_external - f_internal_eval

        K_eval = None
        if need_tangent:
            K_eval = assemble_tangent_stiffness(
                nodes=nodes,
                mesh_elements=mesh_elements,
                gauss_tangents=state_manager.trial_tangents,
                active_element_ids=active_element_ids,
                matrix_storage=matrix_storage,
                kinematics_cache=kinematics_cache,
                profiler=profiler,
                sparse_pattern=sparse_pattern,
            )

        return f_internal_eval, residual_eval, K_eval

    def evaluate_with_factor_sensitivity(u_eval, factor_eval):
        """Return current trial state, tangent and dFint/dSigmaMsf numerically."""
        sensitivity_start = perf_counter() if profiler is not None else None
        if profiler is not None:
            profiler.increment("arc_factor_sensitivity_evaluations")

        relative_h = float(settings["sensitivity_relative_step"])
        minimum_h = float(settings["sensitivity_minimum_step"])
        h = max(minimum_h, relative_h * max(abs(factor_eval), 1.0))

        plus_factor = factor_eval + h
        minus_factor = factor_eval - h

        if minus_factor >= 1.0:
            f_plus, _, _ = evaluate_trial(
                u_eval=u_eval,
                factor_eval=plus_factor,
                need_tangent=False,
            )
            f_minus, _, _ = evaluate_trial(
                u_eval=u_eval,
                factor_eval=minus_factor,
                need_tangent=False,
            )
            q_factor = (f_plus - f_minus) / (2.0 * h)
        else:
            f_base, _, _ = evaluate_trial(
                u_eval=u_eval,
                factor_eval=factor_eval,
                need_tangent=False,
            )
            f_plus, _, _ = evaluate_trial(
                u_eval=u_eval,
                factor_eval=plus_factor,
                need_tangent=False,
            )
            q_factor = (f_plus - f_base) / h

        # Restore both material factor and trial state to the actual Newton
        # point.  No finite-difference evaluation is committed.
        f_current, r_current, K_current = evaluate_trial(
            u_eval=u_eval,
            factor_eval=factor_eval,
            need_tangent=True,
        )

        if profiler is not None:
            profiler.add_time(
                "arc_factor_sensitivity",
                perf_counter() - sensitivity_start,
            )

        return f_current, r_current, K_current, q_factor

    # Parent-stage equilibrium is the physical starting point of safety.
    final_f_internal = assemble_internal_force(
        nodes=nodes,
        mesh_elements=mesh_elements,
        gauss_states=state_manager.committed,
        active_element_ids=active_element_ids,
        kinematics_cache=kinematics_cache,
        profiler=profiler,
    )
    final_residual = f_external - final_f_internal

    force_reference_norm = max(
        np.linalg.norm(f_external[free_dofs]),
        np.linalg.norm(final_f_internal[free_dofs]),
        1.0,
    )

    baseline_state_summary = summarize_gauss_state(
        state_manager.committed,
        active_element_ids=active_element_ids,
    )
    baseline_displacement = displacement_metrics(
        u=u,
        u_safety_start=u_safety_start,
        u_previous=None,
    )

    step_history.append({
        "step": 0,
        "attempt": 0,
        "safety_factor": current_factor,
        "strength_reduction_factor": current_factor,
        "factor_increment": 0.0,
        "arc_length_radius": 0.0,
        "arc_constraint_residual": 0.0,
        "relative_arc_constraint_residual": 0.0,
        "iterations": 0,
        "newton_iterations": 0,
        "residual_evaluations": 0,
        "residual_norm": float(np.linalg.norm(final_residual[free_dofs])),
        "relative_force_residual": float(
            np.linalg.norm(final_residual[free_dofs]) / force_reference_norm
        ),
        **baseline_displacement,
        **baseline_state_summary,
        "material_strengths": current_strength_summary,
        "converged": True,
    })

    print(f"solve arc-length phi-c strength reduction: {label}")
    print("  external load: fixed at parent-stage equilibrium")
    print("  global matrix storage:", matrix_storage)
    print("  initial SigmaMsf: 1.000000")
    print(
        "  arc metric:",
        f"Lchar={model_length:.6g},",
        f"displacement scale={displacement_scale:.6g},",
        f"factor weight={factor_weight:.6g}",
    )
    print(
        "  initial factor increment used to calibrate arc radius:",
        f"{settings['initial_factor_increment']:.6g}",
    )
    print(
        "  plateau window:",
        f"{settings['window_steps']} accepted steps,",
        f"SigmaMsf range <= {settings['factor_range_tolerance']:.6g},",
        "displacement growth ratio >= "
        f"{settings['displacement_growth_ratio']:.6g}",
    )

    safety_status = "running"
    failure_mechanism_detected = False
    last_failed_factor = None

    while accepted_steps < settings["maximum_steps"]:
        if (
            current_factor >= settings["maximum_factor"] - 1.0e-12
            and (
                previous_arc_delta_factor is None
                or previous_arc_delta_factor >= 0.0
            )
        ):
            safety_status = "maximum_factor_reached"
            break

        # --------------------------------------------------------------
        # ARC-LENGTH PREDICTOR FROM THE LAST COMMITTED EQUILIBRIUM POINT
        # --------------------------------------------------------------
        try:
            (
                f_start_eval,
                r_start_eval,
                K_start,
                q_start,
            ) = evaluate_with_factor_sensitivity(
                u_eval=u,
                factor_eval=current_factor,
            )
        except (np.linalg.LinAlgError, ValueError) as error:
            safety_status = "predictor_evaluation_failed"
            attempt_history.append({
                "attempt": len(attempt_history) + 1,
                "accepted": False,
                "safety_factor_start": float(current_factor),
                "failure_reason": f"predictor evaluation failed: {error}",
            })
            break

        K_ff = extract_submatrix(
            K_start,
            free_dofs,
        )
        q_ff = q_start[free_dofs]

        try:
            z_free = solve_direct(K_ff, -q_ff, profiler=profiler)
            predictor_linear_solver = "direct"
        except np.linalg.LinAlgError:
            # At a limit point K may be singular while the augmented system is
            # still traversable.  Least-squares is used only for the tangent
            # predictor; the corrector itself still solves the augmented
            # equilibrium/arc system.
            z_free = solve_least_squares(K_ff, -q_ff, profiler=profiler)
            predictor_linear_solver = "least_squares_limit_point"

        z = np.zeros_like(u)
        z[free_dofs] = z_free

        unit_metric = arc_metric_squared(
            delta_u=z,
            delta_factor=1.0,
            free_dofs=free_dofs,
            displacement_scale=displacement_scale,
            factor_weight=factor_weight,
        )
        unit_metric = max(unit_metric, 1.0e-30)

        if initial_arc_radius is None:
            initial_dm = float(settings["initial_factor_increment"])
            initial_arc_radius = abs(initial_dm) * np.sqrt(unit_metric)
            initial_arc_radius = max(initial_arc_radius, 1.0e-12)
            current_arc_radius = initial_arc_radius
            minimum_arc_radius = (
                initial_arc_radius * settings["minimum_radius_factor"]
            )
            maximum_arc_radius = (
                initial_arc_radius * settings["maximum_radius_factor"]
            )

            print(
                "  calibrated arc radius:",
                f"initial={initial_arc_radius:.6e},",
                f"min={minimum_arc_radius:.6e},",
                f"max={maximum_arc_radius:.6e}",
            )

        predictor_sign = predictor_direction_sign(
            tangent_du_per_factor=z,
            previous_delta_u=previous_arc_delta_u,
            previous_delta_factor=previous_arc_delta_factor,
            free_dofs=free_dofs,
            displacement_scale=displacement_scale,
            factor_weight=factor_weight,
        )

        predictor_dm = (
            predictor_sign
            * current_arc_radius
            / np.sqrt(unit_metric)
        )

        # MaximumFactor is a user stop limit, not a collapse bracket.  If the
        # positive predictor would cross it, shorten only this step.
        step_arc_radius = current_arc_radius
        if (
            predictor_dm > 0.0
            and current_factor + predictor_dm > settings["maximum_factor"]
        ):
            predictor_dm = settings["maximum_factor"] - current_factor
            if predictor_dm <= 1.0e-14:
                safety_status = "maximum_factor_reached"
                break

            step_arc_radius = np.sqrt(
                arc_metric_squared(
                    delta_u=z * predictor_dm,
                    delta_factor=predictor_dm,
                    free_dofs=free_dofs,
                    displacement_scale=displacement_scale,
                    factor_weight=factor_weight,
                )
            )

        if current_factor + predictor_dm < 1.0:
            # The physical safety continuation is defined only for M >= 1.
            predictor_dm = 1.0 - current_factor
            if abs(predictor_dm) <= 1.0e-14:
                safety_status = "minimum_factor_reached"
                break
            step_arc_radius = np.sqrt(
                arc_metric_squared(
                    delta_u=z * predictor_dm,
                    delta_factor=predictor_dm,
                    free_dofs=free_dofs,
                    displacement_scale=displacement_scale,
                    factor_weight=factor_weight,
                )
            )

        step_u_start = u.copy()
        step_factor_start = float(current_factor)

        u_iter = step_u_start + z * predictor_dm
        factor_iter = step_factor_start + predictor_dm

        attempt_number = len(attempt_history) + 1
        attempt_newton_corrections = 0
        attempt_residual_evaluations = 0
        converged = False
        failure_reason = None

        print(
            f"  safety attempt {attempt_number}: predictor "
            f"SigmaMsf {step_factor_start:.6f} -> {factor_iter:.6f}, "
            f"arc={step_arc_radius:.6e}, "
            f"predictor={predictor_linear_solver}"
        )

        for residual_evaluation in range(1, max_iterations + 1):
            attempt_residual_evaluations += 1
            total_residual_evaluations += 1

            if (
                factor_iter < 1.0 - 1.0e-10
                or factor_iter > settings["maximum_factor"] + 1.0e-8
            ):
                failure_reason = (
                    f"arc corrector left allowed SigmaMsf range: "
                    f"{factor_iter:.6f}"
                )
                break

            try:
                (
                    f_internal,
                    residual,
                    K_tangent,
                    q_factor,
                ) = evaluate_with_factor_sensitivity(
                    u_eval=u_iter,
                    factor_eval=factor_iter,
                )
            except (np.linalg.LinAlgError, ValueError) as error:
                failure_reason = (
                    "local safety evaluation failed: "
                    f"{error}"
                )
                break

            residual_norm = np.linalg.norm(residual[free_dofs])
            relative_force_residual = residual_norm / force_reference_norm
            force_ok = relative_force_residual <= force_tolerance

            arc_delta_u = u_iter - step_u_start
            arc_delta_factor = factor_iter - step_factor_start

            arc_value = arc_metric_squared(
                delta_u=arc_delta_u,
                delta_factor=arc_delta_factor,
                free_dofs=free_dofs,
                displacement_scale=displacement_scale,
                factor_weight=factor_weight,
            )
            arc_constraint_residual = arc_value - step_arc_radius ** 2
            relative_arc_residual = (
                abs(arc_constraint_residual)
                / max(step_arc_radius ** 2, 1.0e-30)
            )
            arc_ok = (
                relative_arc_residual
                <= settings["constraint_tolerance"]
            )

            if residual_evaluation == 1 and force_ok and arc_ok:
                delta_u = np.zeros_like(u_iter)
                delta_factor = 0.0
                relative_correction = 0.0
                correction_ok = True
            else:
                K_ff = extract_submatrix(
                    K_tangent,
                    free_dofs,
                )
                q_ff = q_factor[free_dofs]

                nfree = float(free_dofs.size)
                grad_u = (
                    2.0
                    * arc_delta_u[free_dofs]
                    / (
                        nfree
                        * displacement_scale ** 2
                    )
                )
                grad_factor = (
                    2.0
                    * factor_weight ** 2
                    * arc_delta_factor
                )

                try:
                    correction = solve_augmented_arc_system(
                        K_ff=K_ff,
                        q_ff=q_ff,
                        grad_u=grad_u,
                        grad_factor=grad_factor,
                        rhs_equilibrium=residual[free_dofs],
                        rhs_constraint=-arc_constraint_residual,
                        profiler=profiler,
                    )
                except np.linalg.LinAlgError as error:
                    failure_reason = (
                        "augmented arc-length solve failed: "
                        f"{error}"
                    )
                    break

                delta_u = np.zeros_like(u_iter)
                delta_u[free_dofs] = correction[:-1]
                delta_factor = float(correction[-1])

                attempt_newton_corrections += 1
                total_newton_corrections += 1

                correction_metric = np.sqrt(
                    arc_metric_squared(
                        delta_u=delta_u,
                        delta_factor=delta_factor,
                        free_dofs=free_dofs,
                        displacement_scale=displacement_scale,
                        factor_weight=factor_weight,
                    )
                )
                relative_correction = (
                    correction_metric
                    / max(step_arc_radius, 1.0e-30)
                )
                correction_ok = (
                    relative_correction
                    <= settings["correction_tolerance"]
                )

            print(
                f"    residual evaluation {residual_evaluation}: "
                f"R_force={relative_force_residual:.6e}, "
                f"R_arc={relative_arc_residual:.6e}, "
                f"R_corr={relative_correction:.6e}, "
                f"SigmaMsf={factor_iter:.6f}"
            )

            if force_ok and arc_ok and correction_ok:
                # evaluate_with_factor_sensitivity() restores the trial state
                # to (u_iter, factor_iter), so this commit is consistent.
                state_manager.commit()

                u_new = u_iter.copy()
                factor_new = float(factor_iter)
                final_f_internal = f_internal.copy()
                final_residual = residual.copy()

                accepted_steps += 1
                consecutive_cutbacks = 0

                metrics = displacement_metrics(
                    u=u_new,
                    u_safety_start=u_safety_start,
                    u_previous=u_previous_accepted,
                )
                state_summary = summarize_gauss_state(
                    state_manager.committed,
                    active_element_ids=active_element_ids,
                )
                strength_summary = apply_strength_reduction(
                    material_models,
                    factor_new,
                )

                accepted_delta_u = u_new - step_u_start
                accepted_delta_factor = factor_new - step_factor_start

                peak_factor = max(peak_factor, factor_new)

                step_record = {
                    "step": accepted_steps,
                    "attempt": attempt_number,
                    "safety_factor": factor_new,
                    "strength_reduction_factor": factor_new,
                    "factor_increment": float(accepted_delta_factor),
                    "predictor_factor_increment": float(predictor_dm),
                    "arc_length_radius": float(step_arc_radius),
                    "arc_constraint_residual": float(
                        arc_constraint_residual
                    ),
                    "relative_arc_constraint_residual": float(
                        relative_arc_residual
                    ),
                    "relative_arc_correction": float(relative_correction),
                    "iterations": attempt_newton_corrections,
                    "newton_iterations": attempt_newton_corrections,
                    "residual_evaluations": attempt_residual_evaluations,
                    "residual_norm": float(residual_norm),
                    "relative_force_residual": float(
                        relative_force_residual
                    ),
                    **metrics,
                    **state_summary,
                    "material_strengths": strength_summary,
                    "converged": True,
                }

                step_history.append(step_record)
                attempt_history.append({
                    **step_record,
                    "accepted": True,
                    "failure_reason": None,
                })

                u_previous_accepted = u_new.copy()
                u = u_new
                current_factor = factor_new
                current_strength_summary = strength_summary
                previous_arc_delta_u = accepted_delta_u.copy()
                previous_arc_delta_factor = float(accepted_delta_factor)

                final_plateau_diagnostics = plateau_diagnostics(
                    history=step_history,
                    settings=settings,
                    characteristic_length_value=model_length,
                )

                print(
                    "    safety arc step accepted: "
                    f"SigmaMsf={current_factor:.6f}, "
                    f"dMsf={accepted_delta_factor:+.6e}, "
                    f"max|U_safety|="
                    f"{metrics['maximum_safety_displacement']:.6e}, "
                    f"yielded GP={state_summary['yielded_gauss_points']}/"
                    f"{state_summary['gauss_points']}"
                )

                if final_plateau_diagnostics["plateau"]:
                    safety_status = "failure_plateau"
                    failure_mechanism_detected = True
                    plateau_factor = final_plateau_diagnostics[
                        "plateau_factor"
                    ]
                    plateau_factor_range = (
                        final_plateau_diagnostics["factor_min"],
                        final_plateau_diagnostics["factor_max"],
                    )

                # Adapt radius for the NEXT accepted path segment.
                if (
                    attempt_newton_corrections
                    <= settings["fast_convergence_iterations"]
                ):
                    current_arc_radius = min(
                        step_arc_radius * settings["growth_factor"],
                        maximum_arc_radius,
                    )
                elif (
                    attempt_newton_corrections
                    >= settings["slow_convergence_iterations"]
                ):
                    current_arc_radius = max(
                        step_arc_radius * settings["cutback_factor"],
                        minimum_arc_radius,
                    )
                else:
                    current_arc_radius = min(
                        max(step_arc_radius, minimum_arc_radius),
                        maximum_arc_radius,
                    )

                converged = True
                break

            u_iter += delta_u
            factor_iter += delta_factor

        if converged:
            if failure_mechanism_detected:
                break
            continue

        # --------------------------------------------------------------
        # FAILED ARC STEP -> REVERT + REDUCE ARC RADIUS, NO FS BRACKET
        # --------------------------------------------------------------
        state_manager.revert()
        apply_strength_reduction(material_models, current_factor)

        if failure_reason is None:
            failure_reason = (
                f"maximum nonlinear residual evaluations "
                f"({max_iterations}) reached"
            )

        cutback_count += 1
        consecutive_cutbacks += 1
        last_failed_factor = float(factor_iter)

        attempt_history.append({
            "attempt": attempt_number,
            "safety_factor_start": float(step_factor_start),
            "safety_factor_target": float(factor_iter),
            "predictor_factor_increment": float(predictor_dm),
            "arc_length_radius": float(step_arc_radius),
            "iterations": attempt_newton_corrections,
            "newton_iterations": attempt_newton_corrections,
            "residual_evaluations": attempt_residual_evaluations,
            "accepted": False,
            "failure_reason": failure_reason,
        })

        reduced_radius = step_arc_radius * settings["cutback_factor"]

        print("    safety arc step rejected:", failure_reason)
        print(
            "    rollback to",
            f"SigmaMsf={current_factor:.6f};",
            "cut arc radius",
            f"{step_arc_radius:.6e} -> {reduced_radius:.6e}",
        )

        if consecutive_cutbacks >= settings["maximum_cutbacks"]:
            safety_status = "maximum_cutbacks_without_confirmed_plateau"
            break

        if reduced_radius < minimum_arc_radius * (1.0 - 1.0e-12):
            safety_status = "minimum_arc_radius_without_confirmed_plateau"
            break

        current_arc_radius = max(reduced_radius, minimum_arc_radius)

    if safety_status == "running":
        safety_status = "maximum_steps_reached"

    current_strength_summary = apply_strength_reduction(
        material_models,
        current_factor,
    )

    final_state_summary = summarize_gauss_state(
        state_manager.committed,
        active_element_ids=active_element_ids,
    )
    final_metrics = displacement_metrics(
        u=u,
        u_safety_start=u_safety_start,
        u_previous=None,
    )

    final_plateau_diagnostics = plateau_diagnostics(
        history=step_history,
        settings=settings,
        characteristic_length_value=model_length,
    )

    if final_plateau_diagnostics.get("peak_factor") is not None:
        peak_factor = max(
            peak_factor,
            float(final_plateau_diagnostics["peak_factor"]),
        )

    if final_plateau_diagnostics.get("plateau"):
        failure_mechanism_detected = True
        safety_status = "failure_plateau"
        plateau_factor = final_plateau_diagnostics["plateau_factor"]
        plateau_factor_range = (
            final_plateau_diagnostics["factor_min"],
            final_plateau_diagnostics["factor_max"],
        )

    reaction = np.zeros_like(final_f_internal)
    reaction[reaction_dof_array] = (
        final_f_internal[reaction_dof_array]
        - f_external[reaction_dof_array]
    )

    final_free_residual_norm = np.linalg.norm(final_residual[free_dofs])

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

    # User-facing FS follows the maximum converged SigmaMsf on the traced
    # equilibrium path once a deformation plateau has confirmed failure.
    safety_factor = (
        float(peak_factor)
        if failure_mechanism_detected
        else None
    )

    print()
    print("  arc-length safety continuation finished")
    print("  status:", safety_status)
    print("  last converged SigmaMsf:", f"{current_factor:.6f}")
    print("  peak converged SigmaMsf:", f"{peak_factor:.6f}")
    if plateau_factor is not None:
        print("  plateau mean SigmaMsf:", f"{plateau_factor:.6f}")
    if plateau_factor_range is not None:
        print(
            "  plateau SigmaMsf range:",
            f"{plateau_factor_range[0]:.6f} - "
            f"{plateau_factor_range[1]:.6f}",
        )
    if last_failed_factor is not None:
        print(
            "  last rejected SigmaMsf (diagnostic only):",
            f"{last_failed_factor:.6f}",
        )
    if safety_factor is not None:
        print("  reported safety factor:", f"{safety_factor:.6f}")
    print(
        "  maximum safety displacement:",
        f"{final_metrics['maximum_safety_displacement']:.6e}",
    )

    return {
        "u": u,
        "u_safety_start": u_safety_start,
        "u_safety": u - u_safety_start,
        "K": final_K_tangent,
        "f": f_external,
        "f_external": f_external,
        "f_internal": final_f_internal,
        "residual": final_residual,
        "reaction": reaction,
        "free_residual_norm": float(final_free_residual_norm),
        "fixed_dofs": fixed_dofs,
        "gauss_states": state_manager.committed,
        "state_manager": state_manager,
        "active_element_ids": (
            None
            if active_element_ids is None
            else sorted(active_element_ids)
        ),
        "converged": True,
        "iterations": total_newton_corrections,
        "newton_iterations": total_newton_corrections,
        "residual_evaluations": total_residual_evaluations,
        "load_steps": accepted_steps,
        "requested_load_steps": accepted_steps,
        "cutbacks": cutback_count,
        "maximum_cutbacks": settings["maximum_cutbacks"],
        "load_stepping_type": "arc_length_strength_reduction",
        "load_path_kind": "arc_length_strength_reduction",
        "final_load_factor": 1.0,
        "step_history": step_history,
        "attempt_history": attempt_history,
        "force_tolerance": force_tolerance,
        "displacement_tolerance": settings["correction_tolerance"],
        "performance_profile": profiler.snapshot(),
        "analysis_type": "static",
        "procedure": "safety",
        "matrix_storage": matrix_storage,
        "linear_solver": newton["linear_solver"],
        "safety_status": safety_status,
        "failure_mechanism_detected": bool(failure_mechanism_detected),
        "safety_factor": safety_factor,
        "final_strength_reduction_factor": float(current_factor),
        "peak_strength_reduction_factor": float(peak_factor),
        "plateau_strength_reduction_factor": (
            None if plateau_factor is None else float(plateau_factor)
        ),
        "plateau_factor_range": plateau_factor_range,
        # Kept for output/backward compatibility.  A failed Newton attempt is
        # no longer a physical upper safety-factor bound.
        "strength_reduction_factor_lower": float(current_factor),
        "strength_reduction_factor_upper": None,
        "last_failed_strength_reduction_factor": last_failed_factor,
        "initial_arc_length_radius": initial_arc_radius,
        "final_arc_length_radius": current_arc_radius,
        "characteristic_length": model_length,
        "safety_displacement_scale": displacement_scale,
        "strength_reduction_settings": dict(settings),
        "strength_reduction_materials": current_strength_summary,
        "safety_history": step_history,
        "safety_attempt_history": attempt_history,
        "safety_displacement_metrics": final_metrics,
        "safety_gauss_summary": final_state_summary,
        "plateau_diagnostics": final_plateau_diagnostics,
    }
