from copy import deepcopy

import numpy as np


DEFAULT_STRENGTH_REDUCTION_SETTINGS = {
    "initial_factor": 1.0,
    "initial_factor_increment": 0.01,
    "maximum_factor": 3.0,
    "minimum_radius_factor": 0.05,
    "maximum_radius_factor": 2.0,
    "cutback_factor": 0.50,
    "maximum_cutbacks": 50,
    "growth_factor": 1.15,
    "fast_convergence_iterations": 4,
    "slow_convergence_iterations": 10,
    "factor_weight": 1.0,
    "displacement_scale_factor": 1.0,
    "sensitivity_relative_step": 1.0e-4,
    "sensitivity_minimum_step": 1.0e-6,
    "constraint_tolerance": 1.0e-7,
    "correction_tolerance": 1.0e-7,
    "window_steps": 12,
    "factor_range_tolerance": 5.0e-3,
    "displacement_growth_ratio": 0.05,
    "minimum_displacement_ratio": 1.0e-3,
    "maximum_steps": 150,
}


def _apply_legacy_setting_aliases(values):
    """Translate the first GeoPyFEM safety XML to the arc-length settings.

    This keeps existing user XML runnable while removing the old interpretation
    that one failed Newton target is a permanent upper safety-factor bracket.
    New XML should use the explicit ArcLength / FailureDetection settings.
    """
    values = dict(values or {})

    if (
        "initial_factor_increment" not in values
        and "initial_increment" in values
    ):
        values["initial_factor_increment"] = values["initial_increment"]

    initial = float(values.get(
        "initial_factor_increment",
        DEFAULT_STRENGTH_REDUCTION_SETTINGS["initial_factor_increment"],
    ))

    if initial > 0.0:
        if (
            "minimum_radius_factor" not in values
            and "minimum_increment" in values
        ):
            values["minimum_radius_factor"] = max(
                1.0e-6,
                min(1.0, float(values["minimum_increment"]) / initial),
            )

        if (
            "maximum_radius_factor" not in values
            and "maximum_increment" in values
        ):
            values["maximum_radius_factor"] = max(
                1.0,
                float(values["maximum_increment"]) / initial,
            )

    if (
        "factor_range_tolerance" not in values
        and "factor_tolerance" in values
    ):
        values["factor_range_tolerance"] = values["factor_tolerance"]

    if (
        "displacement_growth_ratio" not in values
        and "plateau_displacement_growth_ratio" in values
    ):
        values["displacement_growth_ratio"] = values[
            "plateau_displacement_growth_ratio"
        ]

    # The old PlateauRefinements value was a bracket-refinement count and is
    # not mathematically equivalent to a moving-window plateau detector.
    # Keep the new robust default unless WindowSteps is supplied explicitly.

    return values


def normalize_strength_reduction_settings(settings=None):
    """Return validated arc-length phi-c reduction settings.

    The safety phase starts at SigmaMsf = 1.0 from the converged parent-stage
    state.  The external load remains fixed.  During safety continuation both
    displacement and SigmaMsf are solved as unknowns under an arc-length
    constraint, so SigmaMsf may become nearly constant or decrease slightly
    while the failure mechanism continues to deform.
    """
    supplied = _apply_legacy_setting_aliases(settings)
    values = dict(DEFAULT_STRENGTH_REDUCTION_SETTINGS)
    values.update(supplied)

    float_keys = (
        "initial_factor",
        "initial_factor_increment",
        "maximum_factor",
        "minimum_radius_factor",
        "maximum_radius_factor",
        "cutback_factor",
        "growth_factor",
        "factor_weight",
        "displacement_scale_factor",
        "sensitivity_relative_step",
        "sensitivity_minimum_step",
        "constraint_tolerance",
        "correction_tolerance",
        "factor_range_tolerance",
        "displacement_growth_ratio",
        "minimum_displacement_ratio",
    )
    int_keys = (
        "maximum_cutbacks",
        "fast_convergence_iterations",
        "slow_convergence_iterations",
        "window_steps",
        "maximum_steps",
    )

    for key in float_keys:
        values[key] = float(values[key])

    for key in int_keys:
        values[key] = int(values[key])

    if abs(values["initial_factor"] - 1.0) > 1.0e-12:
        raise ValueError(
            "StrengthReduction InitialFactor must be 1.0. "
            "Safety starts from the converged parent-stage strength."
        )

    if values["initial_factor_increment"] <= 0.0:
        raise ValueError(
            "StrengthReduction InitialFactorIncrement must be positive."
        )

    if values["maximum_factor"] <= 1.0:
        raise ValueError("StrengthReduction MaximumFactor must be > 1.0.")

    if not (0.0 < values["minimum_radius_factor"] <= 1.0):
        raise ValueError(
            "StrengthReduction MinimumRadiusFactor must satisfy 0 < value <= 1."
        )

    if values["maximum_radius_factor"] < 1.0:
        raise ValueError(
            "StrengthReduction MaximumRadiusFactor must be >= 1.0."
        )

    if not (0.0 < values["cutback_factor"] < 1.0):
        raise ValueError(
            "StrengthReduction CutbackFactor must satisfy 0 < value < 1."
        )

    if values["maximum_cutbacks"] < 1:
        raise ValueError("StrengthReduction MaximumCutbacks must be >= 1.")

    if values["growth_factor"] < 1.0:
        raise ValueError("StrengthReduction GrowthFactor must be >= 1.0.")

    if values["fast_convergence_iterations"] < 1:
        raise ValueError(
            "StrengthReduction FastConvergenceIterations must be >= 1."
        )

    if (
        values["slow_convergence_iterations"]
        < values["fast_convergence_iterations"]
    ):
        raise ValueError(
            "StrengthReduction SlowConvergenceIterations must be >= "
            "FastConvergenceIterations."
        )

    if values["factor_weight"] <= 0.0:
        raise ValueError("StrengthReduction FactorWeight must be positive.")

    if values["displacement_scale_factor"] <= 0.0:
        raise ValueError(
            "StrengthReduction DisplacementScaleFactor must be positive."
        )

    if values["sensitivity_relative_step"] <= 0.0:
        raise ValueError(
            "StrengthReduction SensitivityRelativeStep must be positive."
        )

    if values["sensitivity_minimum_step"] <= 0.0:
        raise ValueError(
            "StrengthReduction SensitivityMinimumStep must be positive."
        )

    if values["constraint_tolerance"] <= 0.0:
        raise ValueError(
            "StrengthReduction ConstraintTolerance must be positive."
        )

    if values["correction_tolerance"] <= 0.0:
        raise ValueError(
            "StrengthReduction CorrectionTolerance must be positive."
        )

    if values["window_steps"] < 3:
        raise ValueError(
            "StrengthReduction WindowSteps must be >= 3."
        )

    if values["factor_range_tolerance"] <= 0.0:
        raise ValueError(
            "StrengthReduction FactorRangeTolerance must be positive."
        )

    if not (0.0 <= values["displacement_growth_ratio"] <= 1.0):
        raise ValueError(
            "StrengthReduction DisplacementGrowthRatio must be between 0 and 1."
        )

    if values["minimum_displacement_ratio"] < 0.0:
        raise ValueError(
            "StrengthReduction MinimumDisplacementRatio must be >= 0."
        )

    if values["maximum_steps"] < 1:
        raise ValueError("StrengthReduction MaximumSteps must be >= 1.")

    return values


def validate_safety_stage_definition(stage_definition):
    """Reject construction/load changes inside a safety stage."""
    stage = dict(stage_definition or {})

    forbidden_lists = {
        "activate_loads": "ActivateLoads",
        "deactivate_loads": "DeactivateLoads",
        "activate_regions": "ActivateRegions",
        "deactivate_regions": "DeactivateRegions",
    }

    for key, label in forbidden_lists.items():
        if stage.get(key):
            raise ValueError(
                f"Safety stage cannot contain <{label}>. "
                "Complete construction/loading first, then start safety from "
                "the converged parent stage."
            )

    if stage.get("gravity") is not None:
        raise ValueError(
            "Safety stage cannot redefine <Gravity>. "
            "The external load configuration must remain fixed during "
            "phi-c reduction."
        )

    if stage.get("k0") is not None:
        raise ValueError("Safety stage cannot contain a <K0> definition.")


def _reducible_material_items(material_models):
    items = []

    for region, material in (material_models or {}).items():
        setter = getattr(material, "set_strength_reduction_factor", None)
        if callable(setter):
            items.append((region, material))

    return items


def apply_strength_reduction(material_models, factor):
    """Apply one ABSOLUTE SigmaMsf to all supported materials."""
    factor = float(factor)

    if factor < 1.0:
        raise ValueError("Strength reduction factor must be >= 1.0.")

    reducible = _reducible_material_items(material_models)

    if not reducible:
        raise ValueError(
            "Safety analysis requires at least one material that implements "
            "set_strength_reduction_factor(). The current implementation "
            "supports MohrCoulomb through this interface."
        )

    summary = {}

    for region, material in reducible:
        material.set_strength_reduction_factor(factor)

        getter = getattr(material, "strength_reduction_state", None)
        if callable(getter):
            summary[region] = deepcopy(getter())
        else:
            summary[region] = {
                "factor": factor,
                "model": getattr(material, "model_name", type(material).__name__),
            }

    return summary


def restore_reference_strength(material_models):
    """Restore all strength-reducible materials to their input parameters."""
    restored = {}

    for region, material in _reducible_material_items(material_models):
        restore = getattr(material, "restore_reference_strength", None)
        if callable(restore):
            restore()
        else:
            material.set_strength_reduction_factor(1.0)

        getter = getattr(material, "strength_reduction_state", None)
        if callable(getter):
            restored[region] = deepcopy(getter())

    return restored


def summarize_gauss_state(gauss_states, active_element_ids=None):
    """Summarize yielded points and MC return regions for safety diagnostics."""
    active = (
        None
        if active_element_ids is None
        else {int(element_id) for element_id in active_element_ids}
    )

    total = 0
    yielded = 0
    region_counts = {}

    for element_id, states in (gauss_states or {}).items():
        if active is not None and int(element_id) not in active:
            continue

        for state in states:
            total += 1

            if bool(getattr(state, "yielded", False)):
                yielded += 1

            region = getattr(state, "internal_variables", {}).get("mc_region")

            if region is not None:
                region = str(region)
                region_counts[region] = region_counts.get(region, 0) + 1

    return {
        "gauss_points": total,
        "yielded_gauss_points": yielded,
        "yielded_ratio": 0.0 if total == 0 else float(yielded) / float(total),
        "mc_region_counts": region_counts,
        "apex_gauss_points": int(
            region_counts.get("apex_fortran", 0)
            + region_counts.get("apex_projection", 0)
        ),
    }


def displacement_metrics(u, u_safety_start, u_previous=None):
    """Return displacement measures relative to the start of safety."""
    u = np.asarray(u, dtype=float).reshape(-1)
    u0 = np.asarray(u_safety_start, dtype=float).reshape(-1)

    if u.shape != u0.shape:
        raise ValueError("Safety displacement vectors must have the same size.")

    safety = u - u0
    safety_nodes = safety.reshape(-1, 2)
    safety_magnitude = np.linalg.norm(safety_nodes, axis=1)

    result = {
        "safety_displacement_norm": float(np.linalg.norm(safety)),
        "maximum_safety_displacement": float(
            np.max(safety_magnitude) if safety_magnitude.size else 0.0
        ),
    }

    if u_previous is None:
        result.update({
            "step_displacement_norm": 0.0,
            "maximum_step_displacement": 0.0,
        })
        return result

    up = np.asarray(u_previous, dtype=float).reshape(-1)
    if up.shape != u.shape:
        raise ValueError("Previous displacement vector has incompatible size.")

    step = u - up
    step_nodes = step.reshape(-1, 2)
    step_magnitude = np.linalg.norm(step_nodes, axis=1)

    result.update({
        "step_displacement_norm": float(np.linalg.norm(step)),
        "maximum_step_displacement": float(
            np.max(step_magnitude) if step_magnitude.size else 0.0
        ),
    })

    return result


def characteristic_length(nodes):
    """Return a mesh-unit characteristic length from the model bounding box."""
    xy = np.asarray(nodes, dtype=float)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return 1.0

    xy = xy[:, :2]
    span = np.ptp(xy, axis=0)
    length = float(np.linalg.norm(span))
    return max(length, 1.0e-12)


def arc_metric_squared(
    delta_u,
    delta_factor,
    free_dofs,
    displacement_scale,
    factor_weight,
):
    """Dimensionless spherical arc metric used by the safety continuation."""
    du = np.asarray(delta_u, dtype=float).reshape(-1)
    free = np.asarray(free_dofs, dtype=int).reshape(-1)

    if free.size:
        rms = float(np.linalg.norm(du[free]) / np.sqrt(float(free.size)))
    else:
        rms = 0.0

    u_term = rms / max(float(displacement_scale), 1.0e-30)
    m_term = float(factor_weight) * float(delta_factor)

    return float(u_term * u_term + m_term * m_term)


def predictor_direction_sign(
    tangent_du_per_factor,
    previous_delta_u,
    previous_delta_factor,
    free_dofs,
    displacement_scale,
    factor_weight,
):
    """Choose the arc predictor sign that continues the previous path branch."""
    if previous_delta_u is None or previous_delta_factor is None:
        return 1.0

    z = np.asarray(tangent_du_per_factor, dtype=float).reshape(-1)
    prev = np.asarray(previous_delta_u, dtype=float).reshape(-1)
    free = np.asarray(free_dofs, dtype=int).reshape(-1)

    scale = max(float(displacement_scale), 1.0e-30)
    nfree = max(int(free.size), 1)

    displacement_dot = float(
        np.dot(z[free] / scale, prev[free] / scale) / float(nfree)
    ) if free.size else 0.0

    factor_dot = (
        float(factor_weight) ** 2
        * float(previous_delta_factor)
    )

    dot_positive_candidate = displacement_dot + factor_dot

    return 1.0 if dot_positive_candidate >= 0.0 else -1.0


def plateau_diagnostics(history, settings, characteristic_length_value):
    """Detect a PLAXIS-style SigmaMsf plateau from accepted arc-length steps.

    Failure is not inferred from Newton nonconvergence.  A moving window of
    accepted equilibrium points must show a narrow SigmaMsf range while the
    safety displacement continues to grow and plasticity is present.
    """
    accepted = [
        item
        for item in (history or [])
        if int(item.get("step", 0)) > 0 and item.get("converged", False)
    ]

    peak_factor = (
        None
        if not accepted
        else float(max(item["strength_reduction_factor"] for item in accepted))
    )

    window_steps = int(settings["window_steps"])

    if len(accepted) < window_steps:
        return {
            "plateau": False,
            "reason": "insufficient_accepted_steps",
            "window_steps": len(accepted),
            "peak_factor": peak_factor,
            "plateau_factor": None,
            "factor_min": None,
            "factor_max": None,
            "factor_range": None,
            "displacement_growth": None,
            "displacement_growth_ratio": None,
            "maximum_displacement_ratio": None,
        }

    window = accepted[-window_steps:]
    factors = np.asarray(
        [item["strength_reduction_factor"] for item in window],
        dtype=float,
    )
    displacements = np.asarray(
        [item["maximum_safety_displacement"] for item in window],
        dtype=float,
    )

    factor_min = float(np.min(factors))
    factor_max = float(np.max(factors))
    factor_range = factor_max - factor_min
    plateau_factor = float(np.mean(factors))

    displacement_growth = max(
        0.0,
        float(displacements[-1] - displacements[0]),
    )
    displacement_growth_ratio = (
        0.0
        if displacements[-1] <= 0.0
        else displacement_growth / float(displacements[-1])
    )

    length = max(float(characteristic_length_value), 1.0e-30)
    maximum_displacement_ratio = float(displacements[-1] / length)

    last_yielded = int(window[-1].get("yielded_gauss_points", 0))

    factor_plateau = (
        factor_range <= float(settings["factor_range_tolerance"])
    )
    deformation_growing = (
        displacement_growth_ratio
        >= float(settings["displacement_growth_ratio"])
    )
    deformation_developed = (
        maximum_displacement_ratio
        >= float(settings["minimum_displacement_ratio"])
    )
    plasticity_present = last_yielded > 0

    plateau = (
        factor_plateau
        and deformation_growing
        and deformation_developed
        and plasticity_present
    )

    return {
        "plateau": bool(plateau),
        "reason": (
            "plateau_detected"
            if plateau
            else "plateau_conditions_not_met"
        ),
        "window_steps": window_steps,
        "peak_factor": peak_factor,
        "plateau_factor": plateau_factor,
        "factor_min": factor_min,
        "factor_max": factor_max,
        "factor_range": factor_range,
        "factor_plateau": bool(factor_plateau),
        "displacement_growth": displacement_growth,
        "displacement_growth_ratio": float(displacement_growth_ratio),
        "deformation_growing": bool(deformation_growing),
        "maximum_displacement_ratio": maximum_displacement_ratio,
        "deformation_developed": bool(deformation_developed),
        "plasticity_present": bool(plasticity_present),
    }
