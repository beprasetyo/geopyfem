"""Shared mechanical-solver helpers.

This module contains only reusable infrastructure: Gauss-point trial updates,
DOF bookkeeping, staged-domain helpers, nonlinear settings, and small
kinematic utilities.  It does not choose an analysis procedure.
"""

from time import perf_counter

import numpy as np

from assembly.boundary_condition import get_fixed_dofs
from assembly.matrix_builder import element_dof_indices
from elements.elemtype import get_element_type
from elements.kinematics_cache import get_element_kinematics
from hydraulic.groundwater import initialize_hydrostatic_pore_pressure
from material_model.constitutive import MaterialContext

def _initialize_groundwater_if_present(
    nodes,
    mesh_elements,
    state_manager,
    groundwater,
):
    """Initialize static hydrostatic pore pressure in Gauss-point states."""
    if not groundwater or not groundwater.get("enabled", True):
        return None

    summary = initialize_hydrostatic_pore_pressure(
        nodes=nodes,
        mesh_elements=mesh_elements,
        state_manager=state_manager,
        groundwater_specification=groundwater,
    )

    print(
        "initialize hydrostatic groundwater:",
        f'name={summary["name"]},',
        f'type={summary["type"]},',
        f'submerged GP={summary["submerged_gauss_points"]}/'
        f'{summary["total_gauss_points"]},',
        f'max u={summary["maximum_pore_pressure"]:.6g}',
    )

    return summary

def _element_displacement_vector(
    u,
    connectivity,
):
    """Extract one 2-D element displacement vector by global DOF indexing."""
    dofs = element_dof_indices(
        connectivity,
        ndof_node=2,
    )
    return np.asarray(u, dtype=float)[dofs].copy()

def _update_static_linear_gauss_states(
    nodes,
    mesh_elements,
    u,
    material_models,
    state_manager,
    kinematics_cache=None,
    profiler=None,
):
    """Populate TRIAL Gauss-point states from the solved linear field.

    When a shared kinematics cache is supplied, B matrices and element DOF
    maps are reused rather than rebuilt from the reference mesh.
    """
    profile_start = perf_counter() if profiler is not None else None
    if profiler is not None:
        profiler.increment("linear_state_updates")

    state_manager.reset_trial()
    u = np.asarray(u, dtype=float)

    for element in mesh_elements:
        element_id = int(element["id"])
        kinematics = get_element_kinematics(
            nodes=nodes,
            element=element,
            ndof_node=2,
            kinematics_cache=kinematics_cache,
        )
        if kinematics is None:
            continue

        material_region = element["material"]
        if material_region not in material_models:
            raise ValueError(
                f'Element {element_id} uses material region '
                f'"{material_region}", but no material model is available.'
            )

        material = material_models[material_region]
        if not hasattr(material, "stress_tensor"):
            raise NotImplementedError(
                f'Material model "{getattr(material, "model_name", type(material).__name__)}" '
                "does not provide stress_tensor(), which is required to "
                "store the current static-linear Gauss-point state."
            )

        expected_gp = state_manager.number_of_gauss_points(element_id)
        if expected_gp != kinematics.number_of_gauss_points:
            raise ValueError(
                f"Gauss-point state count mismatch in element {element_id}."
            )

        ue = u[kinematics.dofs]

        for gp_index, gp in enumerate(kinematics.gauss_points):
            strain_2d = gp.B @ ue
            stress_tensor = material.stress_tensor(strain_2d)

            state = state_manager.get_trial(element_id, gp_index)
            state.set_plane_strain(strain_2d)
            state.set_effective_stress_tensor(stress_tensor)
            state.plastic_strain[:] = 0.0
            state.equivalent_plastic_strain = 0.0
            state.yielded = False

    if profiler is not None:
        profiler.add_time(
            "linear_state_update",
            perf_counter() - profile_start,
        )

def _update_trial_gauss_states(
    nodes,
    mesh_elements,
    u,
    material_models,
    state_manager,
    active_element_ids=None,
    stage_id=None,
    stage_name=None,
    load_factor=None,
    increment_number=None,
    iteration_number=None,
    load_path_kind=None,
    kinematics_cache=None,
    profiler=None,
):
    """Perform constitutive updates at every active Gauss point.

    The shared kinematics cache is purely geometric.  Material history remains
    in ``state_manager`` and every Newton evaluation still starts from the
    last committed constitutive state.
    """
    profile_start = perf_counter() if profiler is not None else None
    if profiler is not None:
        profiler.increment("constitutive_update_calls")

    active_element_ids = (
        None
        if active_element_ids is None
        else {int(element_id) for element_id in active_element_ids}
    )
    u = np.asarray(u, dtype=float)

    for element in mesh_elements:
        element_id = int(element["id"])
        if active_element_ids is not None and element_id not in active_element_ids:
            continue

        kinematics = get_element_kinematics(
            nodes=nodes,
            element=element,
            ndof_node=2,
            kinematics_cache=kinematics_cache,
        )
        if kinematics is None:
            continue

        material_region = element["material"]
        if material_region not in material_models:
            raise ValueError(
                f'Element {element_id} uses material region '
                f'"{material_region}", but no material model is available.'
            )

        material = material_models[material_region]
        if not hasattr(material, "integrate"):
            raise NotImplementedError(
                f'Material model "{getattr(material, "model_name", type(material).__name__)}" '
                "does not implement the GeoPyFEM integrate() constitutive "
                "interface required by the nonlinear solver."
            )

        ue = u[kinematics.dofs]

        for gp_index, gp in enumerate(kinematics.gauss_points):
            kinematic_strain_2d = gp.B @ ue
            activation_reference_2d = (
                state_manager.get_activation_strain_reference(
                    element_id,
                    gp_index,
                )
            )
            total_strain_2d = kinematic_strain_2d - activation_reference_2d
            committed_state = state_manager.get_committed(element_id, gp_index)
            strain_increment_2d = (
                total_strain_2d
                - np.asarray(committed_state.strain_2d, dtype=float)
            )

            constitutive_start = (
                perf_counter() if profiler is not None else None
            )

            response = material.integrate(
                total_strain_2d=total_strain_2d,
                strain_increment_2d=strain_increment_2d,
                committed_state=committed_state,
                context=MaterialContext(
                    element_id=element_id,
                    gauss_point_index=gp_index,
                    stage_id=stage_id,
                    stage_name=stage_name,
                    load_factor=load_factor,
                    increment_number=increment_number,
                    iteration_number=iteration_number,
                    load_path_kind=load_path_kind,
                    performance_profiler=profiler,
                ),
            )

            if profiler is not None:
                profiler.add_time(
                    "constitutive_integration",
                    perf_counter() - constitutive_start,
                )
                profiler.increment("constitutive_calls")

            state_manager.set_trial_response(
                element_id,
                gp_index,
                response,
            )

    if profiler is not None:
        profiler.add_time(
            "constitutive_update",
            perf_counter() - profile_start,
        )

def _nonlinear_settings(
    solver_settings,
    load_steps,
):
    """
    Validate Newton-Raphson, dual-convergence, and load-stepping settings.

    ``load_steps`` remains meaningful in both modes:

    fixed
        increment = 1 / load_steps

    adaptive
        1 / load_steps is used as the INITIAL increment unless the XML
        explicitly supplies <InitialIncrement>.
    """
    solver_settings = (
        solver_settings
        or {}
    )

    nonlinear_settings = solver_settings.get(
        "nonlinear_solver",
        {},
    )

    method = str(
        nonlinear_settings.get(
            "method",
            "NewtonRaphson",
        )
    ).lower().replace(
        "_",
        "",
    ).replace(
        "-",
        "",
    )

    if method not in (
        "newtonraphson",
        "newton",
    ):
        raise NotImplementedError(
            "GeoPyFEM currently implements only full Newton-Raphson "
            f"for static nonlinear analysis. Requested method={method}."
        )

    max_iterations = int(
        nonlinear_settings.get(
            "max_iterations",
            25,
        )
    )

    # Backward compatibility with the old single tolerance.
    legacy_tolerance = float(
        nonlinear_settings.get(
            "tolerance",
            1.0e-8,
        )
    )

    force_tolerance = float(
        nonlinear_settings.get(
            "force_tolerance",
            legacy_tolerance,
        )
    )

    displacement_tolerance = float(
        nonlinear_settings.get(
            "displacement_tolerance",
            legacy_tolerance,
        )
    )

    if max_iterations < 1:
        raise ValueError(
            "MaxIterations must be >= 1."
        )

    if force_tolerance <= 0.0:
        raise ValueError(
            "ForceTolerance must be positive."
        )

    if displacement_tolerance <= 0.0:
        raise ValueError(
            "DisplacementTolerance must be positive."
        )

    linear_solver = str(
        solver_settings.get(
            "linear_solver",
            "direct",
        )
    ).lower()

    matrix_storage = str(
        solver_settings.get(
            "matrix_storage",
            "dense",
        )
    ).strip().lower()

    if matrix_storage not in ("dense", "sparse"):
        raise ValueError(
            "MatrixStorage must be 'dense' or 'sparse'."
        )

    if linear_solver != "direct":
        raise NotImplementedError(
            f'Linear solver "{linear_solver}" is not implemented '
            "inside Newton-Raphson."
        )

    load_steps = int(
        load_steps
    )

    if load_steps < 1:
        raise ValueError(
            "LoadSteps must be >= 1."
        )

    stepping = dict(
        nonlinear_settings.get(
            "load_stepping",
            {},
        )
        or {}
    )

    stepping_type = str(
        stepping.get(
            "type",
            "fixed",
        )
    ).strip().lower().replace(
        "_",
        ""
    ).replace(
        "-",
        ""
    )

    if stepping_type not in (
        "fixed",
        "adaptive",
    ):
        raise ValueError(
            "LoadStepping type must be 'fixed' or 'adaptive'."
        )

    default_increment = (
        1.0 / load_steps
    )

    if stepping_type == "fixed":
        initial_increment = (
            default_increment
        )
        minimum_increment = (
            default_increment
        )
        maximum_increment = (
            default_increment
        )

    else:
        initial_increment = stepping.get(
            "initial_increment"
        )

        if initial_increment is None:
            initial_increment = (
                default_increment
            )

        initial_increment = float(
            initial_increment
        )

        minimum_increment = stepping.get(
            "minimum_increment"
        )

        if minimum_increment is None:
            minimum_increment = (
                initial_increment / 10.0
            )

        minimum_increment = float(
            minimum_increment
        )

        maximum_increment = stepping.get(
            "maximum_increment"
        )

        if maximum_increment is None:
            maximum_increment = min(
                2.0 * initial_increment,
                1.0,
            )

        maximum_increment = float(
            maximum_increment
        )

    cutback_factor = float(
        stepping.get(
            "cutback_factor",
            0.50,
        )
    )

    maximum_cutbacks = int(
        stepping.get(
            "maximum_cutbacks",
            5,
        )
    )

    growth_factor = float(
        stepping.get(
            "growth_factor",
            1.50,
        )
    )

    fast_iterations = int(
        stepping.get(
            "fast_convergence_iterations",
            4,
        )
    )

    slow_iterations = int(
        stepping.get(
            "slow_convergence_iterations",
            8,
        )
    )

    if not (
        0.0
        < initial_increment
        <= 1.0
    ):
        raise ValueError(
            "InitialIncrement must satisfy 0 < value <= 1."
        )

    if not (
        0.0
        < minimum_increment
        <= initial_increment
    ):
        raise ValueError(
            "MinimumIncrement must satisfy "
            "0 < minimum <= initial increment."
        )

    if not (
        initial_increment
        <= maximum_increment
        <= 1.0
    ):
        raise ValueError(
            "MaximumIncrement must satisfy "
            "initial increment <= maximum <= 1."
        )

    if not (
        0.0
        < cutback_factor
        < 1.0
    ):
        raise ValueError(
            "CutbackFactor must satisfy 0 < value < 1."
        )

    if maximum_cutbacks < 0:
        raise ValueError(
            "MaximumCutbacks must be >= 0."
        )

    if growth_factor < 1.0:
        raise ValueError(
            "GrowthFactor must be >= 1."
        )

    if fast_iterations < 1:
        raise ValueError(
            "FastConvergenceIterations must be >= 1."
        )

    if slow_iterations < fast_iterations:
        raise ValueError(
            "SlowConvergenceIterations must be >= "
            "FastConvergenceIterations."
        )

    return {
        "max_iterations": max_iterations,
        "force_tolerance": force_tolerance,
        "displacement_tolerance": (
            displacement_tolerance
        ),
        "linear_solver": linear_solver,
        "matrix_storage": matrix_storage,
        "load_stepping_type": stepping_type,
        "initial_increment": initial_increment,
        "minimum_increment": minimum_increment,
        "maximum_increment": maximum_increment,
        "cutback_factor": cutback_factor,
        "maximum_cutbacks": maximum_cutbacks,
        "growth_factor": growth_factor,
        "fast_convergence_iterations": (
            fast_iterations
        ),
        "slow_convergence_iterations": (
            slow_iterations
        ),
    }

def _element_ids_for_regions(
    mesh_elements,
    region_names,
):
    """Return 2-D element IDs whose Gmsh region is in ``region_names``."""
    region_names = set(
        region_names
        or []
    )

    return {
        int(
            element["id"]
        )
        for element in mesh_elements
        if (
            get_element_type(
                element["type"]
            )["dimension"]
            == 2
            and element[
                "material"
            ] in region_names
        )
    }

def _active_node_ids(
    mesh_elements,
    active_element_ids,
):
    """Return all nodes connected to at least one active domain element."""
    active_element_ids = {
        int(
            element_id
        )
        for element_id in active_element_ids
    }

    nodes = set()

    for element in mesh_elements:
        if int(
            element["id"]
        ) not in active_element_ids:
            continue

        nodes.update(
            int(
                node_id
            )
            for node_id in element[
                "connectivity"
            ]
        )

    return nodes

def _build_activation_reference_strains(
    nodes,
    mesh_elements,
    element_ids,
    u,
    kinematics_cache=None,
):
    """Capture B @ u at every GP of newly activated elements."""
    element_ids = {int(element_id) for element_id in element_ids}
    references = {}
    u = np.asarray(u, dtype=float)

    for element in mesh_elements:
        element_id = int(element["id"])
        if element_id not in element_ids:
            continue

        kinematics = get_element_kinematics(
            nodes=nodes,
            element=element,
            ndof_node=2,
            kinematics_cache=kinematics_cache,
        )
        if kinematics is None:
            continue

        ue = u[kinematics.dofs]
        references[element_id] = [
            (gp.B @ ue).copy()
            for gp in kinematics.gauss_points
        ]

    missing = element_ids - set(references.keys())
    if missing:
        raise ValueError(
            "Could not build activation references for element IDs "
            f"{sorted(missing)}."
        )

    return references

def _get_staged_dof_sets(
    nodes,
    mesh_elements,
    active_element_ids,
    physical_groups,
    boundaries,
    verbose=False,
):
    """
    Build physical and solver DOF sets for one construction stage.

    Nodes connected only to inactive elements have no stiffness. Their DOFs
    are therefore suppressed from the Newton system to avoid zero rows /
    singular matrices.

    These suppressed DOFs are NOT physical support reactions.
    """
    physical_fixed_dofs = get_fixed_dofs(
        physical_groups=physical_groups,
        boundaries=boundaries,
        verbose=verbose,
    )

    physical_fixed_set = set(
        int(
            dof
        )
        for dof in physical_fixed_dofs
    )

    active_nodes = _active_node_ids(
        mesh_elements=mesh_elements,
        active_element_ids=active_element_ids,
    )

    inactive_nodes = (
        set(
            range(
                len(
                    nodes
                )
            )
        )
        - active_nodes
    )

    inactive_dofs = set()

    for node_id in inactive_nodes:
        inactive_dofs.add(
            2
            * node_id
        )
        inactive_dofs.add(
            2
            * node_id
            + 1
        )

    solver_fixed_dofs = sorted(
        physical_fixed_set
        | inactive_dofs
    )

    all_dofs = np.arange(
        2
        * len(
            nodes
        ),
        dtype=int,
    )

    free_dofs = np.setdiff1d(
        all_dofs,
        np.asarray(
            solver_fixed_dofs,
            dtype=int,
        ),
        assume_unique=False,
    )

    return {
        "physical_fixed_dofs": sorted(
            physical_fixed_set
        ),
        "physical_fixed_dof_array": np.asarray(
            sorted(
                physical_fixed_set
            ),
            dtype=int,
        ),
        "inactive_dofs": sorted(
            inactive_dofs
        ),
        "solver_fixed_dofs": solver_fixed_dofs,
        "solver_fixed_dof_array": np.asarray(
            solver_fixed_dofs,
            dtype=int,
        ),
        "free_dofs": free_dofs,
        "active_node_ids": active_nodes,
        "inactive_node_ids": inactive_nodes,
    }

def _get_fixed_and_free_dofs(
    nodes,
    physical_groups,
    boundaries,
    verbose=True,
):
    """Return global constrained and free DOF sets."""
    fixed_dofs = get_fixed_dofs(
        physical_groups=physical_groups,
        boundaries=boundaries,
        verbose=verbose,
    )

    fixed_dof_array = np.asarray(
        fixed_dofs,
        dtype=int,
    )

    all_dofs = np.arange(
        2 * len(nodes),
        dtype=int,
    )

    free_dofs = np.setdiff1d(
        all_dofs,
        fixed_dof_array,
        assume_unique=False,
    )

    return (
        fixed_dofs,
        fixed_dof_array,
        free_dofs,
    )

def _relative_displacement_correction(
    delta_u,
    u_current,
    free_dofs,
):
    """
    Return a dimensionless Newton displacement-correction measure.

        ||delta_u|| / max(||u||, ||u + delta_u||, eps)

    Only free DOFs enter the norm.

    This formulation has two useful properties:

    1. The first physically meaningful correction is normally O(1), so it
       does not falsely satisfy a tiny displacement tolerance.
    2. As Newton converges, the required correction becomes small compared
       with the current displacement state.

    If both displacement states are essentially zero, machine epsilon protects
    the denominator.
    """
    delta_free = delta_u[
        free_dofs
    ]

    u_free = u_current[
        free_dofs
    ]

    corrected_u_free = (
        u_free
        + delta_free
    )

    correction_norm = np.linalg.norm(
        delta_free
    )

    reference_norm = max(
        np.linalg.norm(
            u_free
        ),
        np.linalg.norm(
            corrected_u_free
        ),
        np.finfo(float).eps,
    )

    return (
        correction_norm,
        correction_norm
        / reference_norm,
    )
