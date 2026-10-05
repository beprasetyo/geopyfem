from datetime import datetime
from time import perf_counter

import numpy as np

from assembly.global_stiffness import assemble_global_stiffness
from assembly.load_vector import assemble_load_vector
from assembly.boundary_condition import (
    apply_displacement_boundary_conditions,
    get_fixed_dofs,
)
from assembly.internal_force import assemble_internal_force
from assembly.tangent_stiffness import assemble_tangent_stiffness
from elements.elemtype import get_element_type
from elements.kinematics import build_B_matrix
from state.state_manager import GaussPointStateManager
from stage.stage_manager import StageManager
from stage.activation import extend_activation_birth_fields
from stage.excavation import build_excavation_release_path
from hydraulic.groundwater import initialize_hydrostatic_pore_pressure
from initial_stress.k0 import initialize_k0_stress


def solve_problem(
    nodes,
    mesh_elements,
    physical_groups,
    physical_group_elements,
    material_models,
    problem,
):
    """
    Solve the FEM problem according to the analysis/procedure defined in XML.

    This function is the solver dispatcher. main.py does not need to know
    whether the requested analysis is linear, nonlinear, or time-dependent.

    Currently implemented
    ---------------------
    analysis type = "static"
    procedure     = "linear"

    Reserved for future implementation
    ----------------------------------
    analysis type = "static"
    procedure     = "plastic" / "nonlinear"

    analysis type = "time_dependent"

    Returns
    -------
    result : dict
        Solver result. At minimum:

            result["u"]

        For the current linear solver it also contains:

            result["K"]
            result["f"]
            result["fixed_dofs"]
            result["converged"]
            result["iterations"]
            result["analysis_type"]
            result["procedure"]
    """
    analysis = problem.get("analysis", {})
    solver_settings = problem.get("solver", {})

    stages = problem.get(
        "stages",
        [],
    )

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

    analysis_type = str(
        analysis.get("type", "static")
    ).lower()

    procedure = str(
        analysis.get("procedure", "linear")
    ).lower()

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

    if analysis_type == "static" and procedure in (
        "plastic",
        "nonlinear",
    ):
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

    linear_solver = str(
        solver_settings.get("linear_solver", "direct")
    ).lower()

    # ------------------------------------------------------------------
    # Gauss-point material-state infrastructure
    # ------------------------------------------------------------------
    #
    # Even the current linear analysis owns one state per integration point.
    # This makes the solver/result interface ready for K0 initial stress and
    # history-dependent nonlinear constitutive models without redesigning the
    # result structure later.
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
    )

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
        u = np.linalg.solve(K, f)

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
    }



def _element_displacement_vector(
    u,
    connectivity,
):
    """
    Extract one element displacement vector without depending on the
    visualization/post-processing package.
    """
    ue = np.zeros(
        2 * len(connectivity),
        dtype=float,
    )

    for a, node_id in enumerate(
        connectivity
    ):
        ue[2*a] = u[
            2*node_id
        ]

        ue[2*a + 1] = u[
            2*node_id + 1
        ]

    return ue


def _update_static_linear_gauss_states(
    nodes,
    mesh_elements,
    u,
    material_models,
    state_manager,
):
    """
    Populate TRIAL Gauss-point states from the solved static-linear field.

    This function is intentionally part of the CURRENT linear solver path.

    Future nonlinear solver
    -----------------------
    A nonlinear constitutive algorithm will update the trial states inside
    each Newton iteration instead of calling this linear reconstruction.

    At each Gauss point:

        strain_2d = B @ ue

        effective_stress = material.stress_tensor(strain_2d)

    then the full 3-D plane-strain state is stored in GaussPointState.
    """
    state_manager.reset_trial()

    for element in mesh_elements:
        element_info = get_element_type(
            element["type"]
        )

        if element_info["dimension"] != 2:
            continue

        connectivity = element[
            "connectivity"
        ]

        xIe = nodes[
            connectivity,
            :
        ]

        gradshape = element_info[
            "gradshape"
        ]

        gauss_rule = element_info[
            "gauss"
        ]

        gauss_points, _ = gauss_rule()

        ue = _element_displacement_vector(
            u,
            connectivity,
        )

        material_region = element[
            "material"
        ]

        if material_region not in material_models:
            raise ValueError(
                f'Element {element["id"]} uses material region '
                f'"{material_region}", but no material model is available.'
            )

        material = material_models[
            material_region
        ]

        if not hasattr(
            material,
            "stress_tensor",
        ):
            raise NotImplementedError(
                f'Material model "{getattr(material, "model_name", type(material).__name__)}" '
                "does not provide stress_tensor(), which is required to "
                "store the current static-linear Gauss-point state."
            )

        element_id = int(
            element["id"]
        )

        expected_gp = (
            state_manager.number_of_gauss_points(
                element_id
            )
        )

        if expected_gp != len(
            gauss_points
        ):
            raise ValueError(
                f"Gauss-point state count mismatch in element {element_id}."
            )

        for gp_index, q in enumerate(
            gauss_points
        ):
            B, _, _ = build_B_matrix(
                element_coordinates=xIe,
                gradshape=gradshape,
                natural_coordinates=q,
            )

            strain_2d = B @ ue

            stress_tensor = (
                material.stress_tensor(
                    strain_2d
                )
            )

            state = (
                state_manager.get_trial(
                    element_id,
                    gp_index,
                )
            )

            state.set_plane_strain(
                strain_2d
            )

            state.set_effective_stress_tensor(
                stress_tensor
            )

            # Explicit for clarity. These remain zero/false for
            # LinearElastic but already occupy their future nonlinear place.
            state.plastic_strain[:] = 0.0
            state.equivalent_plastic_strain = 0.0
            state.yielded = False


def _update_trial_gauss_states(
    nodes,
    mesh_elements,
    u,
    material_models,
    state_manager,
    active_element_ids=None,
):
    """
    Perform the constitutive update at every Gauss point for the CURRENT
    Newton displacement iterate.

    Important
    ---------
    The caller must first call:

        state_manager.reset_trial()

    Therefore every Newton iteration starts from the COMMITTED material state
    of the last converged load step.

    At one Gauss point:

        total_strain = B @ ue

        trial_state =
            material.update_state(
                total_strain,
                committed_state,
            )

    The constitutive model decides how EFFECTIVE stress/history evolve.
    Pore pressure remains a separate Gauss-point state variable; total stress
    is reconstructed only when global equilibrium/internal force is assembled.

    For LinearElastic the update is path-independent.

    For future Von Mises / MC / MCC, update_state() will perform the
    elastoplastic stress-integration algorithm using the committed history.
    """
    active_element_ids = (
        None
        if active_element_ids is None
        else {
            int(element_id)
            for element_id in active_element_ids
        }
    )

    for element in mesh_elements:
        element_id = int(
            element["id"]
        )

        if (
            active_element_ids is not None
            and element_id not in active_element_ids
        ):
            continue
        element_info = get_element_type(
            element["type"]
        )

        if element_info["dimension"] != 2:
            continue

        connectivity = element[
            "connectivity"
        ]

        element_coordinates = nodes[
            connectivity,
            :
        ]

        gradshape = element_info[
            "gradshape"
        ]

        gauss_rule = element_info[
            "gauss"
        ]

        gauss_points, _ = (
            gauss_rule()
        )

        ue = _element_displacement_vector(
            u,
            connectivity,
        )

        material_region = element[
            "material"
        ]

        if material_region not in material_models:
            raise ValueError(
                f'Element {element["id"]} uses material region '
                f'"{material_region}", but no material model is available.'
            )

        material = material_models[
            material_region
        ]

        if not hasattr(
            material,
            "update_state",
        ):
            raise NotImplementedError(
                f'Material model "{getattr(material, "model_name", type(material).__name__)}" '
                "does not implement update_state(), which is required by "
                "the static nonlinear solver."
            )

        for gp_index, q in enumerate(
            gauss_points
        ):
            B, _, _ = build_B_matrix(
                element_coordinates=element_coordinates,
                gradshape=gradshape,
                natural_coordinates=q,
            )

            kinematic_strain_2d = (
                B @ ue
            )

            activation_reference_2d = (
                state_manager.get_activation_strain_reference(
                    element_id,
                    gp_index,
                )
            )

            total_strain_2d = (
                kinematic_strain_2d
                - activation_reference_2d
            )

            committed_state = (
                state_manager.get_committed(
                    element_id,
                    gp_index,
                )
            )

            trial_state = (
                material.update_state(
                    total_strain_2d=total_strain_2d,
                    committed_state=committed_state,
                )
            )

            state_manager.set_trial(
                element_id,
                gp_index,
                trial_state,
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
):
    """
    Capture B @ u at every GP of newly activated elements.

    The constitutive strain used after activation becomes:

        epsilon_material = B @ u - epsilon_activation_reference

    so the element is stress-free at its birth configuration.
    """
    element_ids = {
        int(
            element_id
        )
        for element_id in element_ids
    }

    references = {}

    for element in mesh_elements:
        element_id = int(
            element["id"]
        )

        if element_id not in element_ids:
            continue

        element_info = get_element_type(
            element[
                "type"
            ]
        )

        if element_info[
            "dimension"
        ] != 2:
            continue

        connectivity = element[
            "connectivity"
        ]

        coordinates = nodes[
            connectivity,
            :
        ]

        ue = _element_displacement_vector(
            u,
            connectivity,
        )

        gradshape = element_info[
            "gradshape"
        ]

        gauss_points, _ = element_info[
            "gauss"
        ]()

        references[
            element_id
        ] = []

        for q in gauss_points:
            B, _, _ = build_B_matrix(
                element_coordinates=coordinates,
                gradshape=gradshape,
                natural_coordinates=q,
            )

            references[
                element_id
            ].append(
                (
                    B
                    @ ue
                ).copy()
            )

    missing = (
        element_ids
        - set(
            references.keys()
        )
    )

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
            )

            f_internal = assemble_internal_force(
                nodes=nodes,
                mesh_elements=mesh_elements,
                gauss_states=state_manager.trial,
                active_element_ids=active_element_ids,
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
                    material_models=material_models,
                    gauss_states=state_manager.trial,
                    active_element_ids=active_element_ids,
                )

                K_correction = (
                    K_tangent.copy()
                )

                r_correction = (
                    residual.copy()
                )

                for dof in fixed_dofs:
                    K_correction[
                        dof,
                        :,
                    ] = 0.0

                    K_correction[
                        dof,
                        dof,
                    ] = 1.0

                    r_correction[
                        dof
                    ] = 0.0

                try:
                    delta_u = np.linalg.solve(
                        K_correction,
                        r_correction,
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
        material_models=material_models,
        gauss_states=state_manager.committed,
        active_element_ids=active_element_ids,
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
        "final_load_factor": (
            current_load_factor
        ),
        "step_history": step_history,
        "attempt_history": attempt_history,

        "force_tolerance": force_tolerance,
        "displacement_tolerance": (
            displacement_tolerance
        ),

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
    )

    result["groundwater"] = groundwater
    result["groundwater_summary"] = groundwater_summary

    return result


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
    }


def solve_staged_static(
    nodes,
    mesh_elements,
    physical_groups,
    physical_group_elements,
    material_models,
    loads,
    boundaries,
    stages,
    groundwater=None,
    analysis=None,
    units=None,
    solver_settings=None,
):
    """
    Solve sequential static construction stages while preserving the INTERNAL
    total displacement and committed Gauss-point material history between
    stages.

    A stage may request ``ResetDisplacements``.  This does NOT erase the
    physical displacement/strain history used by the constitutive model.
    Instead GeoPyFEM moves the reporting reference so the next stage starts
    from zero reported displacement:

        u_reported = u_total - u_reference

    This is the appropriate staged-construction behavior: stress, strain,
    plastic history, pore pressure, and hardening remain committed while the
    displacement output can be reset.

    Current infrastructure supports cumulative gravity, named boundary-load
    activation/deactivation, domain-region activation/deactivation, K0
    initialization, stress-free element birth references, excavation
    holding-force release, inactive-node DOF suppression, and
    displacement-reference reset.
    """
    analysis = analysis or {}

    if not stages:
        raise ValueError(
            "solve_staged_static() requires at least one stage."
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

    # Print/validate the physical displacement boundaries once. Additional
    # inactive-node suppression DOFs are rebuilt separately for each stage.
    get_fixed_dofs(
        physical_groups=physical_groups,
        boundaries=boundaries,
        verbose=True,
    )

    mesh_region_names = list(
        dict.fromkeys(
            element["material"]
            for element in mesh_elements
            if get_element_type(
                element["type"]
            )["dimension"] == 2
        )
    )

    construction = StageManager(
        load_definitions=loads,
        region_names=mesh_region_names,
        initial_gravity=analysis.get("gravity", {}),
    )

    # ``u_current`` is the INTERNAL total displacement used by the
    # constitutive update. It is never artificially zeroed because
    # Gauss-point strain/stress history must remain mechanically consistent.
    u_current = np.zeros(2 * len(nodes), dtype=float)

    # ``displacement_reference`` controls only the displacement reported to
    # the user/output after <ResetDisplacements>true</ResetDisplacements>.
    #
    # reported displacement = u_current - displacement_reference
    #
    # This preserves the full constitutive strain history while allowing a
    # construction phase to redefine zero displacement for subsequent stages.
    displacement_reference = np.zeros(
        2 * len(nodes),
        dtype=float,
    )

    f_external_current = np.zeros(2 * len(nodes), dtype=float)
    stage_results = []

    default_load_steps = analysis.get("load_steps", 1)
    default_procedure = str(
        analysis.get("procedure", "nonlinear")
    ).lower()

    for stage_definition in stages:
        context = construction.apply_stage(stage_definition)
        stage_id = context["stage_id"]
        stage_name = context["stage_name"]

        # --------------------------------------------------------------
        # Construction-region birth/death
        # --------------------------------------------------------------
        deactivated_element_ids = _element_ids_for_regions(
            mesh_elements=mesh_elements,
            region_names=context[
                "deactivated_regions"
            ],
        )

        if deactivated_element_ids:
            state_manager.deactivate_elements(
                deactivated_element_ids
            )

        activated_element_ids = _element_ids_for_regions(
            mesh_elements=mesh_elements,
            region_names=context[
                "activated_regions"
            ],
        )

        activation_birth = {
            "region_node_ids": [],
            "anchor_node_ids": [],
            "birth_node_ids": [],
            "maximum_birth_displacement": 0.0,
        }

        if activated_element_ids:
            # ----------------------------------------------------------
            # Cumulative geometric birth displacement
            # ----------------------------------------------------------
            #
            # Nodes that belonged only to inactive elements were suppressed
            # in previous stages, so they do not yet carry the accumulated
            # settlement/deformation of the existing domain.  Before the new
            # elements are born, smoothly extend the displacement of shared
            # old-active interface nodes into those newly active nodes.
            #
            # The reporting displacement reference is extended at the same
            # time so ResetDisplacements remains consistent for later-born
            # material.
            previously_active_node_ids = _active_node_ids(
                mesh_elements=mesh_elements,
                active_element_ids=state_manager.active_element_ids,
            )

            activation_birth = extend_activation_birth_fields(
                nodes=nodes,
                mesh_elements=mesh_elements,
                activated_element_ids=activated_element_ids,
                previously_active_node_ids=previously_active_node_ids,
                displacement=u_current,
                displacement_reference=displacement_reference,
            )

            u_current = activation_birth[
                "displacement"
            ]

            displacement_reference = activation_birth[
                "displacement_reference"
            ]

            # Build the constitutive reference only AFTER the cumulative
            # birth displacement has been assigned. Therefore:
            #
            #     epsilon_material = B @ u - epsilon_ref = 0
            #
            # at the instant of activation, while U remains cumulative and
            # geometrically continuous across construction stages.
            activation_references = (
                _build_activation_reference_strains(
                    nodes=nodes,
                    mesh_elements=mesh_elements,
                    element_ids=activated_element_ids,
                    u=u_current,
                )
            )

            state_manager.activate_elements(
                activation_references
            )

        active_element_ids = _element_ids_for_regions(
            mesh_elements=mesh_elements,
            region_names=context[
                "active_region_names"
            ],
        )

        if (
            active_element_ids
            != state_manager.active_element_ids
        ):
            raise RuntimeError(
                "StageManager region activity and GaussPointStateManager "
                "element activity are inconsistent."
            )

        dof_sets = _get_staged_dof_sets(
            nodes=nodes,
            mesh_elements=mesh_elements,
            active_element_ids=active_element_ids,
            physical_groups=physical_groups,
            boundaries=boundaries,
            verbose=False,
        )

        fixed_dofs = dof_sets[
            "solver_fixed_dofs"
        ]
        fixed_dof_array = dof_sets[
            "solver_fixed_dof_array"
        ]
        free_dofs = dof_sets[
            "free_dofs"
        ]
        physical_fixed_dof_array = dof_sets[
            "physical_fixed_dof_array"
        ]

        procedure = str(
            context.get("procedure") or default_procedure
        ).lower()
        if procedure not in ("nonlinear", "plastic", "k0"):
            raise NotImplementedError(
                "GeoPyFEM staged construction currently supports "
                "K0 and nonlinear/plastic static stages. "
                f'Stage {stage_id} requested procedure="{procedure}".'
            )

        load_steps = context.get("load_steps")
        if load_steps is None:
            load_steps = (
                default_load_steps
                if default_load_steps is not None
                else 1
            )

        # Wall-clock timestamp is useful for logs; perf_counter() is used
        # for the actual elapsed-time measurement.
        stage_start_datetime = datetime.now()
        stage_start_timer = perf_counter()

        print()
        print("=" * 72)
        print(f"STAGE {stage_id}: {stage_name}")
        print("=" * 72)
        print(
            "active boundary loads:",
            context["active_load_names"] or [],
        )
        print("gravity:", context["gravity"])
        print(
            "active regions:",
            context[
                "active_region_names"
            ],
        )
        print(
            "activated this stage:",
            context[
                "activated_regions"
            ],
        )
        print(
            "deactivated this stage:",
            context[
                "deactivated_regions"
            ],
        )

        if activated_element_ids:
            print(
                "activation interface anchor nodes:",
                len(
                    activation_birth[
                        "anchor_node_ids"
                    ]
                ),
            )
            print(
                "new nodes given cumulative birth displacement:",
                len(
                    activation_birth[
                        "birth_node_ids"
                    ]
                ),
            )
            print(
                "maximum inherited birth displacement:",
                f"{activation_birth['maximum_birth_displacement']:.6e}",
            )

        print(
            "active domain elements:",
            len(
                active_element_ids
            ),
            "/",
            len(
                mesh_elements
            ),
        )
        print(
            "inactive-only nodes suppressed:",
            len(
                dof_sets[
                    "inactive_node_ids"
                ]
            ),
        )

        # Assemble the target PHYSICAL load configuration for this stage.
        # For excavation this is the final state after the temporary holding
        # force has been fully released.
        f_external_target = assemble_load_vector(
            nodes=nodes,
            mesh_elements=mesh_elements,
            physical_group_elements=physical_group_elements,
            material_models=material_models,
            loads=context["active_loads"],
            gravity=context["gravity"],
            units=units,
            active_element_ids=active_element_ids,
        )

        excavation_release = None
        f_external_stage_start = f_external_current
        load_path_kind = "load_change"

        # --------------------------------------------------------------
        # Excavation stress release
        # --------------------------------------------------------------
        # A deactivation in a stage AFTER a previously converged mechanical
        # state represents excavation/removal.  The removed elements have
        # already left the active domain, so using the old external vector
        # directly would create an instantaneous residual jump.
        #
        # Instead, build a temporary holding force from the committed
        # remaining-domain stress state.  The ordinary lambda=0->1 load
        # advancement then releases that holding force smoothly to zero.
        #
        # Deactivation in the FIRST stage is intentionally excluded: that is
        # how GeoPyFEM defines regions that simply do not exist initially,
        # e.g. future embankment lifts before a K0 stage.
        if (
            deactivated_element_ids
            and stage_results
            and procedure != "k0"
        ):
            excavation_release = build_excavation_release_path(
                nodes=nodes,
                mesh_elements=mesh_elements,
                state_manager=state_manager,
                active_element_ids=active_element_ids,
                f_physical_target=f_external_target,
                free_dofs=free_dofs,
            )

            f_external_stage_start = excavation_release[
                "f_external_start"
            ]
            load_path_kind = "excavation_release"

            print(
                "excavation holding-force norm (free DOFs):",
                f"{excavation_release['holding_force_norm']:.6e}",
            )
            print(
                "excavation lambda=0 free residual norm:",
                f"{excavation_release['initial_free_residual_norm']:.6e}",
            )

        if procedure == "k0":
            if stage_results:
                raise ValueError(
                    "The current K0 procedure is an initial-stress procedure "
                    "and must be the first construction stage."
                )

            if np.linalg.norm(u_current) > 1.0e-14:
                raise ValueError(
                    "K0 initial stress requires zero pre-existing displacement."
                )

            result = _solve_k0_initial_stage(
                nodes=nodes,
                mesh_elements=mesh_elements,
                physical_group_elements=physical_group_elements,
                material_models=material_models,
                state_manager=state_manager,
                fixed_dofs=fixed_dofs,
                fixed_dof_array=fixed_dof_array,
                free_dofs=free_dofs,
                active_loads=context["active_loads"],
                gravity=context["gravity"],
                k0_specification=context.get("k0"),
                f_external_target=f_external_target,
                units=units,
                active_element_ids=active_element_ids,
                reaction_dof_array=physical_fixed_dof_array,
            )

        else:
            result = _solve_nonlinear_load_path(
                nodes=nodes,
                mesh_elements=mesh_elements,
                material_models=material_models,
                state_manager=state_manager,
                fixed_dofs=fixed_dofs,
                fixed_dof_array=fixed_dof_array,
                free_dofs=free_dofs,
                u_start=u_current,
                f_external_start=f_external_stage_start,
                f_external_target=f_external_target,
                load_steps=load_steps,
                solver_settings=solver_settings,
                stage_id=stage_id,
                stage_name=stage_name,
                active_element_ids=active_element_ids,
                reaction_dof_array=physical_fixed_dof_array,
                load_path_kind=load_path_kind,
            )

        stage_end_timer = perf_counter()
        stage_end_datetime = datetime.now()
        stage_elapsed_time = (
            stage_end_timer
            - stage_start_timer
        )

        u_current = result["u"].copy()
        f_external_current = result["f_external"].copy()

        reset_displacements = bool(
            context.get(
                "reset_displacements",
                False,
            )
        )

        # The constitutive solver keeps the physical accumulated displacement
        # ``u_current``. Output displacement is measured from the most recent
        # reset reference.
        u_reported = (
            u_current
            - displacement_reference
        )

        displacement_reference_after_stage = (
            u_current.copy()
            if reset_displacements
            else displacement_reference.copy()
        )

        # Freeze a deep state snapshot for stage-specific VTU output BEFORE
        # changing the displacement reference. Therefore the stage that asks
        # for a reset still shows the displacement that developed during that
        # stage, while the NEXT stage starts its reported displacement at zero.
        stage_snapshot = {
            "u": u_reported.copy(),
            "u_total": u_current.copy(),
            "displacement_reference": displacement_reference.copy(),
            "displacement_reference_after_stage": (
                displacement_reference_after_stage.copy()
            ),
            "reset_displacements": reset_displacements,
            "f_external": result["f_external"].copy(),
            "f_internal": result["f_internal"].copy(),
            "residual": result["residual"].copy(),
            "reaction": result["reaction"].copy(),
            "free_residual_norm": result["free_residual_norm"],
            # User-facing fixed_dofs contains only physical supports.
            # Inactive-only DOFs are recorded separately.
            "fixed_dofs": list(
                dof_sets[
                    "physical_fixed_dofs"
                ]
            ),
            "inactive_dofs": list(
                dof_sets[
                    "inactive_dofs"
                ]
            ),
            "active_node_ids": sorted(
                dof_sets[
                    "active_node_ids"
                ]
            ),
            "inactive_node_ids": sorted(
                dof_sets[
                    "inactive_node_ids"
                ]
            ),
            "active_element_ids": sorted(
                active_element_ids
            ),
            "inactive_element_ids": sorted(
                state_manager.inactive_element_ids
            ),
            "active_region_names": list(
                context[
                    "active_region_names"
                ]
            ),
            "activated_regions": list(
                context[
                    "activated_regions"
                ]
            ),
            "deactivated_regions": list(
                context[
                    "deactivated_regions"
                ]
            ),
            "deactivated_element_ids": sorted(
                deactivated_element_ids
            ),
            "load_path_kind": result.get(
                "load_path_kind",
                "direct" if procedure == "k0" else load_path_kind,
            ),
            "excavation_release": excavation_release is not None,
            "excavation_holding_force": (
                None
                if excavation_release is None
                else excavation_release[
                    "holding_force"
                ].copy()
            ),
            "excavation_holding_force_norm": (
                0.0
                if excavation_release is None
                else excavation_release[
                    "holding_force_norm"
                ]
            ),
            "excavation_initial_free_residual_norm": (
                0.0
                if excavation_release is None
                else excavation_release[
                    "initial_free_residual_norm"
                ]
            ),
            "activation_anchor_node_ids": list(
                activation_birth[
                    "anchor_node_ids"
                ]
            ),
            "activation_birth_node_ids": list(
                activation_birth[
                    "birth_node_ids"
                ]
            ),
            "maximum_activation_birth_displacement": float(
                activation_birth[
                    "maximum_birth_displacement"
                ]
            ),
            "active_element_count": len(
                active_element_ids
            ),
            "inactive_element_count": (
                len(
                    mesh_elements
                )
                - len(
                    active_element_ids
                )
            ),
            "gauss_states": state_manager.snapshot_committed(),
            "activation_strain_reference": (
                state_manager.snapshot_activation_references()
            ),
            "converged": True,
            "iterations": result["iterations"],
            "newton_iterations": result.get(
                "newton_iterations",
                result["iterations"],
            ),
            "residual_evaluations": result.get(
                "residual_evaluations",
                result["iterations"],
            ),
            "load_steps": result["load_steps"],
            "requested_load_steps": result.get(
                "requested_load_steps",
                result["load_steps"],
            ),
            "cutbacks": result.get(
                "cutbacks",
                0,
            ),
            "maximum_cutbacks": result.get(
                "maximum_cutbacks",
                0,
            ),
            "load_stepping_type": result.get(
                "load_stepping_type",
                "direct" if procedure == "k0" else "fixed",
            ),
            "final_load_factor": result.get(
                "final_load_factor",
                1.0,
            ),
            "force_tolerance": result.get(
                "force_tolerance",
            ),
            "displacement_tolerance": result.get(
                "displacement_tolerance",
            ),
            "step_history": list(result["step_history"]),
            "attempt_history": list(
                result.get(
                    "attempt_history",
                    [],
                )
            ),
            "analysis_type": "static",
            "procedure": procedure,
            "stage_id": stage_id,
            "stage_name": stage_name,
            "active_load_names": list(context["active_load_names"]),
            "gravity": dict(context["gravity"]),
            "groundwater": groundwater,
            "groundwater_summary": groundwater_summary,
            "k0_summary": result.get("k0_summary"),
            "started_at": stage_start_datetime.isoformat(
                timespec="seconds"
            ),
            "finished_at": stage_end_datetime.isoformat(
                timespec="seconds"
            ),
            "elapsed_time": stage_elapsed_time,
        }
        stage_results.append(stage_snapshot)

        if reset_displacements:
            displacement_reference = (
                u_current.copy()
            )

            print(
                f"stage {stage_id}: "
                "reset reported displacements to zero "
                "for subsequent stages"
            )

        print(
            f"stage {stage_id} converged: "
            f"|r_free|={result['free_residual_norm']:.6e}"
        )
        print(
            f"stage {stage_id} elapsed time: "
            f"{stage_elapsed_time:.3f} s"
        )

    final_result = dict(result)

    # Public/output displacement follows the current reset reference.
    # ``u_total`` remains available for diagnostics and future advanced
    # construction features.
    final_result["u_total"] = u_current.copy()
    final_result["displacement_reference"] = (
        displacement_reference.copy()
    )
    final_result["u"] = (
        u_current
        - displacement_reference
    )

    final_result["procedure"] = "staged"
    final_result["stage_results"] = stage_results
    final_result["number_of_stages"] = len(stage_results)
    final_result["groundwater"] = groundwater
    final_result["groundwater_summary"] = groundwater_summary
    final_result["stage_elapsed_time"] = sum(
        stage_result["elapsed_time"]
        for stage_result in stage_results
    )
    return final_result

def solve_time_dependent(
    nodes,
    mesh_elements,
    physical_groups,
    physical_group_elements,
    material_models,
    loads,
    boundaries,
    analysis,
    solver_settings=None,
):
    """
    Reserved interface for future time-dependent analysis.

    Possible future uses include:
        - consolidation
        - transient groundwater flow
        - coupled hydro-mechanical analysis
        - rainfall infiltration
        - BBM with suction evolution

    The main.py interface will remain the same: it calls solve_problem(), and
    solver.py decides which time-stepping algorithm is required.
    """
    time_settings = analysis.get("time", {})

    start = time_settings.get("start")
    end = time_settings.get("end")
    time_step = time_settings.get("time_step")

    raise NotImplementedError(
        "Time-dependent solver is not implemented yet. "
        f"Requested time settings: start={start}, end={end}, "
        f"time_step={time_step}."
    )
