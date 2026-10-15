"""Staged-construction analysis orchestrator.

This module owns stage sequencing and delegates numerical solution to the K0,
ordinary nonlinear, and safety arc-length solver modules.
"""

from datetime import datetime
from time import perf_counter

import numpy as np

from assembly.boundary_condition import get_fixed_dofs
from assembly.load_vector import assemble_load_vector
from assembly.matrix_builder import SparseAssemblyPattern, normalize_matrix_storage
from elements.elemtype import get_element_type
from elements.kinematics_cache import ElementKinematicsCache
from state.state_manager import GaussPointStateManager
from stage.stage_manager import StageManager
from stage.activation import extend_activation_birth_fields
from stage.excavation import build_excavation_release_path
from stage.safety import validate_safety_stage_definition
from solvers.common import (
    _initialize_groundwater_if_present,
    _element_ids_for_regions,
    _active_node_ids,
    _build_activation_reference_strains,
    _get_staged_dof_sets,
)
from solvers.nonlinear import _solve_nonlinear_load_path
from solvers.safety_arc_length import _solve_strength_reduction_path
from solvers.k0_stage import _solve_k0_initial_stage
from solvers.profiler import SolverProfiler

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

    cache_start = perf_counter()
    kinematics_cache = ElementKinematicsCache(
        nodes=nodes,
        mesh_elements=mesh_elements,
    )
    kinematics_cache_build_time = perf_counter() - cache_start
    print(
        "initialize element kinematics cache:",
        f"elements={kinematics_cache.number_of_elements},",
        f"Gauss points={kinematics_cache.number_of_gauss_points}",
    )

    matrix_storage = normalize_matrix_storage(
        (solver_settings or {}).get("matrix_storage", "dense")
    )
    # Sparse topology is fixed only while the active construction domain is
    # unchanged.  Cache one CSR pattern per unique active-element set so each
    # Newton iteration updates data values only, while excavation/activation
    # stages still obtain the smallest valid sparse structure.
    sparse_pattern_cache = {}
    sparse_pattern_build_time = 0.0
    sparse_pattern_build_count = 0
    sparse_pattern = None

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

    for stage_index, stage_definition in enumerate(stages):
        requested_procedure = str(
            stage_definition.get("procedure")
            or default_procedure
        ).lower()

        if requested_procedure == "safety":
            validate_safety_stage_definition(stage_definition)

            if stage_index == 0:
                raise ValueError(
                    "Safety cannot be the first construction stage. "
                    "It must start from a previously converged physical state."
                )

            if stage_index != len(stages) - 1:
                raise ValueError(
                    "The current GeoPyFEM safety implementation requires "
                    "the safety stage to be the final stage. Its committed "
                    "states contain reduced-strength failure-path history."
                )

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
                kinematics_cache=kinematics_cache,
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
                    kinematics_cache=kinematics_cache,
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

        sparse_pattern = None
        if matrix_storage == "sparse":
            pattern_key = tuple(sorted(active_element_ids))
            sparse_pattern = sparse_pattern_cache.get(pattern_key)
            if sparse_pattern is None:
                pattern_start = perf_counter()
                sparse_pattern = SparseAssemblyPattern.from_kinematics_cache(
                    size=2 * len(nodes),
                    kinematics_cache=kinematics_cache,
                    active_element_ids=active_element_ids,
                )
                sparse_pattern_build_time += perf_counter() - pattern_start
                sparse_pattern_build_count += 1
                sparse_pattern_cache[pattern_key] = sparse_pattern
                print(
                    "initialize fixed sparse CSR assembly pattern:",
                    f"active elements={len(active_element_ids)},",
                    f"ndof={sparse_pattern.size},",
                    f"nnz={sparse_pattern.nnz}",
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
        if procedure not in ("nonlinear", "plastic", "k0", "safety"):
            raise NotImplementedError(
                "GeoPyFEM staged construction currently supports K0, "
                "nonlinear/plastic, and safety static stages. "
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
        stage_profiler = SolverProfiler()

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

        if procedure == "safety":
            load_change = f_external_target - f_external_current
            load_change_norm = np.linalg.norm(load_change[free_dofs])
            load_scale = max(
                np.linalg.norm(f_external_current[free_dofs]),
                np.linalg.norm(f_external_target[free_dofs]),
                1.0,
            )

            if load_change_norm > 1.0e-10 * load_scale:
                raise ValueError(
                    "Safety stage must keep the parent-stage physical load "
                    "vector unchanged. Complete all load/gravity changes in "
                    "an earlier stage."
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
                kinematics_cache=kinematics_cache,
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

        safety_u_start = None

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
                matrix_storage=matrix_storage,
                kinematics_cache=kinematics_cache,
                profiler=stage_profiler,
                sparse_pattern=sparse_pattern,
            )

        elif procedure == "safety":
            safety_u_start = u_current.copy()

            result = _solve_strength_reduction_path(
                nodes=nodes,
                mesh_elements=mesh_elements,
                material_models=material_models,
                state_manager=state_manager,
                fixed_dofs=fixed_dofs,
                fixed_dof_array=fixed_dof_array,
                free_dofs=free_dofs,
                u_start=u_current,
                f_external=f_external_target,
                solver_settings=solver_settings,
                strength_reduction_settings=context.get(
                    "strength_reduction"
                ),
                stage_id=stage_id,
                stage_name=stage_name,
                active_element_ids=active_element_ids,
                reaction_dof_array=physical_fixed_dof_array,
                kinematics_cache=kinematics_cache,
                profiler=stage_profiler,
                sparse_pattern=sparse_pattern,
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
                kinematics_cache=kinematics_cache,
                profiler=stage_profiler,
                sparse_pattern=sparse_pattern,
            )

        stage_end_timer = perf_counter()
        stage_end_datetime = datetime.now()
        stage_elapsed_time = (
            stage_end_timer
            - stage_start_timer
        )
        result["performance_profile"] = stage_profiler.snapshot(
            total_elapsed=stage_elapsed_time
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
        # ``u_current``. Ordinary construction output is measured from the
        # most recent user reset reference.  Safety output is intentionally
        # incremental from the START of the safety phase so the failure
        # mechanism is visible without the earlier construction displacement.
        reporting_reference = (
            safety_u_start.copy()
            if procedure == "safety" and safety_u_start is not None
            else displacement_reference.copy()
        )

        u_reported = (
            u_current
            - reporting_reference
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
            "displacement_reference": reporting_reference.copy(),
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
            "safety_status": result.get("safety_status"),
            "failure_mechanism_detected": bool(
                result.get("failure_mechanism_detected", False)
            ),
            "safety_factor": result.get("safety_factor"),
            "final_strength_reduction_factor": result.get(
                "final_strength_reduction_factor"
            ),
            "strength_reduction_factor_lower": result.get(
                "strength_reduction_factor_lower"
            ),
            "strength_reduction_factor_upper": result.get(
                "strength_reduction_factor_upper"
            ),
            "last_failed_strength_reduction_factor": result.get(
                "last_failed_strength_reduction_factor"
            ),
            "peak_strength_reduction_factor": result.get(
                "peak_strength_reduction_factor"
            ),
            "plateau_strength_reduction_factor": result.get(
                "plateau_strength_reduction_factor"
            ),
            "plateau_factor_range": result.get(
                "plateau_factor_range"
            ),
            "initial_arc_length_radius": result.get(
                "initial_arc_length_radius"
            ),
            "final_arc_length_radius": result.get(
                "final_arc_length_radius"
            ),
            "characteristic_length": result.get(
                "characteristic_length"
            ),
            "safety_displacement_scale": result.get(
                "safety_displacement_scale"
            ),
            "strength_reduction_settings": result.get(
                "strength_reduction_settings"
            ),
            "strength_reduction_materials": result.get(
                "strength_reduction_materials"
            ),
            "safety_history": list(
                result.get("safety_history", [])
            ),
            "safety_attempt_history": list(
                result.get("safety_attempt_history", [])
            ),
            "safety_displacement_metrics": result.get(
                "safety_displacement_metrics"
            ),
            "safety_gauss_summary": result.get(
                "safety_gauss_summary"
            ),
            "plateau_diagnostics": result.get(
                "plateau_diagnostics"
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
            "performance_profile": result.get(
                "performance_profile",
                {}
            ),
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

    if stage_results and stage_results[-1].get("procedure") == "safety":
        # Safety output is incremental from the start of the safety phase so
        # the collapse mechanism is not obscured by construction history.
        final_result["displacement_reference"] = stage_results[-1][
            "displacement_reference"
        ].copy()
        final_result["u"] = stage_results[-1]["u"].copy()
    else:
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
    final_result["kinematics_cache_build_time"] = (
        kinematics_cache_build_time
    )
    final_result["sparse_pattern_build_time"] = sparse_pattern_build_time
    final_result["sparse_pattern_build_count"] = sparse_pattern_build_count
    final_result["sparse_pattern_nnz"] = (
        None if sparse_pattern is None else sparse_pattern.nnz
    )
    final_result["sparse_pattern_element_maps"] = (
        0 if sparse_pattern is None else sparse_pattern.number_of_element_maps
    )
    return final_result
