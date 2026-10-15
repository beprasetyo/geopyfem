import sys
import re
from datetime import datetime
from time import perf_counter

import numpy as np

from gmsh_parser import parse_gmsh
from problem_reader import read_problem
from mesh_checker import check_and_normalize_mesh
from material_model.materialtype import build_material_models
from solver import solve_problem
from output.vtk_writer import write_vtk_results
from output.run_logger import start_run_logger
from visualization.postprocess import (
    plot_displacement,
    print_element_gauss_results,
)


# =============================================================================
# COMMAND-LINE INPUT + AUTOMATIC RUN LOG
# =============================================================================

if len(sys.argv) != 2:
    print("Usage: python3 main.py problem.xml")
    raise SystemExit(1)

problem_file = sys.argv[1]

# Start the log BEFORE normal preprocessing so nearly all console output,
# warnings, solver traces, and any unhandled Python traceback are mirrored
# automatically into a timestamped .log file.
#
# By default the log is written to the same output directory used by the VTK
# section of the XML (normally <xml-directory>/results).
run_logger = start_run_logger(
    problem_file=problem_file,
)


# =============================================================================
# PROGRAM TIMING
# =============================================================================

# datetime gives a human-readable start/finish timestamp.
# perf_counter gives the monotonic elapsed time used for performance timing.
program_start_datetime = datetime.now()
program_start_timer = perf_counter()

print(
    "Program started:",
    program_start_datetime.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
)

if run_logger is not None:
    print(
        "Run log:",
        run_logger.path,
    )


# =============================================================================
# PRE-PROCESSING
# =============================================================================

print("read problem specification")

problem = read_problem(
    problem_file
)

print("read gmsh mesh")

(
    mesh_nodes,
    mesh_elements,
    physical_groups,
    physical_group_elements,
) = parse_gmsh(
    problem["mesh"]["file"],
    include_group_elements=True,
)


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------

num_nodes = len(
    mesh_nodes
)

nodes = np.zeros((
    num_nodes,
    2,
))

for node in mesh_nodes:
    nodes[
        node["id"],
        0,
    ] = node["x"]

    nodes[
        node["id"],
        1,
    ] = node["y"]


# -----------------------------------------------------------------------------
# Mesh checking / element orientation
# -----------------------------------------------------------------------------

mesh_report = check_and_normalize_mesh(
    nodes,
    mesh_elements,
)

print(
    "Reoriented clockwise elements:",
    mesh_report[
        "reoriented_elements"
    ],
)

print(
    "Minimum mesh det(J):",
    mesh_report[
        "minimum_detJ"
    ],
)

num_element = len(
    mesh_elements
)


# -----------------------------------------------------------------------------
# Material models
# -----------------------------------------------------------------------------

formulation = (
    problem["general"].get(
        "formulation"
    )
    or "plane_strain"
)

(
    material_models,
    materials_by_region,
) = build_material_models(
    problem["materials"],
    formulation=formulation,
)


mesh_material_regions = sorted({
    element["material"]
    for element in mesh_elements
})

for region in mesh_material_regions:
    if region == "UNASSIGNED":
        raise ValueError(
            "At least one 2-D element has no "
            "Gmsh Physical Surface assignment."
        )

    if region not in material_models:
        raise ValueError(
            f'Gmsh material region "{region}" '
            "is used by the mesh but is not "
            'defined by any <Material region="..."> '
            "in the XML."
        )


elements_per_material = {}

for element in mesh_elements:
    region = element["material"]

    elements_per_material[
        region
    ] = (
        elements_per_material.get(
            region,
            0,
        )
        + 1
    )


print(
    "Number of nodes:",
    num_nodes,
)

print(
    "Number of elements:",
    num_element,
)

print(
    "Element types:",
    sorted({
        element["type"]
        for element in mesh_elements
    }),
)

print(
    "Analysis:",
    problem["analysis"]["type"],
    problem["analysis"]["procedure"],
)

if problem.get("stages"):
    print(
        "Construction stages:",
        len(problem["stages"]),
    )

print(
    "Formulation:",
    formulation,
)

print(
    "Physical groups:",
    list(
        physical_groups.keys()
    ),
)

print("Materials:")

for region, count in (
    elements_per_material.items()
):
    material_spec = (
        materials_by_region[
            region
        ]
    )

    material_model = (
        material_models[
            region
        ]
    )

    print(
        f"  region={region}, "
        f'name={material_spec["name"]}, '
        f"model={material_model.model_name}, "
        f"elements={count}"
    )


# =============================================================================
# SOLVER
# =============================================================================

print("solve problem")

analysis_start_datetime = datetime.now()
analysis_start_timer = perf_counter()

solution = solve_problem(
    nodes=nodes,
    mesh_elements=mesh_elements,
    physical_groups=physical_groups,
    physical_group_elements=physical_group_elements,
    material_models=material_models,
    problem=problem,
)

analysis_end_timer = perf_counter()
analysis_end_datetime = datetime.now()
analysis_elapsed_time = (
    analysis_end_timer
    - analysis_start_timer
)

# Keep timing metadata in the result object so it can later be written to a
# log/report without changing the solver mechanics.
solution.setdefault(
    "timing",
    {},
)
solution["timing"].update({
    "analysis_started_at": analysis_start_datetime.isoformat(
        timespec="seconds"
    ),
    "analysis_finished_at": analysis_end_datetime.isoformat(
        timespec="seconds"
    ),
    "solver_elapsed_time": analysis_elapsed_time,
})

u = solution["u"]


# =============================================================================
# RESULT OUTPUT
# =============================================================================

output_settings = problem.get(
    "output",
    {},
)


# -----------------------------------------------------------------------------
# ParaView / VTU output
# -----------------------------------------------------------------------------

vtk_settings = output_settings.get(
    "vtk",
    {},
)

if vtk_settings.get(
    "enabled",
    False,
):
    stage_results = solution.get(
        "stage_results",
        [],
    )

    if stage_results:
        print(
            "write ParaView VTU results for each construction stage"
        )

        original_base_name = str(
            vtk_settings.get(
                "base_name",
                "result",
            )
        )

        for stage_result in stage_results:
            stage_id = stage_result["stage_id"]
            stage_name = str(
                stage_result.get("stage_name") or "stage"
            )

            safe_stage_name = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                stage_name.strip(),
            ).strip("_").lower()

            if not safe_stage_name:
                safe_stage_name = "stage"

            if isinstance(stage_id, int):
                stage_id_text = f"{stage_id:02d}"
            else:
                stage_id_text = str(stage_id)

            stage_vtk_settings = dict(vtk_settings)
            stage_vtk_settings["base_name"] = (
                f"{original_base_name}"
                f"_stage_{stage_id_text}"
                f"_{safe_stage_name}"
            )

            written_files = write_vtk_results(
                nodes=nodes,
                mesh_elements=mesh_elements,
                solution=stage_result,
                material_models=material_models,
                materials_by_region=materials_by_region,
                vtk_settings=stage_vtk_settings,
            )

            print(
                f"  stage {stage_id} mesh  :",
                written_files["mesh"],
            )

            if "gauss" in written_files:
                print(
                    f"  stage {stage_id} Gauss :",
                    written_files["gauss"],
                )

    else:
        print("write ParaView VTU results")

        written_files = write_vtk_results(
            nodes=nodes,
            mesh_elements=mesh_elements,
            solution=solution,
            material_models=material_models,
            materials_by_region=materials_by_region,
            vtk_settings=vtk_settings,
        )

        print(
            "  mesh results :",
            written_files["mesh"],
        )

        if "gauss" in written_files:
            print(
                "  Gauss points :",
                written_files["gauss"],
            )


# -----------------------------------------------------------------------------
# Optional quick Matplotlib displacement plot
# -----------------------------------------------------------------------------

matplotlib_settings = output_settings.get(
    "matplotlib",
    {},
)

if matplotlib_settings.get(
    "enabled",
    True,
):
    print(
        "plot quick displacement view"
    )

    plot_displacement(
        nodes,
        mesh_elements,
        u,
        component=output_settings.get(
            "displacement",
            "uy",
        ),
        deformation_scale=output_settings.get(
            "deformation_scale",
            1.0,
        ),
    )


# -----------------------------------------------------------------------------
# Optional console Gauss-point report
# -----------------------------------------------------------------------------

gp_settings = output_settings.get(
    "gauss_point_results",
    {},
)

if gp_settings.get(
    "enabled",
    False,
):
    element_id = gp_settings.get(
        "element_id"
    )

    if element_id is None:
        raise ValueError(
            '<GaussPointResults enabled="true"> '
            "requires <ElementID>."
        )

    print_element_gauss_results(
        nodes=nodes,
        mesh_elements=mesh_elements,
        u=u,
        material_models=material_models,
        element_id=element_id,
        gauss_point=gp_settings.get(
            "gauss_point",
            "all",
        ),
        units=problem.get(
            "general",
            {},
        ).get(
            "units",
            {},
        ),
    )

# =============================================================================
# ANALYSIS / PERFORMANCE SUMMARY
# =============================================================================

program_end_timer = perf_counter()
program_end_datetime = datetime.now()
program_elapsed_time = (
    program_end_timer
    - program_start_timer
)

solution["timing"].update({
    "program_started_at": program_start_datetime.isoformat(
        timespec="seconds"
    ),
    "program_finished_at": program_end_datetime.isoformat(
        timespec="seconds"
    ),
    "program_elapsed_time": program_elapsed_time,
})

print()
print("=" * 118)
print("GEOPYFEM ANALYSIS SUMMARY")
print("=" * 118)

stage_results = solution.get(
    "stage_results",
    [],
)

if stage_results:
    print(
        f"{'Stage':<32}"
        f"{'Active elem':>12}"
        f"{'Steps':>8}"
        f"{'Cutbacks':>10}"
        f"{'Newton':>10}"
        f"{'Residual evals':>16}"
        f"{'Time':>14}"
    )
    print("-" * 118)

    total_steps = 0
    total_cutbacks = 0
    total_newton = 0
    total_residual_evaluations = 0

    for stage_result in stage_results:
        stage_id = stage_result["stage_id"]

        stage_name = str(
            stage_result.get("stage_name")
            or ""
        )

        stage_label = (
            f"{stage_id}  {stage_name}"
        )

        if len(stage_label) > 31:
            stage_label = (
                stage_label[:28]
                + "..."
            )

        stage_active_elements = int(
            stage_result.get(
                "active_element_count",
                len(
                    stage_result.get(
                        "active_element_ids",
                        [],
                    )
                ),
            )
        )

        stage_steps = int(
            stage_result.get(
                "load_steps",
                0,
            )
        )

        stage_cutbacks = int(
            stage_result.get(
                "cutbacks",
                0,
            )
        )

        stage_newton = int(
            stage_result.get(
                "newton_iterations",
                stage_result.get(
                    "iterations",
                    0,
                ),
            )
        )

        stage_residual_evaluations = int(
            stage_result.get(
                "residual_evaluations",
                stage_newton,
            )
        )

        stage_time = float(
            stage_result.get(
                "elapsed_time",
                0.0,
            )
        )

        total_steps += stage_steps
        total_cutbacks += stage_cutbacks
        total_newton += stage_newton
        total_residual_evaluations += (
            stage_residual_evaluations
        )

        print(
            f"{stage_label:<32}"
            f"{stage_active_elements:>12d}"
            f"{stage_steps:>8d}"
            f"{stage_cutbacks:>10d}"
            f"{stage_newton:>10d}"
            f"{stage_residual_evaluations:>16d}"
            f"{stage_time:>11.3f} s"
        )

    print("-" * 118)

    safety_stage_results = [
        item
        for item in stage_results
        if item.get("procedure") == "safety"
    ]

    for safety_result in safety_stage_results:
        print()
        print(
            f"Safety stage {safety_result['stage_id']} status :",
            safety_result.get("safety_status"),
        )

        if safety_result.get("safety_factor") is not None:
            print(
                "Safety factor (SigmaMsf) :",
                f"{float(safety_result['safety_factor']):.6f}",
            )
        else:
            print(
                "Last converged SigmaMsf  :",
                f"{float(safety_result.get('final_strength_reduction_factor', 1.0)):.6f}",
            )

        peak = safety_result.get(
            "peak_strength_reduction_factor"
        )
        if peak is not None:
            print(
                "Peak SigmaMsf            :",
                f"{float(peak):.6f}",
            )

        plateau = safety_result.get(
            "plateau_strength_reduction_factor"
        )
        if plateau is not None:
            print(
                "Plateau mean SigmaMsf    :",
                f"{float(plateau):.6f}",
            )

        plateau_range = safety_result.get(
            "plateau_factor_range"
        )
        if plateau_range is not None:
            print(
                "Plateau SigmaMsf range   :",
                f"{float(plateau_range[0]):.6f} - "
                f"{float(plateau_range[1]):.6f}",
            )

        rejected = safety_result.get(
            "last_failed_strength_reduction_factor"
        )
        if rejected is not None:
            print(
                "Last rejected SigmaMsf   :",
                f"{float(rejected):.6f} (diagnostic only)",
            )

        safety_disp = safety_result.get(
            "safety_displacement_metrics"
        ) or {}
        if "maximum_safety_displacement" in safety_disp:
            print(
                "Max safety displacement  :",
                f"{float(safety_disp['maximum_safety_displacement']):.6e}",
            )

    if safety_stage_results:
        print()
        print("-" * 118)

    print(
        f"{'Total stages':<32}"
        f"{'':>12}"
        f"{total_steps:>8d}"
        f"{total_cutbacks:>10d}"
        f"{total_newton:>10d}"
        f"{total_residual_evaluations:>16d}"
        f"{analysis_elapsed_time:>11.3f} s"
    )

else:
    print(
        "Accepted load steps  :",
        solution.get(
            "load_steps",
            1,
        ),
    )

    print(
        "Cutbacks             :",
        solution.get(
            "cutbacks",
            0,
        ),
    )

    print(
        "Newton corrections   :",
        solution.get(
            "newton_iterations",
            solution.get(
                "iterations",
                0,
            ),
        ),
    )

    print(
        "Residual evaluations :",
        solution.get(
            "residual_evaluations",
            0,
        ),
    )

    if "load_stepping_type" in solution:
        print(
            "Load stepping        :",
            solution[
                "load_stepping_type"
            ],
        )

# =============================================================================
# DETAILED SOLVER PERFORMANCE PROFILE
# =============================================================================

if stage_results:
    profiled_stage_results = [
        item
        for item in stage_results
        if item.get("performance_profile")
    ]

    if profiled_stage_results:
        print()
        print("=" * 118)
        print("GEOPYFEM SOLVER PERFORMANCE PROFILE")
        print("=" * 118)
        print(
            f"{'Stage':<32}"
            f"{'Constitutive':>15}"
            f"{'Fint':>12}"
            f"{'Tangent':>12}"
            f"{'Lin solve':>12}"
            f"{'Other':>12}"
            f"{'Total':>12}"
        )
        print("-" * 118)

        aggregate_timings = {}
        aggregate_counts = {}
        aggregate_total = 0.0

        for stage_result in profiled_stage_results:
            profile = stage_result.get("performance_profile") or {}
            timings = profile.get("timings") or {}
            counts = profile.get("counts") or {}

            stage_id = stage_result.get("stage_id")
            stage_name = str(stage_result.get("stage_name") or "")
            stage_label = f"{stage_id}  {stage_name}"
            if len(stage_label) > 31:
                stage_label = stage_label[:28] + "..."

            constitutive_time = float(timings.get("constitutive_update", 0.0))
            internal_time = float(timings.get("internal_force_assembly", 0.0))
            tangent_time = float(timings.get("tangent_assembly", 0.0))
            linear_time = float(timings.get("linear_solve", 0.0))
            other_time = float(profile.get("other_time", 0.0))
            total_time = float(
                profile.get(
                    "total_elapsed_time",
                    stage_result.get("elapsed_time", 0.0),
                )
            )

            print(
                f"{stage_label:<32}"
                f"{constitutive_time:>12.3f} s"
                f"{internal_time:>9.3f} s"
                f"{tangent_time:>9.3f} s"
                f"{linear_time:>9.3f} s"
                f"{other_time:>9.3f} s"
                f"{total_time:>9.3f} s"
            )

            aggregate_total += total_time
            for key, value in timings.items():
                aggregate_timings[key] = (
                    aggregate_timings.get(key, 0.0)
                    + float(value)
                )
            for key, value in counts.items():
                aggregate_counts[key] = (
                    aggregate_counts.get(key, 0)
                    + int(value)
                )

        print("-" * 118)

        primary_keys = (
            "constitutive_update",
            "internal_force_assembly",
            "tangent_assembly",
            "linear_solve",
            "k0_initialization",
            "global_stiffness_assembly",
            "linear_state_update",
        )
        primary_total = sum(
            aggregate_timings.get(key, 0.0)
            for key in primary_keys
        )
        aggregate_other = max(aggregate_total - primary_total, 0.0)

        def _profile_pct(value):
            if aggregate_total <= 0.0:
                return 0.0
            return 100.0 * float(value) / aggregate_total

        print("Aggregate primary timing:")
        aggregate_rows = [
            ("Constitutive update", "constitutive_update"),
            ("Internal-force assembly", "internal_force_assembly"),
            ("Tangent assembly", "tangent_assembly"),
            ("Linear solve", "linear_solve"),
            ("K0 initialization", "k0_initialization"),
            ("Other stage work", None),
        ]
        for label, key in aggregate_rows:
            value = (
                aggregate_other
                if key is None
                else aggregate_timings.get(key, 0.0)
            )
            if value <= 0.0 and key == "k0_initialization":
                continue
            print(
                f"  {label:<27}: "
                f"{value:>9.3f} s  "
                f"({_profile_pct(value):5.1f} %)"
            )

        print("Nested/detail timing (already included above where applicable):")
        detail_rows = [
            ("material.integrate()", "constitutive_integration"),
            ("tangent element integration", "tangent_element_integration"),
            ("global matrix scatter", "matrix_scatter"),
            ("global matrix finalize", "matrix_finalize"),
            ("arc factor sensitivity", "arc_factor_sensitivity"),
            ("arc augmented-system build", "arc_augmented_system_build"),
        ]
        detail_printed = False
        for label, key in detail_rows:
            value = aggregate_timings.get(key, 0.0)
            if value > 0.0:
                detail_printed = True
                print(f"  {label:<27}: {value:>9.3f} s")
        if not detail_printed:
            print("  (no nested timing recorded)")

        cache_time = solution.get("kinematics_cache_build_time")
        if cache_time is not None:
            print(
                "One-time kinematics cache build :",
                f"{float(cache_time):.3f} s",
            )

        pattern_time = solution.get("sparse_pattern_build_time")
        pattern_nnz = solution.get("sparse_pattern_nnz")
        pattern_count = int(solution.get("sparse_pattern_build_count", 0) or 0)
        if pattern_time is not None and float(pattern_time) > 0.0:
            message = f"{float(pattern_time):.3f} s"
            details = []
            if pattern_count:
                details.append(f"patterns={pattern_count}")
            if pattern_nnz is not None:
                details.append(f"final nnz={int(pattern_nnz)}")
            if details:
                message += "  (" + ", ".join(details) + ")"
            print("Sparse CSR pattern build total :", message)

        print("Selected operation counts:")
        count_rows = [
            ("Constitutive integrations", "constitutive_calls"),
            ("Internal-force assemblies", "internal_force_assemblies"),
            ("Tangent assemblies", "tangent_assemblies"),
            ("Linear solves", "linear_solve_calls"),
            ("Matrix scatter calls", "matrix_scatter_calls"),
            ("Fixed-CSR scatter calls", "fixed_csr_scatter_calls"),
            ("Fixed-CSR finalizations", "fixed_csr_finalize_calls"),
            ("Dynamic-COO finalizations", "dynamic_coo_finalize_calls"),
            ("Arc sensitivity evaluations", "arc_factor_sensitivity_evaluations"),
        ]
        for label, key in count_rows:
            value = aggregate_counts.get(key, 0)
            if value:
                print(f"  {label:<27}: {value}")

        mc_total = aggregate_timings.get("mc_integrate_total", 0.0)
        if mc_total > 0.0:
            print()
            print(
                "Mohr-Coulomb fine-grained timing "
                "(nested; percentages use MC integrate total):"
            )
            print(
                "  note: fine-grained instrumentation adds runtime overhead; "
                "use this run for breakdown, not wall-clock benchmarking."
            )

            def _mc_pct(value):
                return 100.0 * float(value) / mc_total

            mc_rows = [
                ("MC integrate total", "mc_integrate_total"),
                ("state integration", "mc_state_integration"),
                ("  elastic predictor", "mc_elastic_predictor"),
                ("  analytic principal values", "mc_analytic_principal_values"),
                ("  fast elastic yield check", "mc_elastic_fast_yield_check"),
                ("  full principal decomposition", "mc_principal_decomposition"),
                ("  principal return", "mc_principal_return"),
                ("    yield/setup", "mc_yield_setup"),
                ("    face candidate", "mc_face_candidate"),
                (
                    "    compression-edge candidate",
                    "mc_compression_edge_candidate",
                ),
                (
                    "    extension-edge candidate",
                    "mc_extension_edge_candidate",
                ),
                ("    apex return", "mc_apex_return"),
                ("    active-set solves", "mc_active_set_linear_solve"),
                ("  stress reconstruction", "mc_stress_reconstruction"),
                ("  state copy/update", "mc_state_copy_update"),
                ("  plastic-strain update", "mc_plastic_strain_update"),
                ("  persistent state diagnostics", "mc_state_diagnostics"),
                ("tangent selection/build", "mc_tangent_total"),
                (
                    "  analytic spectral tangent",
                    "mc_analytic_spectral_tangent",
                ),
                ("  apex tangent", "mc_apex_tangent"),
                (
                    "  numerical tangent fallback",
                    "mc_numerical_tangent_fallback",
                ),
                ("response construction", "mc_response_construction"),
            ]
            for label, key in mc_rows:
                value = aggregate_timings.get(key, 0.0)
                if value > 0.0:
                    print(
                        f"  {label:<34}: "
                        f"{value:>9.3f} s  "
                        f"({_mc_pct(value):5.1f} %)"
                    )

            print("Mohr-Coulomb branch counts (top-level constitutive calls):")
            mc_count_rows = [
                ("MC constitutive calls", "mc_constitutive_calls"),
                ("MC local state integrations", "mc_state_integrations"),
                ("elastic responses", "mc_region_elastic"),
                ("elastic fast-path returns", "mc_elastic_fast_returns"),
                ("analytic principal-value checks", "mc_analytic_principal_value_evaluations"),
                ("fast-path plastic candidates", "mc_fast_path_plastic_candidates"),
                ("yield-boundary fallbacks", "mc_fast_path_yield_boundary_fallbacks"),
                ("non-plane-strain fallbacks", "mc_fast_path_non_plane_strain_fallbacks"),
                ("face returns", "mc_region_face"),
                ("compression-edge returns", "mc_region_compression_edge"),
                ("extension-edge returns", "mc_region_extension_edge"),
                ("Fortran apex returns", "mc_region_apex_fortran"),
                ("legacy apex projections", "mc_region_apex_projection"),
                ("numerical tangent fallbacks", "mc_numerical_tangent_fallbacks"),
                ("principal decompositions", "mc_principal_decompositions"),
                ("active-set linear solves", "mc_active_set_linear_solve_calls"),
            ]
            for label, key in mc_count_rows:
                value = aggregate_counts.get(key, 0)
                if value:
                    print(f"  {label:<34}: {value}")

            print("Mohr-Coulomb tangent-mode counts:")
            tangent_count_rows = [
                ("elastic", "mc_tangent_mode_elastic"),
                ("analytic spectral", "mc_tangent_mode_analytic_spectral"),
                (
                    "numerical repeated-eigen fallback",
                    "mc_tangent_mode_numerical_repeated_eigenvalue_fallback",
                ),
                (
                    "Fortran modified apex",
                    "mc_tangent_mode_fortran_modified_apex_single_singular",
                ),
                (
                    "legacy zero apex",
                    "mc_tangent_mode_apex_zero_legacy_projection",
                ),
            ]
            for label, key in tangent_count_rows:
                value = aggregate_counts.get(key, 0)
                if value:
                    print(f"  {label:<34}: {value}")

else:
    profile = solution.get("performance_profile") or {}
    if profile:
        timings = profile.get("timings") or {}
        total_time = float(
            profile.get("total_elapsed_time", analysis_elapsed_time)
        )
        primary_total = float(profile.get("primary_profiled_time", 0.0))
        other_time = float(
            profile.get("other_time", max(total_time - primary_total, 0.0))
        )

        print()
        print("=" * 118)
        print("GEOPYFEM SOLVER PERFORMANCE PROFILE")
        print("=" * 118)
        for label, key in (
            ("Global stiffness assembly", "global_stiffness_assembly"),
            ("Linear state update", "linear_state_update"),
            ("Constitutive update", "constitutive_update"),
            ("Internal-force assembly", "internal_force_assembly"),
            ("Tangent assembly", "tangent_assembly"),
            ("Linear solve", "linear_solve"),
        ):
            value = float(timings.get(key, 0.0))
            if value > 0.0:
                print(f"{label:<28}: {value:>9.3f} s")
        print(f"{'Other solver work':<28}: {other_time:>9.3f} s")
        print(f"{'Profiled solver total':<28}: {total_time:>9.3f} s")


print()
print(
    "Analysis started  :",
    analysis_start_datetime.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
)
print(
    "Analysis finished :",
    analysis_end_datetime.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
)
print(
    "Solver elapsed    :",
    f"{analysis_elapsed_time:.3f} s",
)
print(
    "Program finished  :",
    program_end_datetime.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
)
print(
    "Program elapsed   :",
    f"{program_elapsed_time:.3f} s",
)
print("=" * 118)


# Mark a normal successful end.  The logger itself is closed automatically
# through atexit, so an unhandled exception is also captured before shutdown.
if run_logger is not None:
    run_logger.mark_success()
