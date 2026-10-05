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
from visualization.postprocess import (
    plot_displacement,
    print_element_gauss_results,
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


# =============================================================================
# PRE-PROCESSING
# =============================================================================

if len(sys.argv) != 2:
    print("Usage: python3 main.py problem.xml")
    raise SystemExit(1)

problem_file = sys.argv[1]

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
