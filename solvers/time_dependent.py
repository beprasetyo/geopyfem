"""Reserved time-dependent solver interface.

The interface is separated now so future consolidation / coupled hydraulic
analysis can be added without growing the top-level solver facade.
"""

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
