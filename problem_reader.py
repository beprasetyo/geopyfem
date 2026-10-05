import xml.etree.ElementTree as ET
from pathlib import Path


def read_problem(filename):
    """
    Read the XML problem specification and return a Python dictionary.

    This module only reads/organizes input data.
    It does not perform FEM calculations.
    """
    filename = Path(filename)
    tree = ET.parse(filename)
    root = tree.getroot()

    problem = {
        "name": root.get("name"),
        "version": root.get("version"),
        "general": _read_general(root),
        "mesh": _read_mesh(
            root,
            filename.parent,
        ),
        "materials": _read_materials(root),
        "groundwater": _read_groundwater(root),
        "boundaries": _read_boundaries(root),
        "loads": _read_loads(root),
        "analysis": _read_analysis(root),
        "solver": _read_solver(root),
        "stages": _read_stages(root),
        "output": _read_output(
            root,
            filename.parent,
            root.get("name"),
        ),
    }

    return problem


def _read_general(root):
    general = root.find("General")

    if general is None:
        return {}

    units = general.find("Units")

    return {
        "dimension": _text(
            general,
            "Dimension",
        ),
        "formulation": _text(
            general,
            "Formulation",
        ),
        "units": {
            "length": (
                _text(units, "Length")
                if units is not None
                else None
            ),
            "force": (
                _text(units, "Force")
                if units is not None
                else None
            ),
            "time": (
                _text(units, "Time")
                if units is not None
                else None
            ),
        },
    }


def _read_mesh(
    root,
    xml_directory,
):
    mesh = root.find("Mesh")

    if mesh is None:
        raise ValueError(
            "XML does not contain a <Mesh> section."
        )

    mesh_name = _text(
        mesh,
        "File",
    )

    mesh_format = _text(
        mesh,
        "Format",
    )

    if mesh_name is None:
        raise ValueError(
            "<Mesh> must contain <File>."
        )

    mesh_path = Path(
        mesh_name
    )

    if not mesh_path.is_absolute():
        mesh_path = (
            xml_directory
            / mesh_path
        )

    return {
        "file": str(mesh_path),
        "format": mesh_format,
    }


def _read_materials(root):
    section = root.find(
        "Materials"
    )

    if section is None:
        return []

    materials = []

    for material in section.findall(
        "Material"
    ):
        parameters = {}

        parameters_xml = material.find(
            "Parameters"
        )

        if parameters_xml is not None:
            for parameter in parameters_xml:
                parameters[
                    parameter.tag
                ] = _convert_number(
                    parameter.text
                )

        materials.append({
            "id": (
                int(material.get("id"))
                if material.get("id") is not None
                else None
            ),
            "name": material.get("name"),
            "region": material.get("region"),
            "model": material.get("model"),
            "parameters": parameters,
        })

    return materials


def _read_groundwater(root):
    """
    Read one static phreatic surface.

    Horizontal example
    ------------------
    <Groundwater enabled="true" name="initial_gwt" type="horizontal">
      <UnitWeightWater>9.81</UnitWeightWater>
      <Elevation>1.5</Elevation>
    </Groundwater>

    Polyline example
    ----------------
    <Groundwater enabled="true" name="initial_gwt" type="polyline">
      <UnitWeightWater>9.81</UnitWeightWater>
      <Point x="0.0" y="1.5"/>
      <Point x="1.0" y="1.2"/>
    </Groundwater>

    UnitWeightWater must use force/length^3 units consistent with the model.
    """
    section = root.find(
        "Groundwater"
    )

    if section is None:
        return None

    enabled = (
        section.get(
            "enabled",
            "true",
        ).lower()
        == "true"
    )

    surface_type = str(
        section.get(
            "type",
            "horizontal",
        )
    ).lower()

    specification = {
        "enabled": enabled,
        "name": (
            section.get("name")
            or "groundwater"
        ),
        "type": surface_type,
        "unit_weight_water": _convert_number(
            _text(
                section,
                "UnitWeightWater",
            )
        ),
    }

    if surface_type == "horizontal":
        specification[
            "elevation"
        ] = _convert_number(
            _text(
                section,
                "Elevation",
            )
        )

    elif surface_type == "polyline":
        points = []

        for point in section.findall(
            "Point"
        ):
            x = point.get("x")
            y = point.get("y")

            if x is None or y is None:
                raise ValueError(
                    "Each groundwater <Point> requires x and y attributes."
                )

            points.append((
                float(x),
                float(y),
            ))

        specification[
            "points"
        ] = points

    else:
        raise NotImplementedError(
            f'Unsupported Groundwater type "{surface_type}". '
            'Supported types: "horizontal", "polyline".'
        )

    return specification


def _read_boundaries(root):
    section = root.find(
        "BoundaryConditions"
    )

    if section is None:
        return []

    boundaries = []

    for bc in section.findall(
        "DisplacementBoundary"
    ):
        boundaries.append({
            "name": bc.get("name"),
            "group": bc.get("group"),
            "ux": _text(bc, "ux"),
            "uy": _text(bc, "uy"),
        })

    return boundaries


def _read_loads(root):
    section = root.find(
        "Loads"
    )

    if section is None:
        return []

    loads = []

    for load in section.findall(
        "BoundaryLoad"
    ):
        time_function = load.find(
            "TimeFunction"
        )

        loads.append({
            "name": load.get("name"),
            "group": load.get("group"),
            "type": load.get("type"),
            "tx": _convert_number(
                _text(load, "tx")
            ),
            "ty": _convert_number(
                _text(load, "ty")
            ),
            "time_function": (
                time_function.get("type")
                if time_function is not None
                else None
            ),
        })

    return loads


def _read_analysis(root):
    section = root.find(
        "Analysis"
    )

    if section is None:
        return {}

    gravity = section.find(
        "Gravity"
    )

    time = section.find(
        "Time"
    )

    analysis = {
        "type": _text(
            section,
            "Type",
        ),
        "procedure": _text(
            section,
            "Procedure",
        ),

        # Static nonlinear load stepping.
        # Defaults are handled by solver.py when this tag is absent.
        "load_steps": _convert_number(
            _text(
                section,
                "LoadSteps",
            )
        ),
    }

    if gravity is not None:
        analysis["gravity"] = {
            "enabled": (
                gravity.get(
                    "enabled",
                    "false",
                ).lower()
                == "true"
            ),
            "gx": _convert_number(
                _text(gravity, "gx")
            ),
            "gy": _convert_number(
                _text(gravity, "gy")
            ),
        }

    if time is not None:
        analysis["time"] = {
            "start": _convert_number(
                _text(time, "Start")
            ),
            "end": _convert_number(
                _text(time, "End")
            ),
            "time_step": _convert_number(
                _text(time, "TimeStep")
            ),
        }

    return analysis


def _read_stages(root):
    """Read optional staged-construction definitions."""
    section = root.find("Stages")
    if section is None:
        return []

    stages = []
    for index, stage in enumerate(section.findall("Stage")):
        stage_id_text = stage.get("id")
        if stage_id_text is None:
            stage_id = index
        else:
            try:
                stage_id = int(stage_id_text)
            except ValueError:
                stage_id = stage_id_text

        load_steps = _convert_number(_text(stage, "LoadSteps"))

        gravity_xml = stage.find("Gravity")
        gravity = None
        if gravity_xml is not None:
            enabled_text = gravity_xml.get("enabled")
            gravity = {
                "enabled": (
                    None if enabled_text is None
                    else enabled_text.lower() == "true"
                ),
                "gx": _convert_number(_text(gravity_xml, "gx")),
                "gy": _convert_number(_text(gravity_xml, "gy")),
            }

        reset_displacements_text = _text(
            stage,
            "ResetDisplacements",
        )

        if reset_displacements_text is None:
            reset_displacements = False
        else:
            normalized_reset = (
                reset_displacements_text
                .strip()
                .lower()
            )

            if normalized_reset in (
                "true",
                "1",
                "yes",
                "on",
            ):
                reset_displacements = True

            elif normalized_reset in (
                "false",
                "0",
                "no",
                "off",
            ):
                reset_displacements = False

            else:
                raise ValueError(
                    "<ResetDisplacements> must be true or false "
                    f'in stage "{stage.get("name") or stage_id}".'
                )

        stages.append({
            "id": stage_id,
            "name": stage.get("name") or f"Stage {stage_id}",
            "procedure": _text(stage, "Procedure"),
            "load_steps": None if load_steps is None else int(load_steps),
            "activate_loads": _read_stage_load_names(stage, "ActivateLoads"),
            "deactivate_loads": _read_stage_load_names(stage, "DeactivateLoads"),
            "activate_regions": _read_stage_region_names(
                stage,
                "ActivateRegions",
            ),
            "deactivate_regions": _read_stage_region_names(
                stage,
                "DeactivateRegions",
            ),
            "gravity": gravity,
            "k0": _read_stage_k0(stage),
            "reset_displacements": reset_displacements,
        })

    return stages


def _read_stage_k0(stage):
    """Read optional K0 initial-stress settings from one construction stage."""
    section = stage.find("K0")
    if section is None:
        return None

    regions = {}

    for item in section.findall("Region"):
        name = item.get("name")
        value = item.get("value")

        if not name:
            raise ValueError(
                'Each <K0><Region> must define name="...".'
            )

        if value is None:
            if item.text is not None:
                value = item.text.strip()

        value = _convert_number(value)

        if value is None:
            raise ValueError(
                f'K0 region "{name}" must define value="...".'
            )

        if name in regions:
            raise ValueError(
                f'K0 region "{name}" is defined more than once.'
            )

        regions[name] = float(value)

    tolerance = _convert_number(
        _text(section, "EquilibriumTolerance")
    )

    return {
        "regions": regions,
        "equilibrium_tolerance": (
            1.0e-8
            if tolerance is None
            else float(tolerance)
        ),
    }


def _read_stage_load_names(stage, container_tag):
    """Read named loads from ActivateLoads or DeactivateLoads."""
    container = stage.find(container_tag)
    if container is None:
        return []

    names = []
    for item in container.findall("Load"):
        name = item.get("name")
        if name is None and item.text is not None:
            name = item.text.strip()
        if name:
            names.append(name)
    return names


def _read_stage_region_names(stage, container_tag):
    """
    Read construction-region names from ActivateRegions/DeactivateRegions.

    XML example
    -----------
    <ActivateRegions>
      <Region name="timbunan_bawah"/>
      <Region name="timbunan_atas"/>
    </ActivateRegions>
    """
    container = stage.find(
        container_tag
    )

    if container is None:
        return []

    names = []

    for item in container.findall(
        "Region"
    ):
        name = item.get(
            "name"
        )

        if name is None and item.text is not None:
            name = item.text.strip()

        if name:
            names.append(
                name
            )

    if len(names) != len(set(names)):
        raise ValueError(
            f"<{container_tag}> contains duplicate region names."
        )

    return names


def _read_solver(root):
    """
    Read linear/nonlinear solver settings.

    Backward compatibility
    ----------------------
    The legacy tag:

        <Tolerance>1.0e-8</Tolerance>

    is still accepted.  If the newer criteria are absent, it is used for
    BOTH force and displacement convergence tolerances.

    Adaptive load stepping
    ----------------------
    Example:

        <LoadStepping type="adaptive">
          <InitialIncrement>0.05</InitialIncrement>
          <MinimumIncrement>0.005</MinimumIncrement>
          <MaximumIncrement>0.20</MaximumIncrement>
          <CutbackFactor>0.50</CutbackFactor>
          <MaximumCutbacks>5</MaximumCutbacks>
          <GrowthFactor>1.50</GrowthFactor>
          <FastConvergenceIterations>4</FastConvergenceIterations>
          <SlowConvergenceIterations>8</SlowConvergenceIterations>
        </LoadStepping>

    If InitialIncrement is omitted, GeoPyFEM uses 1 / LoadSteps for the
    current analysis/stage.  This keeps stage-specific <LoadSteps> useful
    under adaptive stepping.
    """
    section = root.find(
        "Solver"
    )

    if section is None:
        return {}

    nonlinear = section.find(
        "NonlinearSolver"
    )

    solver = {
        "linear_solver": (
            _text(
                section,
                "LinearSolver",
            )
            or "direct"
        ),
    }

    if nonlinear is not None:
        max_iterations = _convert_number(
            _text(
                nonlinear,
                "MaxIterations",
            )
        )

        legacy_tolerance = _convert_number(
            _text(
                nonlinear,
                "Tolerance",
            )
        )

        force_tolerance = _convert_number(
            _text(
                nonlinear,
                "ForceTolerance",
            )
        )

        displacement_tolerance = _convert_number(
            _text(
                nonlinear,
                "DisplacementTolerance",
            )
        )

        default_tolerance = (
            1.0e-8
            if legacy_tolerance is None
            else float(
                legacy_tolerance
            )
        )

        if force_tolerance is None:
            force_tolerance = default_tolerance

        if displacement_tolerance is None:
            displacement_tolerance = default_tolerance

        load_stepping_xml = nonlinear.find(
            "LoadStepping"
        )

        load_stepping = {
            "type": "fixed",
            "initial_increment": None,
            "minimum_increment": None,
            "maximum_increment": None,
            "cutback_factor": 0.50,
            "maximum_cutbacks": 5,
            "growth_factor": 1.50,
            "fast_convergence_iterations": 4,
            "slow_convergence_iterations": 8,
        }

        if load_stepping_xml is not None:
            stepping_type = (
                load_stepping_xml.get(
                    "type"
                )
                or "fixed"
            )

            load_stepping[
                "type"
            ] = stepping_type

            initial_increment = _convert_number(
                _text(
                    load_stepping_xml,
                    "InitialIncrement",
                )
            )

            minimum_increment = _convert_number(
                _text(
                    load_stepping_xml,
                    "MinimumIncrement",
                )
            )

            maximum_increment = _convert_number(
                _text(
                    load_stepping_xml,
                    "MaximumIncrement",
                )
            )

            cutback_factor = _convert_number(
                _text(
                    load_stepping_xml,
                    "CutbackFactor",
                )
            )

            maximum_cutbacks = _convert_number(
                _text(
                    load_stepping_xml,
                    "MaximumCutbacks",
                )
            )

            growth_factor = _convert_number(
                _text(
                    load_stepping_xml,
                    "GrowthFactor",
                )
            )

            fast_iterations = _convert_number(
                _text(
                    load_stepping_xml,
                    "FastConvergenceIterations",
                )
            )

            slow_iterations = _convert_number(
                _text(
                    load_stepping_xml,
                    "SlowConvergenceIterations",
                )
            )

            load_stepping[
                "initial_increment"
            ] = (
                None
                if initial_increment is None
                else float(
                    initial_increment
                )
            )

            load_stepping[
                "minimum_increment"
            ] = (
                None
                if minimum_increment is None
                else float(
                    minimum_increment
                )
            )

            load_stepping[
                "maximum_increment"
            ] = (
                None
                if maximum_increment is None
                else float(
                    maximum_increment
                )
            )

            if cutback_factor is not None:
                load_stepping[
                    "cutback_factor"
                ] = float(
                    cutback_factor
                )

            if maximum_cutbacks is not None:
                load_stepping[
                    "maximum_cutbacks"
                ] = int(
                    maximum_cutbacks
                )

            if growth_factor is not None:
                load_stepping[
                    "growth_factor"
                ] = float(
                    growth_factor
                )

            if fast_iterations is not None:
                load_stepping[
                    "fast_convergence_iterations"
                ] = int(
                    fast_iterations
                )

            if slow_iterations is not None:
                load_stepping[
                    "slow_convergence_iterations"
                ] = int(
                    slow_iterations
                )

        solver[
            "nonlinear_solver"
        ] = {
            "method": (
                _text(
                    nonlinear,
                    "Method",
                )
                or "NewtonRaphson"
            ),
            "max_iterations": (
                25
                if max_iterations is None
                else int(
                    max_iterations
                )
            ),

            # New dual convergence criteria.
            "force_tolerance": float(
                force_tolerance
            ),
            "displacement_tolerance": float(
                displacement_tolerance
            ),

            # Keep this key so older code or user scripts that inspect the
            # parsed dictionary do not immediately break.
            "tolerance": float(
                force_tolerance
            ),

            "load_stepping": load_stepping,
        }

    return solver

def _read_output(
    root,
    xml_directory,
    problem_name,
):
    """
    Read ParaView/VTK, optional Matplotlib, and optional console GP settings.

    Recommended XML
    ---------------
    <Output>

      <VTK enabled="true">
        <Directory>results</Directory>
        <BaseName>case01</BaseName>
        <WriteGaussPoints>true</WriteGaussPoints>
      </VTK>

      <Matplotlib enabled="false"/>

      <Displacement>|u|</Displacement>
      <DeformationScale>1.0</DeformationScale>

      <GaussPointResults enabled="false">
        <ElementID>0</ElementID>
        <GaussPoint>all</GaussPoint>
      </GaussPointResults>

    </Output>

    Relative output directories are interpreted relative to the XML file.
    """
    section = root.find(
        "Output"
    )

    default_directory = (
        xml_directory
        / "results"
    )

    defaults = {
        "displacement": "uy",
        "deformation_scale": 1.0,
        "matplotlib": {
            "enabled": True,
        },
        "vtk": {
            "enabled": False,
            "directory": str(
                default_directory
            ),
            "base_name": (
                problem_name
                or "result"
            ),
            "write_gauss_points": True,
        },
        "gauss_point_results": {
            "enabled": False,
            "element_id": None,
            "gauss_point": "all",
        },
    }

    if section is None:
        return defaults

    displacement = _text(
        section,
        "Displacement",
    )

    deformation_scale = _convert_number(
        _text(
            section,
            "DeformationScale",
        )
    )

    matplotlib_section = section.find(
        "Matplotlib"
    )

    if matplotlib_section is None:
        matplotlib_enabled = True
    else:
        matplotlib_enabled = (
            matplotlib_section.get(
                "enabled",
                "true",
            ).lower()
            == "true"
        )

    vtk_section = section.find(
        "VTK"
    )

    vtk_settings = dict(
        defaults["vtk"]
    )

    if vtk_section is not None:
        vtk_settings["enabled"] = (
            vtk_section.get(
                "enabled",
                "false",
            ).lower()
            == "true"
        )

        directory_text = _text(
            vtk_section,
            "Directory",
        )

        if directory_text:
            directory = Path(
                directory_text
            )

            if not directory.is_absolute():
                directory = (
                    xml_directory
                    / directory
                )

            vtk_settings[
                "directory"
            ] = str(
                directory
            )

        base_name = _text(
            vtk_section,
            "BaseName",
        )

        if base_name:
            vtk_settings[
                "base_name"
            ] = base_name

        write_gauss = _text(
            vtk_section,
            "WriteGaussPoints",
        )

        if write_gauss is not None:
            vtk_settings[
                "write_gauss_points"
            ] = (
                write_gauss.strip().lower()
                == "true"
            )

    gp_section = section.find(
        "GaussPointResults"
    )

    gp_settings = dict(
        defaults[
            "gauss_point_results"
        ]
    )

    if gp_section is not None:
        gp_settings["enabled"] = (
            gp_section.get(
                "enabled",
                "false",
            ).lower()
            == "true"
        )

        element_id = _convert_number(
            _text(
                gp_section,
                "ElementID",
            )
        )

        gauss_point = _text(
            gp_section,
            "GaussPoint",
        )

        if element_id is not None:
            gp_settings[
                "element_id"
            ] = int(
                element_id
            )

        if gauss_point is not None:
            gp_settings[
                "gauss_point"
            ] = gauss_point

    return {
        "displacement": (
            displacement
            or defaults[
                "displacement"
            ]
        ),
        "deformation_scale": (
            defaults[
                "deformation_scale"
            ]
            if deformation_scale is None
            else float(
                deformation_scale
            )
        ),
        "matplotlib": {
            "enabled": matplotlib_enabled,
        },
        "vtk": vtk_settings,
        "gauss_point_results": gp_settings,
    }


def _text(
    parent,
    tag,
):
    if parent is None:
        return None

    item = parent.find(
        tag
    )

    if (
        item is None
        or item.text is None
    ):
        return None

    return item.text.strip()


def _convert_number(value):
    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value
