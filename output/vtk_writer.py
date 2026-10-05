from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from visualization.postprocess import build_result_database


# =============================================================================
# VTK CELL TYPE MAPPING
# =============================================================================
#
# Current active GeoPyFEM domain elements:
#
#   Quad4 -> VTK_QUAD               = 9
#   Tri6  -> VTK_QUADRATIC_TRIANGLE = 22
#
# Gmsh Tri6 and VTK quadratic-triangle local node ordering are compatible:
#
#   corners first: 0,1,2
#   midsides:       0-1, 1-2, 2-0
#
# FUTURE Tri15:
# A complete fourth-order triangle should be written using a high-order /
# Lagrange VTK triangle representation. Do NOT activate that mapping until
# Gmsh-to-VTK high-order local-node ordering has been explicitly verified.
# =============================================================================

VTK_CELL_TYPES = {
    "Tri3": 5,
    "Quad4": 9,
    "Tri6": 22,

    # Future possibilities after the corresponding FEM element is implemented:
    # "Quad8": 23,   # VTK_QUADRATIC_QUAD
    # "Quad9": 28,   # VTK_BIQUADRATIC_QUAD
    #
    # "Tri15": ...   # verify VTK Lagrange-triangle ordering first
}


def write_vtk_results(
    nodes,
    mesh_elements,
    solution,
    material_models,
    materials_by_region,
    vtk_settings,
):
    """
    Write GeoPyFEM results for ParaView.

    Two files are produced:

    1. <basename>.vtu
       Original FEM mesh:
           PointData -> nodal displacement
           CellData  -> integration-weighted element-average strain/stress

    2. <basename>_gauss.vtu
       One VTK_VERTEX cell per Gauss point:
           exact Gauss-point displacement/strain/stress/principal stress

    Parameters
    ----------
    solution : dict
        Current solver result. Must contain solution["u"].

        FUTURE:
        Nonlinear solvers should additionally expose converged Gauss-point
        states. At that stage build_result_database() should read those states
        instead of reconstructing nonlinear stress from total strain.

    vtk_settings : dict
        Parsed <VTK ...> XML settings.

    Returns
    -------
    written_files : dict
        Paths of generated files.
    """
    if "u" not in solution:
        raise ValueError(
            'VTK output requires solution["u"].'
        )

    output_directory = Path(
        vtk_settings.get(
            "directory",
            "results",
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = str(
        vtk_settings.get(
            "base_name",
            "result",
        )
    )

    write_gauss_points = bool(
        vtk_settings.get(
            "write_gauss_points",
            True,
        )
    )

    result_database = build_result_database(
        nodes=nodes,
        mesh_elements=mesh_elements,
        u=solution["u"],
        material_models=material_models,
        gauss_states=solution.get(
            "gauss_states"
        ),
    )

    mesh_path = output_directory / (
        f"{base_name}.vtu"
    )

    _write_mesh_vtu(
        filename=mesh_path,
        nodes=nodes,
        mesh_elements=mesh_elements,
        materials_by_region=materials_by_region,
        result_database=result_database,
        active_element_ids=solution.get(
            "active_element_ids"
        ),
        active_node_ids=solution.get(
            "active_node_ids"
        ),
    )

    written_files = {
        "mesh": str(mesh_path),
    }

    if write_gauss_points:
        gauss_path = output_directory / (
            f"{base_name}_gauss.vtu"
        )

        _write_gauss_vtu(
            filename=gauss_path,
            gauss_results=result_database[
                "gauss_results"
            ],
            materials_by_region=materials_by_region,
            active_element_ids=solution.get(
                "active_element_ids"
            ),
        )

        written_files["gauss"] = str(
            gauss_path
        )

    return written_files


def _write_mesh_vtu(
    filename,
    nodes,
    mesh_elements,
    materials_by_region,
    result_database,
    active_element_ids=None,
    active_node_ids=None,
):
    """
    Write the original FEM unstructured grid.
    """
    npoint = len(nodes)
    ncell = len(mesh_elements)

    points = np.zeros((
        npoint,
        3,
    ))

    points[:, :2] = nodes

    connectivity = []
    offsets = []
    cell_types = []

    current_offset = 0

    for element in mesh_elements:
        element_type = element["type"]

        if element_type not in VTK_CELL_TYPES:
            raise NotImplementedError(
                f'VTK writer does not yet support '
                f'element type "{element_type}".'
            )

        c = element["connectivity"]

        connectivity.extend(
            int(node_id)
            for node_id in c
        )

        current_offset += len(c)
        offsets.append(
            current_offset
        )

        cell_types.append(
            VTK_CELL_TYPES[
                element_type
            ]
        )

    point_results = result_database[
        "point_results"
    ]

    cell_results = result_database[
        "cell_results"
    ]

    if len(cell_results) != ncell:
        raise ValueError(
            "Cell-result count does not match mesh cell count."
        )

    element_ids = np.asarray([
        item["element_id"]
        for item in cell_results
    ], dtype=int)

    material_ids = np.asarray([
        _material_id(
            item["material_region"],
            materials_by_region,
        )
        for item in cell_results
    ], dtype=int)

    if active_element_ids is None:
        active_element_set = set(
            int(
                item["element_id"]
            )
            for item in cell_results
        )
    else:
        active_element_set = {
            int(
                element_id
            )
            for element_id in active_element_ids
        }

    element_active = np.asarray([
        int(
            int(
                item["element_id"]
            )
            in active_element_set
        )
        for item in cell_results
    ], dtype=int)

    if active_node_ids is None:
        node_active = np.ones(
            npoint,
            dtype=int,
        )
    else:
        active_node_set = {
            int(
                node_id
            )
            for node_id in active_node_ids
        }

        node_active = np.asarray([
            int(
                node_id
                in active_node_set
            )
            for node_id in range(
                npoint
            )
        ], dtype=int)

    strain = np.asarray([
        item["strain"]
        for item in cell_results
    ])

    strain_tensor = np.asarray([
        item["strain_tensor"].reshape(-1)
        for item in cell_results
    ])

    stress_tensor = np.asarray([
        item["stress_tensor"].reshape(-1)
        for item in cell_results
    ])

    effective_stress_tensor = np.asarray([
        item["effective_stress_tensor"].reshape(-1)
        for item in cell_results
    ])

    principal = np.asarray([
        item["principal_stresses"]
        for item in cell_results
    ])

    effective_principal = np.asarray([
        item["effective_principal_stresses"]
        for item in cell_results
    ])

    k0_value = np.asarray([
        item["k0"]
        for item in cell_results
    ])

    plastic_strain_tensor = np.asarray([
        item[
            "plastic_strain_tensor"
        ].reshape(-1)
        for item in cell_results
    ])

    equivalent_plastic_strain = np.asarray([
        item[
            "equivalent_plastic_strain"
        ]
        for item in cell_results
    ])

    yielded = np.asarray([
        int(
            item["yielded"]
        )
        for item in cell_results
    ], dtype=int)

    pore_pressure = np.asarray([
        item["pore_pressure"]
        for item in cell_results
    ])

    q_value = np.asarray([
        item["q"]
        for item in cell_results
    ])

    current_yield_stress = np.asarray([
        item["current_yield_stress"]
        for item in cell_results
    ])

    yield_function = np.asarray([
        item["yield_function"]
        for item in cell_results
    ])

    plastic_multiplier = np.asarray([
        item["plastic_multiplier"]
        for item in cell_results
    ])

    root, piece = _new_unstructured_grid(
        number_of_points=npoint,
        number_of_cells=ncell,
    )

    point_data = ET.SubElement(
        piece,
        "PointData",
        {
            "Vectors": "U",
        },
    )

    _add_data_array(
        point_data,
        "U",
        point_results["U"],
        number_of_components=3,
    )

    _add_data_array(
        point_data,
        "ux",
        point_results["ux"],
    )

    _add_data_array(
        point_data,
        "uy",
        point_results["uy"],
    )

    _add_data_array(
        point_data,
        "U_magnitude",
        point_results[
            "U_magnitude"
        ],
    )

    _add_data_array(
        point_data,
        "active_node",
        node_active,
        vtk_type="Int64",
    )

    cell_data = ET.SubElement(
        piece,
        "CellData",
    )

    _add_data_array(
        cell_data,
        "element_id",
        element_ids,
        vtk_type="Int64",
    )

    _add_data_array(
        cell_data,
        "material_id",
        material_ids,
        vtk_type="Int64",
    )

    _add_data_array(
        cell_data,
        "active",
        element_active,
        vtk_type="Int64",
    )

    _add_data_array(
        cell_data,
        "Strain",
        strain_tensor,
        number_of_components=9,
    )

    _add_data_array(
        cell_data,
        "epsilon_xx",
        strain[:, 0],
    )

    _add_data_array(
        cell_data,
        "epsilon_yy",
        strain[:, 1],
    )

    _add_data_array(
        cell_data,
        "gamma_xy",
        strain[:, 2],
    )

    _add_data_array(
        cell_data,
        "Stress",
        stress_tensor,
        number_of_components=9,
    )

    _add_data_array(
        cell_data,
        "sigma_xx",
        stress_tensor[:, 0],
    )

    _add_data_array(
        cell_data,
        "tau_xy",
        stress_tensor[:, 1],
    )

    _add_data_array(
        cell_data,
        "sigma_yy",
        stress_tensor[:, 4],
    )

    _add_data_array(
        cell_data,
        "sigma_zz",
        stress_tensor[:, 8],
    )

    _add_data_array(
        cell_data,
        "sigma1",
        principal[:, 0],
    )

    _add_data_array(
        cell_data,
        "sigma2",
        principal[:, 1],
    )

    _add_data_array(
        cell_data,
        "sigma3",
        principal[:, 2],
    )

    _add_data_array(
        cell_data,
        "EffectiveStress",
        effective_stress_tensor,
        number_of_components=9,
    )

    _add_data_array(
        cell_data,
        "sigma_eff_xx",
        effective_stress_tensor[:, 0],
    )

    _add_data_array(
        cell_data,
        "tau_eff_xy",
        effective_stress_tensor[:, 1],
    )

    _add_data_array(
        cell_data,
        "sigma_eff_yy",
        effective_stress_tensor[:, 4],
    )

    _add_data_array(
        cell_data,
        "sigma_eff_zz",
        effective_stress_tensor[:, 8],
    )

    _add_data_array(
        cell_data,
        "sigma_eff_1",
        effective_principal[:, 0],
    )

    _add_data_array(
        cell_data,
        "sigma_eff_2",
        effective_principal[:, 1],
    )

    _add_data_array(
        cell_data,
        "sigma_eff_3",
        effective_principal[:, 2],
    )

    _add_data_array(
        cell_data,
        "K0",
        k0_value,
    )

    # Nonlinear-ready material-state fields.
    # They are zero/false for the current LinearElastic analysis.
    _add_data_array(
        cell_data,
        "PlasticStrain",
        plastic_strain_tensor,
        number_of_components=9,
    )

    _add_data_array(
        cell_data,
        "equivalent_plastic_strain",
        equivalent_plastic_strain,
    )

    _add_data_array(
        cell_data,
        "yielded",
        yielded,
        vtk_type="Int64",
    )

    _add_data_array(
        cell_data,
        "pore_pressure",
        pore_pressure,
    )

    _add_data_array(
        cell_data,
        "q",
        q_value,
    )

    _add_data_array(
        cell_data,
        "current_yield_stress",
        current_yield_stress,
    )

    _add_data_array(
        cell_data,
        "yield_function",
        yield_function,
    )

    _add_data_array(
        cell_data,
        "plastic_multiplier",
        plastic_multiplier,
    )

    points_xml = ET.SubElement(
        piece,
        "Points",
    )

    _add_data_array(
        points_xml,
        None,
        points,
        number_of_components=3,
    )

    cells_xml = ET.SubElement(
        piece,
        "Cells",
    )

    _add_data_array(
        cells_xml,
        "connectivity",
        np.asarray(
            connectivity,
            dtype=int,
        ),
        vtk_type="Int64",
    )

    _add_data_array(
        cells_xml,
        "offsets",
        np.asarray(
            offsets,
            dtype=int,
        ),
        vtk_type="Int64",
    )

    _add_data_array(
        cells_xml,
        "types",
        np.asarray(
            cell_types,
            dtype=np.uint8,
        ),
        vtk_type="UInt8",
    )

    _write_xml(
        root,
        filename,
    )


def _write_gauss_vtu(
    filename,
    gauss_results,
    materials_by_region,
    active_element_ids=None,
):
    """
    Write exact Gauss-point results as VTK_VERTEX cells.
    """
    ngauss = len(
        gauss_results
    )

    points = np.zeros((
        ngauss,
        3,
    ))

    displacement = np.zeros((
        ngauss,
        3,
    ))

    natural_coordinates = np.zeros((
        ngauss,
        3,
    ))

    element_ids = np.zeros(
        ngauss,
        dtype=int,
    )

    gauss_ids = np.zeros(
        ngauss,
        dtype=int,
    )

    material_ids = np.zeros(
        ngauss,
        dtype=int,
    )

    active = np.ones(
        ngauss,
        dtype=int,
    )

    active_element_set = (
        None
        if active_element_ids is None
        else {
            int(
                element_id
            )
            for element_id in active_element_ids
        }
    )

    weights = np.zeros(
        ngauss,
    )

    detJ = np.zeros(
        ngauss,
    )

    integration_weight = np.zeros(
        ngauss,
    )

    strain = np.zeros((
        ngauss,
        3,
    ))

    strain_tensor = np.zeros((
        ngauss,
        9,
    ))

    stress_tensor = np.zeros((
        ngauss,
        9,
    ))

    effective_stress_tensor = np.zeros((
        ngauss,
        9,
    ))

    principal = np.zeros((
        ngauss,
        3,
    ))

    effective_principal = np.zeros((
        ngauss,
        3,
    ))

    k0_value = np.zeros(
        ngauss,
    )

    plastic_strain_tensor = np.zeros((
        ngauss,
        9,
    ))

    equivalent_plastic_strain = np.zeros(
        ngauss,
    )

    yielded = np.zeros(
        ngauss,
        dtype=int,
    )

    pore_pressure = np.zeros(
        ngauss,
    )

    q_value = np.zeros(
        ngauss,
    )

    current_yield_stress = np.zeros(
        ngauss,
    )

    yield_function = np.zeros(
        ngauss,
    )

    plastic_multiplier = np.zeros(
        ngauss,
    )

    for i, gp in enumerate(
        gauss_results
    ):
        points[i, :2] = gp[
            "position"
        ]

        displacement[i, :2] = gp[
            "displacement"
        ]

        q = np.asarray(
            gp[
                "natural_coordinates"
            ]
        ).reshape(-1)

        natural_coordinates[
            i,
            :len(q),
        ] = q

        element_ids[i] = gp[
            "element_id"
        ]

        gauss_ids[i] = gp[
            "gauss_point"
        ]

        material_ids[i] = _material_id(
            gp["material_region"],
            materials_by_region,
        )

        if active_element_set is not None:
            active[i] = int(
                int(
                    gp["element_id"]
                )
                in active_element_set
            )

        weights[i] = gp[
            "weight"
        ]

        detJ[i] = gp[
            "detJ"
        ]

        integration_weight[i] = gp[
            "integration_weight"
        ]

        strain[i, :] = gp[
            "strain"
        ]

        strain_tensor[i, :] = gp[
            "strain_tensor"
        ].reshape(-1)

        stress_tensor[i, :] = gp[
            "stress_tensor"
        ].reshape(-1)

        effective_stress_tensor[i, :] = gp[
            "effective_stress_tensor"
        ].reshape(-1)

        principal[i, :] = gp[
            "principal_stresses"
        ]

        effective_principal[i, :] = gp[
            "effective_principal_stresses"
        ]

        k0_value[i] = gp[
            "k0"
        ]

        plastic_strain_tensor[
            i,
            :,
        ] = gp[
            "plastic_strain_tensor"
        ].reshape(-1)

        equivalent_plastic_strain[
            i
        ] = gp[
            "equivalent_plastic_strain"
        ]

        yielded[i] = int(
            gp["yielded"]
        )

        pore_pressure[i] = gp[
            "pore_pressure"
        ]

        q_value[i] = gp["q"]
        current_yield_stress[i] = gp[
            "current_yield_stress"
        ]
        yield_function[i] = gp[
            "yield_function"
        ]
        plastic_multiplier[i] = gp[
            "plastic_multiplier"
        ]

    root, piece = _new_unstructured_grid(
        number_of_points=ngauss,
        number_of_cells=ngauss,
    )

    point_data = ET.SubElement(
        piece,
        "PointData",
        {
            "Vectors": "U",
        },
    )

    _add_data_array(
        point_data,
        "element_id",
        element_ids,
        vtk_type="Int64",
    )

    _add_data_array(
        point_data,
        "gauss_point_id",
        gauss_ids,
        vtk_type="Int64",
    )

    _add_data_array(
        point_data,
        "material_id",
        material_ids,
        vtk_type="Int64",
    )

    _add_data_array(
        point_data,
        "active",
        active,
        vtk_type="Int64",
    )

    _add_data_array(
        point_data,
        "natural_coordinates",
        natural_coordinates,
        number_of_components=3,
    )

    _add_data_array(
        point_data,
        "integration_weight",
        integration_weight,
    )

    _add_data_array(
        point_data,
        "gauss_weight",
        weights,
    )

    _add_data_array(
        point_data,
        "detJ",
        detJ,
    )

    _add_data_array(
        point_data,
        "U",
        displacement,
        number_of_components=3,
    )

    _add_data_array(
        point_data,
        "ux",
        displacement[:, 0],
    )

    _add_data_array(
        point_data,
        "uy",
        displacement[:, 1],
    )

    _add_data_array(
        point_data,
        "U_magnitude",
        np.linalg.norm(
            displacement,
            axis=1,
        ),
    )

    _add_data_array(
        point_data,
        "Strain",
        strain_tensor,
        number_of_components=9,
    )

    _add_data_array(
        point_data,
        "epsilon_xx",
        strain[:, 0],
    )

    _add_data_array(
        point_data,
        "epsilon_yy",
        strain[:, 1],
    )

    _add_data_array(
        point_data,
        "gamma_xy",
        strain[:, 2],
    )

    _add_data_array(
        point_data,
        "Stress",
        stress_tensor,
        number_of_components=9,
    )

    _add_data_array(
        point_data,
        "sigma_xx",
        stress_tensor[:, 0],
    )

    _add_data_array(
        point_data,
        "tau_xy",
        stress_tensor[:, 1],
    )

    _add_data_array(
        point_data,
        "sigma_yy",
        stress_tensor[:, 4],
    )

    _add_data_array(
        point_data,
        "sigma_zz",
        stress_tensor[:, 8],
    )

    _add_data_array(
        point_data,
        "sigma1",
        principal[:, 0],
    )

    _add_data_array(
        point_data,
        "sigma2",
        principal[:, 1],
    )

    _add_data_array(
        point_data,
        "sigma3",
        principal[:, 2],
    )

    _add_data_array(
        point_data,
        "EffectiveStress",
        effective_stress_tensor,
        number_of_components=9,
    )

    _add_data_array(
        point_data,
        "sigma_eff_xx",
        effective_stress_tensor[:, 0],
    )

    _add_data_array(
        point_data,
        "tau_eff_xy",
        effective_stress_tensor[:, 1],
    )

    _add_data_array(
        point_data,
        "sigma_eff_yy",
        effective_stress_tensor[:, 4],
    )

    _add_data_array(
        point_data,
        "sigma_eff_zz",
        effective_stress_tensor[:, 8],
    )

    _add_data_array(
        point_data,
        "sigma_eff_1",
        effective_principal[:, 0],
    )

    _add_data_array(
        point_data,
        "sigma_eff_2",
        effective_principal[:, 1],
    )

    _add_data_array(
        point_data,
        "sigma_eff_3",
        effective_principal[:, 2],
    )

    _add_data_array(
        point_data,
        "K0",
        k0_value,
    )

    _add_data_array(
        point_data,
        "PlasticStrain",
        plastic_strain_tensor,
        number_of_components=9,
    )

    _add_data_array(
        point_data,
        "equivalent_plastic_strain",
        equivalent_plastic_strain,
    )

    _add_data_array(
        point_data,
        "yielded",
        yielded,
        vtk_type="Int64",
    )

    _add_data_array(
        point_data,
        "pore_pressure",
        pore_pressure,
    )

    _add_data_array(
        point_data,
        "q",
        q_value,
    )

    _add_data_array(
        point_data,
        "current_yield_stress",
        current_yield_stress,
    )

    _add_data_array(
        point_data,
        "yield_function",
        yield_function,
    )

    _add_data_array(
        point_data,
        "plastic_multiplier",
        plastic_multiplier,
    )

    points_xml = ET.SubElement(
        piece,
        "Points",
    )

    _add_data_array(
        points_xml,
        None,
        points,
        number_of_components=3,
    )

    cells_xml = ET.SubElement(
        piece,
        "Cells",
    )

    _add_data_array(
        cells_xml,
        "connectivity",
        np.arange(
            ngauss,
            dtype=int,
        ),
        vtk_type="Int64",
    )

    _add_data_array(
        cells_xml,
        "offsets",
        np.arange(
            1,
            ngauss + 1,
            dtype=int,
        ),
        vtk_type="Int64",
    )

    # VTK_VERTEX = 1
    _add_data_array(
        cells_xml,
        "types",
        np.ones(
            ngauss,
            dtype=np.uint8,
        ),
        vtk_type="UInt8",
    )

    _write_xml(
        root,
        filename,
    )


def _new_unstructured_grid(
    number_of_points,
    number_of_cells,
):
    root = ET.Element(
        "VTKFile",
        {
            "type": "UnstructuredGrid",
            "version": "0.1",
            "byte_order": "LittleEndian",
        },
    )

    grid = ET.SubElement(
        root,
        "UnstructuredGrid",
    )

    piece = ET.SubElement(
        grid,
        "Piece",
        {
            "NumberOfPoints": str(
                number_of_points
            ),
            "NumberOfCells": str(
                number_of_cells
            ),
        },
    )

    return root, piece


def _add_data_array(
    parent,
    name,
    values,
    vtk_type="Float64",
    number_of_components=None,
):
    values = np.asarray(
        values
    )

    attributes = {
        "type": vtk_type,
        "format": "ascii",
    }

    if name is not None:
        attributes["Name"] = name

    if number_of_components is not None:
        attributes[
            "NumberOfComponents"
        ] = str(
            number_of_components
        )

    data_array = ET.SubElement(
        parent,
        "DataArray",
        attributes,
    )

    flat = values.reshape(-1)

    if vtk_type in (
        "Int64",
        "UInt8",
    ):
        data_array.text = "\n" + " ".join(
            str(int(value))
            for value in flat
        ) + "\n"

    else:
        data_array.text = "\n" + " ".join(
            f"{float(value):.16e}"
            for value in flat
        ) + "\n"

    return data_array


def _material_id(
    material_region,
    materials_by_region,
):
    spec = materials_by_region.get(
        material_region,
        {},
    )

    material_id = spec.get(
        "id"
    )

    if material_id is None:
        # Deterministic fallback if XML material id is absent.
        regions = sorted(
            materials_by_region.keys()
        )

        return (
            regions.index(
                material_region
            )
            + 1
        )

    return int(
        material_id
    )


def _write_xml(
    root,
    filename,
):
    filename = Path(
        filename
    )

    ET.indent(
        root,
        space="  ",
    )

    tree = ET.ElementTree(
        root
    )

    tree.write(
        filename,
        encoding="utf-8",
        xml_declaration=True,
    )
