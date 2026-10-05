import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from elements.elemtype import get_element_type
from elements.kinematics import build_B_matrix


# =============================================================================
# SHARED KINEMATICS
# =============================================================================
#
# GeoPyFEM now uses one B-matrix implementation:
#
#     elements/kinematics.py -> build_B_matrix()
#
# The same function is called by:
#
#     assembly/global_stiffness.py
#     visualization/postprocess.py
#
# This keeps strain evaluation consistent between the solver and output.
# =============================================================================


# =============================================================================
# QUICK MATPLOTLIB VISUALIZATION
# =============================================================================

def build_plot_triangles(mesh_elements):
    """
    Convert supported 2-D FEM elements into linear triangles for Matplotlib.

    This subdivision is visualization-only. The FEM formulation and the VTU
    output retain the original finite element type.
    """
    triangles = []

    for element in mesh_elements:
        element_info = get_element_type(element["type"])
        local_triangles = element_info["plot_triangles"]

        if not local_triangles:
            continue

        c = element["connectivity"]

        for local_triangle in local_triangles:
            triangles.append([
                c[local_index]
                for local_index in local_triangle
            ])

    if not triangles:
        raise ValueError(
            "No drawable 2-D element triangles were generated for visualization."
        )

    return np.asarray(triangles, dtype=int)


def displacement_components(u):
    """
    Split:
        u = [ux0, uy0, ux1, uy1, ...]
    into ux and uy.
    """
    return u[0::2], u[1::2]


def displacement_result(u, component="uy"):
    """
    Return one nodal displacement scalar field.
    """
    ux, uy = displacement_components(u)

    component = str(component).strip().lower()

    if component == "ux":
        return ux, "Horizontal displacement ux"

    if component == "uy":
        return uy, "Vertical displacement uy"

    if component in ("magnitude", "|u|", "u"):
        return np.sqrt(ux**2 + uy**2), "Displacement magnitude |u|"

    raise ValueError(
        f'Unknown displacement component "{component}". '
        'Use "ux", "uy", "magnitude", or "|u|".'
    )


def plot_displacement(
    nodes,
    mesh_elements,
    u,
    component="uy",
    deformation_scale=1.0,
    levels=14,
    show_nodes=True,
    show_mesh=True,
):
    """
    Quick/debug Matplotlib displacement plot.

    VTU/ParaView output is intended as the main result-visualization path.
    """
    ux, uy = displacement_components(u)

    if len(nodes) != len(ux):
        raise ValueError(
            "Number of displacement nodes does not match the node array."
        )

    result, label = displacement_result(
        u,
        component=component,
    )

    xvec = nodes[:, 0] + deformation_scale * ux
    yvec = nodes[:, 1] + deformation_scale * uy

    triangles = build_plot_triangles(mesh_elements)

    triangulation = tri.Triangulation(
        xvec,
        yvec,
        triangles,
    )

    contour = plt.tricontourf(
        triangulation,
        result,
        levels=levels,
        cmap=plt.cm.jet,
    )

    if show_nodes:
        plt.scatter(
            xvec,
            yvec,
            marker="o",
            c="b",
            s=2,
        )

    if show_mesh:
        plt.triplot(
            triangulation,
            linewidth=0.3,
        )

    plt.grid()
    plt.colorbar(contour, label=label)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")

    if deformation_scale == 1.0:
        scale_text = ""
    else:
        scale_text = f" (deformation scale = {deformation_scale:g})"

    plt.title(f"Deformed mesh and {label}{scale_text}")
    plt.show()


# =============================================================================
# GAUSS-POINT KINEMATICS AND STRESS
# =============================================================================

def element_displacement_vector(u, connectivity):
    """
    Extract:
        ue = [ux1, uy1, ux2, uy2, ...]^T
    """
    ue = np.zeros(2 * len(connectivity))

    for a, node_id in enumerate(connectivity):
        ue[2*a] = u[2*node_id]
        ue[2*a + 1] = u[2*node_id + 1]

    return ue


def strain_tensor_plane_strain(strain):
    """
    Engineering strain vector -> complete 3x3 plane-strain tensor.
    """
    epsilon_xx, epsilon_yy, gamma_xy = np.asarray(
        strain,
        dtype=float,
    )

    epsilon_xy = 0.5 * gamma_xy

    return np.array([
        [epsilon_xx, epsilon_xy, 0.0],
        [epsilon_xy, epsilon_yy, 0.0],
        [0.0, 0.0, 0.0],
    ])


def principal_stresses(stress_tensor):
    """
    Return:
        [sigma1, sigma2, sigma3]
    with mathematical ordering:
        sigma1 >= sigma2 >= sigma3.

    Current GeoPyFEM sign convention is tension-positive.
    """
    stress_tensor = np.asarray(
        stress_tensor,
        dtype=float,
    )

    if stress_tensor.shape != (3, 3):
        raise ValueError(
            "principal_stresses() requires a 3x3 stress tensor."
        )

    values = np.linalg.eigvalsh(stress_tensor)

    sigma3, sigma2, sigma1 = values

    return np.array([
        sigma1,
        sigma2,
        sigma3,
    ])


def _calculate_one_element_gauss_results(
    nodes,
    element,
    u,
    material_models,
    gauss_states=None,
):
    """
    Calculate/output all Gauss-point results for one domain element.

    Preferred path
    --------------
    If committed ``gauss_states`` are supplied by solver.py, stress and strain
    are READ from those states.

    This is now the normal GeoPyFEM output path and is essential for future
    history-dependent constitutive models.

    Backward-compatible fallback
    ----------------------------
    If no state database is supplied, LinearElastic results can still be
    reconstructed from:

        strain = B @ ue
        stress = material.stress_tensor(strain)

    That fallback is useful for tests, but nonlinear MC/MCC/BBM output must
    use the solver-stored committed state.
    """
    element_info = get_element_type(
        element["type"]
    )

    if element_info["dimension"] != 2:
        raise ValueError(
            f'Element {element["id"]} is not a 2-D domain element.'
        )

    connectivity = element["connectivity"]

    element_coordinates = nodes[
        connectivity,
        :
    ]

    shape = element_info["shape"]
    gradshape = element_info["gradshape"]
    gauss_rule = element_info["gauss"]

    gauss_points, gauss_weights = (
        gauss_rule()
    )

    ue = element_displacement_vector(
        u,
        connectivity,
    )

    nodal_displacements = np.column_stack((
        ue[0::2],
        ue[1::2],
    ))

    material_region = element[
        "material"
    ]

    if material_region not in material_models:
        raise ValueError(
            f'No material model exists for region "{material_region}".'
        )

    material = material_models[
        material_region
    ]

    element_id = int(
        element["id"]
    )

    stored_states = None

    if gauss_states is not None:
        if element_id not in gauss_states:
            raise ValueError(
                f"No committed Gauss-point states exist for "
                f"element {element_id}."
            )

        stored_states = gauss_states[
            element_id
        ]

        if len(stored_states) != len(
            gauss_points
        ):
            raise ValueError(
                f"Stored Gauss-point state count does not match "
                f"the integration rule for element {element_id}."
            )

    gauss_results = []

    for gp_index0, (q, w) in enumerate(
        zip(
            gauss_points,
            gauss_weights,
        )
    ):
        # User-facing Gauss-point number remains one-based.
        gp_number = gp_index0 + 1

        N = shape(q)

        # B is still evaluated here because:
        #   - detJ is needed for weighted cell averaging;
        #   - the fallback path needs strain = B @ ue.
        B, detJ, _ = build_B_matrix(
            element_coordinates=element_coordinates,
            gradshape=gradshape,
            natural_coordinates=q,
        )

        if stored_states is not None:
            state = stored_states[
                gp_index0
            ]

            strain = np.asarray(
                state.strain_2d,
                dtype=float,
            )

            strain_tensor = np.asarray(
                state.strain_tensor,
                dtype=float,
            )

            effective_stress_tensor = np.asarray(
                state.effective_stress_tensor,
                dtype=float,
            )

            stress_tensor = np.asarray(
                state.total_stress_tensor,
                dtype=float,
            )

            plastic_strain_tensor = np.asarray(
                state.plastic_strain_tensor,
                dtype=float,
            )

            equivalent_plastic_strain = float(
                state.equivalent_plastic_strain
            )

            yielded = bool(
                state.yielded
            )

            pore_pressure = float(
                state.pore_pressure
            )

            k0_value = float(
                state.internal_variables.get(
                    "k0",
                    0.0,
                )
            )

            q_value = float(
                state.internal_variables.get(
                    "q",
                    0.0,
                )
            )

            current_yield_stress = float(
                state.internal_variables.get(
                    "current_yield_stress",
                    0.0,
                )
            )

            yield_function = float(
                state.internal_variables.get(
                    "yield_function",
                    0.0,
                )
            )

            plastic_multiplier = float(
                state.internal_variables.get(
                    "plastic_multiplier",
                    0.0,
                )
            )

        else:
            # Linear-elastic fallback only.
            if not hasattr(
                material,
                "stress_tensor",
            ):
                raise NotImplementedError(
                    f'Material model "{getattr(material, "model_name", type(material).__name__)}" '
                    "does not provide stress_tensor(), and no committed "
                    "Gauss-point state was supplied."
                )

            strain = B @ ue

            strain_tensor = (
                strain_tensor_plane_strain(
                    strain
                )
            )

            effective_stress_tensor = (
                material.stress_tensor(
                    strain
                )
            )

            # Fallback has no supplied pore-pressure state.
            stress_tensor = np.asarray(
                effective_stress_tensor,
                dtype=float,
            ).copy()

            plastic_strain_tensor = np.zeros(
                (3, 3),
                dtype=float,
            )

            equivalent_plastic_strain = 0.0
            yielded = False
            pore_pressure = 0.0
            k0_value = 0.0

            q_value = 0.0
            current_yield_stress = 0.0
            yield_function = 0.0
            plastic_multiplier = 0.0

        principal = principal_stresses(
            stress_tensor
        )

        effective_principal = principal_stresses(
            effective_stress_tensor
        )

        position = (
            N @ element_coordinates
        )

        displacement = (
            N @ nodal_displacements
        )

        gauss_results.append({
            "element_id": element_id,
            "element_type": element[
                "type"
            ],
            "material_region": material_region,
            "gauss_point": gp_number,
            "natural_coordinates": np.asarray(
                q,
                dtype=float,
            ),
            "weight": float(w),
            "detJ": float(detJ),
            "integration_weight": float(
                w * detJ
            ),
            "position": np.asarray(
                position,
                dtype=float,
            ),
            "displacement": np.asarray(
                displacement,
                dtype=float,
            ),
            "displacement_magnitude": float(
                np.linalg.norm(
                    displacement
                )
            ),
            "strain": np.asarray(
                strain,
                dtype=float,
            ),
            "strain_tensor": np.asarray(
                strain_tensor,
                dtype=float,
            ),
            "stress_tensor": np.asarray(
                stress_tensor,
                dtype=float,
            ),
            "principal_stresses": np.asarray(
                principal,
                dtype=float,
            ),
            "effective_stress_tensor": np.asarray(
                effective_stress_tensor,
                dtype=float,
            ),
            "effective_principal_stresses": np.asarray(
                effective_principal,
                dtype=float,
            ),
            "k0": k0_value,

            # Nonlinear-ready state output.
            "plastic_strain_tensor": np.asarray(
                plastic_strain_tensor,
                dtype=float,
            ),
            "equivalent_plastic_strain": (
                equivalent_plastic_strain
            ),
            "yielded": yielded,
            "pore_pressure": pore_pressure,

            # Generic scalar constitutive-history outputs used by the
            # current Von Mises model. Other material models may leave them
            # equal to zero or define their own future output fields.
            "q": q_value,
            "current_yield_stress": current_yield_stress,
            "yield_function": yield_function,
            "plastic_multiplier": plastic_multiplier,
        })

    return {
        "element_id": element_id,
        "element_type": element[
            "type"
        ],
        "material_region": material_region,
        "connectivity": list(
            connectivity
        ),
        "gauss_points": gauss_results,
    }

def calculate_element_gauss_results(
    nodes,
    mesh_elements,
    u,
    material_models,
    element_id,
    gauss_states=None,
):
    """
    Calculate/read all Gauss-point results for one internal GeoPyFEM element.
    """
    element = next(
        (
            item
            for item in mesh_elements
            if item["id"] == element_id
        ),
        None,
    )

    if element is None:
        raise ValueError(
            f"Element ID {element_id} was not found."
        )

    return _calculate_one_element_gauss_results(
        nodes=nodes,
        element=element,
        u=u,
        material_models=material_models,
        gauss_states=gauss_states,
    )


def calculate_all_gauss_results(
    nodes,
    mesh_elements,
    u,
    material_models,
    gauss_states=None,
):
    """
    Calculate/read Gauss-point results for every domain element.
    """
    return [
        _calculate_one_element_gauss_results(
            nodes=nodes,
            element=element,
            u=u,
            material_models=material_models,
            gauss_states=gauss_states,
        )
        for element in mesh_elements
    ]


def calculate_cell_average_results(
    element_results,
):
    """
    Convert Gauss-point results to one integration-weighted result per cell.

    Weight:
        gauss_weight * detJ

    Stress principal values are calculated from the AVERAGED stress tensor,
    not by averaging sigma1/sigma2/sigma3 independently.

    Plastic-state quantities
    ------------------------
    plastic_strain:
        integration-weighted average tensor

    equivalent_plastic_strain:
        integration-weighted average scalar

    yielded:
        True if ANY Gauss point in the cell is yielded.

    Exact Gauss-point values remain available in the *_gauss.vtu database.
    """
    cell_results = []

    for element_result in element_results:
        gps = element_result[
            "gauss_points"
        ]

        weights = np.asarray([
            gp["integration_weight"]
            for gp in gps
        ])

        weight_sum = np.sum(
            weights
        )

        if weight_sum <= 0.0:
            raise ValueError(
                f'Element {element_result["element_id"]} '
                "has non-positive integration-weight sum."
            )

        def weighted_average(key):
            return sum(
                gp[key] * weight
                for gp, weight in zip(
                    gps,
                    weights,
                )
            ) / weight_sum

        strain = weighted_average(
            "strain"
        )

        strain_tensor = weighted_average(
            "strain_tensor"
        )

        stress_tensor = weighted_average(
            "stress_tensor"
        )

        effective_stress_tensor = weighted_average(
            "effective_stress_tensor"
        )

        plastic_strain_tensor = (
            weighted_average(
                "plastic_strain_tensor"
            )
        )

        equivalent_plastic_strain = (
            sum(
                gp[
                    "equivalent_plastic_strain"
                ] * weight
                for gp, weight in zip(
                    gps,
                    weights,
                )
            )
            / weight_sum
        )

        def weighted_scalar(key):
            return (
                sum(
                    gp[key] * weight
                    for gp, weight in zip(
                        gps,
                        weights,
                    )
                )
                / weight_sum
            )

        pore_pressure = weighted_scalar(
            "pore_pressure"
        )

        k0_value = weighted_scalar(
            "k0"
        )

        q_value = weighted_scalar(
            "q"
        )

        current_yield_stress = weighted_scalar(
            "current_yield_stress"
        )

        yield_function = weighted_scalar(
            "yield_function"
        )

        plastic_multiplier = weighted_scalar(
            "plastic_multiplier"
        )

        yielded = any(
            bool(
                gp["yielded"]
            )
            for gp in gps
        )

        principal = principal_stresses(
            stress_tensor
        )

        effective_principal = principal_stresses(
            effective_stress_tensor
        )

        cell_results.append({
            "element_id": element_result[
                "element_id"
            ],
            "element_type": element_result[
                "element_type"
            ],
            "material_region": element_result[
                "material_region"
            ],
            "strain": np.asarray(
                strain,
                dtype=float,
            ),
            "strain_tensor": np.asarray(
                strain_tensor,
                dtype=float,
            ),
            "stress_tensor": np.asarray(
                stress_tensor,
                dtype=float,
            ),
            "principal_stresses": np.asarray(
                principal,
                dtype=float,
            ),
            "effective_stress_tensor": np.asarray(
                effective_stress_tensor,
                dtype=float,
            ),
            "effective_principal_stresses": np.asarray(
                effective_principal,
                dtype=float,
            ),
            "k0": float(k0_value),
            "plastic_strain_tensor": np.asarray(
                plastic_strain_tensor,
                dtype=float,
            ),
            "equivalent_plastic_strain": float(
                equivalent_plastic_strain
            ),
            "yielded": bool(
                yielded
            ),
            "pore_pressure": float(
                pore_pressure
            ),
            "q": float(
                q_value
            ),
            "current_yield_stress": float(
                current_yield_stress
            ),
            "yield_function": float(
                yield_function
            ),
            "plastic_multiplier": float(
                plastic_multiplier
            ),
        })

    return cell_results


def build_result_database(
    nodes,
    mesh_elements,
    u,
    material_models,
    gauss_states=None,
):
    """
    Build the result database used by GeoPyFEM output writers.

    PointData
    ---------
    Nodal displacement.

    CellData
    --------
    Integration-weighted element-average strain/stress/material-state output.

    Gauss-point data
    ----------------
    Exact integration-point result/state.

    State precedence
    ----------------
    If ``gauss_states`` is provided, committed solver state is the authoritative
    source for stress/strain/history.

    If it is absent, the current LinearElastic fallback reconstructs stress
    from total displacement for backward compatibility.
    """
    ux, uy = displacement_components(
        u
    )

    point_results = {
        "U": np.column_stack((
            ux,
            uy,
            np.zeros_like(ux),
        )),
        "ux": np.asarray(
            ux
        ),
        "uy": np.asarray(
            uy
        ),
        "U_magnitude": np.sqrt(
            ux**2 + uy**2
        ),
    }

    element_results = (
        calculate_all_gauss_results(
            nodes=nodes,
            mesh_elements=mesh_elements,
            u=u,
            material_models=material_models,
            gauss_states=gauss_states,
        )
    )

    cell_results = (
        calculate_cell_average_results(
            element_results
        )
    )

    gauss_results = [
        gp
        for element_result in element_results
        for gp in element_result[
            "gauss_points"
        ]
    ]

    return {
        "point_results": point_results,
        "cell_results": cell_results,
        "gauss_results": gauss_results,
    }


# =============================================================================
# OPTIONAL CONSOLE GAUSS-POINT REPORT
# =============================================================================

def print_element_gauss_results(
    nodes,
    mesh_elements,
    u,
    material_models,
    element_id,
    gauss_point="all",
    units=None,
    gauss_states=None,
):
    """
    Print one element's Gauss-point results for benchmark/debug work.
    """
    result = calculate_element_gauss_results(
        nodes=nodes,
        mesh_elements=mesh_elements,
        u=u,
        material_models=material_models,
        element_id=element_id,
        gauss_states=gauss_states,
    )

    selected_gp = _normalize_gauss_point_selection(
        gauss_point
    )

    stress_unit = _stress_unit_label(
        units
    )

    length_unit = _length_unit_label(
        units
    )

    print()
    print("=" * 72)
    print("GAUSS-POINT RESULTS")
    print("=" * 72)
    print(
        f'Element ID      : {result["element_id"]} '
        "(internal zero-based ID)"
    )
    print(
        f'Element type    : {result["element_type"]}'
    )
    print(
        f'Material region : {result["material_region"]}'
    )
    print(
        f'Connectivity    : {result["connectivity"]}'
    )
    print("Stress sign     : tension positive")
    print(
        "Principal order : sigma1 >= sigma2 >= sigma3"
    )

    printed = 0

    for gp in result["gauss_points"]:
        if (
            selected_gp != "all"
            and gp["gauss_point"] != selected_gp
        ):
            continue

        printed += 1

        x, y = gp["position"]
        ux, uy = gp["displacement"]

        exx, eyy, gxy = gp["strain"]

        stress = gp["stress_tensor"]

        sxx = stress[0, 0]
        syy = stress[1, 1]
        szz = stress[2, 2]
        txy = stress[0, 1]

        sigma1, sigma2, sigma3 = (
            gp["principal_stresses"]
        )

        print()
        print("-" * 72)
        print(
            f'Gauss point {gp["gauss_point"]}'
        )
        print("-" * 72)

        print(
            "Natural coord.  : "
            + np.array2string(
                gp["natural_coordinates"],
                precision=8,
            )
        )

        print(
            f"Position        : "
            f"x={x:.8e} {length_unit}, "
            f"y={y:.8e} {length_unit}"
        )

        print("Displacement:")
        print(
            f"  ux            = "
            f"{ux:.8e} {length_unit}"
        )
        print(
            f"  uy            = "
            f"{uy:.8e} {length_unit}"
        )
        print(
            f"  |u|           = "
            f"{gp['displacement_magnitude']:.8e} "
            f"{length_unit}"
        )

        print("Strain:")
        print(
            f"  epsilon_xx    = {exx:.8e}"
        )
        print(
            f"  epsilon_yy    = {eyy:.8e}"
        )
        print(
            f"  gamma_xy      = {gxy:.8e}"
        )
        print(
            "  epsilon_zz    = "
            "0.00000000e+00  (plane strain)"
        )

        print("Stress:")
        print(
            f"  sigma_xx      = "
            f"{sxx:.8e} {stress_unit}"
        )
        print(
            f"  sigma_yy      = "
            f"{syy:.8e} {stress_unit}"
        )
        print(
            f"  sigma_zz      = "
            f"{szz:.8e} {stress_unit}"
        )
        print(
            f"  tau_xy        = "
            f"{txy:.8e} {stress_unit}"
        )

        print("Principal stress:")
        print(
            f"  sigma1        = "
            f"{sigma1:.8e} {stress_unit}"
        )
        print(
            f"  sigma2        = "
            f"{sigma2:.8e} {stress_unit}"
        )
        print(
            f"  sigma3        = "
            f"{sigma3:.8e} {stress_unit}"
        )

    if printed == 0:
        available = len(
            result["gauss_points"]
        )

        raise ValueError(
            f"Gauss point {selected_gp} does not exist. "
            f"Available: 1 to {available}."
        )

    print("=" * 72)
    print()

    return result


def _normalize_gauss_point_selection(value):
    if value is None:
        return "all"

    if isinstance(value, str):
        text = value.strip().lower()

        if text == "all":
            return "all"

        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(
                'GaussPoint must be "all" or a one-based integer.'
            ) from exc

    value = int(value)

    if value < 1:
        raise ValueError(
            "GaussPoint numbering is one-based and must be >= 1."
        )

    return value


def _stress_unit_label(units):
    units = units or {}

    force = units.get("force")
    length = units.get("length")

    if force is None or length is None:
        return "[stress unit]"

    force_text = str(force)
    length_text = str(length)

    if (
        force_text.lower() == "kn"
        and length_text.lower() == "m"
    ):
        return "kPa"

    if (
        force_text.lower() == "n"
        and length_text.lower() == "m"
    ):
        return "Pa"

    return (
        f"{force_text}/{length_text}^2"
    )


def _length_unit_label(units):
    units = units or {}

    return str(
        units.get("length")
        or "[length unit]"
    )
