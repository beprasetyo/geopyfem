import numpy as np

from assembly.load_vector import _newton_to_problem_force_scale
from elements.elemtype import get_element_type


def _vertical_intersection_interval(polygon, x, tolerance=1.0e-12):
    """
    Return the y-interval cut by the vertical line X=x through a straight-sided
    convex element polygon.

    GeoPyFEM currently uses straight-sided Tri6/Quad4 geometry for K0
    overburden integration; midside nodes do not define a curved boundary.
    """
    polygon = np.asarray(polygon, dtype=float)
    values = []

    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        xmin = min(x1, x2)
        xmax = max(x1, x2)

        if x < xmin - tolerance or x > xmax + tolerance:
            continue

        dx = x2 - x1

        if abs(dx) <= tolerance:
            if abs(x - x1) <= tolerance:
                values.extend([y1, y2])
            continue

        t = (x - x1) / dx
        if -tolerance <= t <= 1.0 + tolerance:
            t = min(max(t, 0.0), 1.0)
            values.append(y1 + t * (y2 - y1))

    if not values:
        return None

    # Remove duplicate vertex intersections.
    values = sorted(values)
    unique = []
    for value in values:
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)

    if len(unique) < 2:
        return None

    return float(unique[0]), float(unique[-1])


def _element_corner_polygon(nodes, element):
    info = get_element_type(element["type"])

    if info["dimension"] != 2:
        return None

    corner_nodes = info.get("corner_nodes")
    if not corner_nodes:
        raise NotImplementedError(
            f'K0 initialization requires corner-node information for '
            f'element type "{element["type"]}".'
        )

    connectivity = element["connectivity"]
    corner_connectivity = [
        connectivity[index]
        for index in corner_nodes
    ]

    return np.asarray(
        nodes[corner_connectivity, :],
        dtype=float,
    )


def _build_overburden_column_data(
    nodes,
    mesh_elements,
    material_models,
    gravity,
    units,
    active_element_ids=None,
):
    """Precompute element polygons and unit weights used by K0 integration."""
    gravity = gravity or {}

    if not gravity.get("enabled", False):
        raise ValueError(
            "K0 initial stress requires gravity enabled in the K0 stage."
        )

    gx = float(gravity.get("gx", 0.0) or 0.0)
    gy = float(
        gravity.get("gy", -9.81)
        if gravity.get("gy") is not None
        else -9.81
    )

    if abs(gx) > 1.0e-12:
        raise NotImplementedError(
            "The current K0 initializer supports vertical gravity only (gx=0)."
        )

    if gy >= 0.0:
        raise ValueError(
            "K0 initialization expects downward gravity (gy < 0)."
        )

    force_scale = _newton_to_problem_force_scale(units)
    g = abs(gy)

    column_elements = []

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
        info = get_element_type(element["type"])
        if info["dimension"] != 2:
            continue

        region = element["material"]
        if region not in material_models:
            raise ValueError(
                f'K0 element {element["id"]} uses unknown material '
                f'region "{region}".'
            )

        density = getattr(material_models[region], "density", None)
        if density is None:
            raise ValueError(
                f'K0 requires density for material region "{region}".'
            )

        unit_weight = float(density) * g * force_scale

        column_elements.append({
            "polygon": _element_corner_polygon(nodes, element),
            "unit_weight": unit_weight,
            "region": region,
            "element_id": element_id,
        })

    return column_elements


def _vertical_total_stress_compression(
    x,
    y,
    column_elements,
):
    """
    Integrate total overburden stress vertically above point (x,y).

    Compression is returned positive here. GeoPyFEM's stored stress convention
    is tension-positive, so the value is negated when written to state.stress.
    """
    sigma_v = 0.0

    scale = max(1.0, abs(x), abs(y))
    tol = 1.0e-10 * scale

    for item in column_elements:
        interval = _vertical_intersection_interval(
            item["polygon"],
            x,
            tolerance=tol,
        )

        if interval is None:
            continue

        y_low, y_high = interval

        # Only the portion vertically above the evaluation point contributes.
        thickness = y_high - max(y_low, y)

        if thickness > tol:
            sigma_v += (
                item["unit_weight"]
                * thickness
            )

    return float(sigma_v)


def initialize_k0_stress(
    nodes,
    mesh_elements,
    material_models,
    state_manager,
    k0_specification,
    gravity,
    units=None,
    active_element_ids=None,
):
    """
    Initialize a geostatic K0 stress field at every Gauss point.

    Current K0 assumptions
    ----------------------
    - 2-D plane strain.
    - Vertical gravity only.
    - K0 principal directions coincide with global x/y/z axes.
    - Vertical total stress comes from vertical overburden integration.
    - Pore pressure has already been initialized in GaussPointState.
    - K0 acts on EFFECTIVE stress:

          sigma'_h = K0 * sigma'_v

    - Total stress is reconstructed by:

          sigma_v = sigma'_v + u
          sigma_h = sigma'_h + u

      in geotechnical compression-positive notation.

    GeoPyFEM storage convention
    ---------------------------
    GeoPyFEM stores the constitutive state as EFFECTIVE stress, with tension
    positive.  Therefore compression is negative:

        state.effective_stress
            = [-sigma'_h, -sigma'_v, -sigma'_h, 0, 0, 0]

    Pore pressure is stored separately.  Total stress used by global
    equilibrium is derived automatically as:

        sigma_total = sigma_effective - u I

    This makes K0, future Mohr-Coulomb/MCC stress integration, and pore
    pressure use one unambiguous stress contract.
    """
    k0_specification = k0_specification or {}

    region_values = dict(
        k0_specification.get("regions", {})
    )

    if not region_values:
        raise ValueError(
            "K0 stage requires at least one <Region name=... value=.../>."
        )

    for region, value in region_values.items():
        value = float(value)
        if value < 0.0:
            raise ValueError(
                f'K0 for region "{region}" must be non-negative.'
            )
        if region not in material_models:
            raise ValueError(
                f'K0 region "{region}" has no material model.'
            )
        region_values[region] = value

    column_elements = _build_overburden_column_data(
        nodes=nodes,
        mesh_elements=mesh_elements,
        material_models=material_models,
        gravity=gravity,
        units=units,
        active_element_ids=active_element_ids,
    )

    minimum_sigma_v = None
    maximum_sigma_v = None
    maximum_u = 0.0
    minimum_effective_sigma_v = None
    maximum_effective_sigma_v = None
    total_points = 0

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

        info = get_element_type(element["type"])
        if info["dimension"] != 2:
            continue

        region = element["material"]
        if region not in region_values:
            raise ValueError(
                f'K0 stage does not define a coefficient for active material '
                f'region "{region}".'
            )

        k0 = region_values[region]
        connectivity = element["connectivity"]
        coordinates = nodes[connectivity, :]
        shape = info["shape"]
        gauss_points, _ = info["gauss"]()
        for gp_index, q in enumerate(gauss_points):
            position = shape(q) @ coordinates
            x = float(position[0])
            y = float(position[1])

            committed_state = state_manager.get_committed(
                element_id,
                gp_index,
            )

            pore_pressure = max(
                float(committed_state.pore_pressure),
                0.0,
            )

            sigma_v_total = _vertical_total_stress_compression(
                x=x,
                y=y,
                column_elements=column_elements,
            )

            sigma_v_effective = max(
                sigma_v_total - pore_pressure,
                0.0,
            )

            sigma_h_effective = (
                k0 * sigma_v_effective
            )

            sigma_h_total = (
                sigma_h_effective
                + pore_pressure
            )

            total_stress = np.array([
                -sigma_h_total,
                -sigma_v_total,
                -sigma_h_total,
                0.0,
                0.0,
                0.0,
            ], dtype=float)

            effective_stress = np.array([
                -sigma_h_effective,
                -sigma_v_effective,
                -sigma_h_effective,
                0.0,
                0.0,
                0.0,
            ], dtype=float)

            for state in (
                committed_state,
                state_manager.get_trial(
                    element_id,
                    gp_index,
                ),
            ):
                state.strain[:] = 0.0
                state.effective_stress[:] = effective_stress
                state.plastic_strain[:] = 0.0
                state.equivalent_plastic_strain = 0.0
                state.yielded = False

                state.internal_variables["k0"] = float(k0)
                state.internal_variables[
                    "k0_effective_stress_voigt6"
                ] = effective_stress.copy()
                state.internal_variables[
                    "k0_total_stress_voigt6"
                ] = total_stress.copy()
                state.internal_variables[
                    "vertical_total_stress_compression"
                ] = float(sigma_v_total)
                state.internal_variables[
                    "vertical_effective_stress_compression"
                ] = float(sigma_v_effective)
                state.internal_variables[
                    "horizontal_total_stress_compression"
                ] = float(sigma_h_total)
                state.internal_variables[
                    "horizontal_effective_stress_compression"
                ] = float(sigma_h_effective)

            total_points += 1
            maximum_u = max(maximum_u, pore_pressure)

            if minimum_sigma_v is None:
                minimum_sigma_v = sigma_v_total
                maximum_sigma_v = sigma_v_total
                minimum_effective_sigma_v = sigma_v_effective
                maximum_effective_sigma_v = sigma_v_effective
            else:
                minimum_sigma_v = min(minimum_sigma_v, sigma_v_total)
                maximum_sigma_v = max(maximum_sigma_v, sigma_v_total)
                minimum_effective_sigma_v = min(
                    minimum_effective_sigma_v,
                    sigma_v_effective,
                )
                maximum_effective_sigma_v = max(
                    maximum_effective_sigma_v,
                    sigma_v_effective,
                )

    if minimum_sigma_v is None:
        minimum_sigma_v = 0.0
        maximum_sigma_v = 0.0
        minimum_effective_sigma_v = 0.0
        maximum_effective_sigma_v = 0.0

    return {
        "total_gauss_points": total_points,
        "regions": region_values,
        "minimum_vertical_total_stress": float(minimum_sigma_v),
        "maximum_vertical_total_stress": float(maximum_sigma_v),
        "minimum_vertical_effective_stress": float(
            minimum_effective_sigma_v
        ),
        "maximum_vertical_effective_stress": float(
            maximum_effective_sigma_v
        ),
        "maximum_pore_pressure": float(maximum_u),
    }
