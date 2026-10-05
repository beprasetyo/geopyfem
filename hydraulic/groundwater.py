import numpy as np

from elements.elemtype import get_element_type


class PhreaticSurface:
    """
    Static phreatic surface with hydrostatic pore-water pressure.

    Supported geometry
    ------------------
    horizontal
        Constant water-table elevation.

    polyline
        Piecewise-linear water table defined by (x, y) points.

    Hydrostatic pressure
    --------------------
    Below the phreatic surface:

        u = gamma_w * (y_w(x) - y)

    At or above the phreatic surface:

        u = 0

    No suction is generated above the water table in this first implementation.
    """

    def __init__(self, specification):
        specification = specification or {}

        self.enabled = bool(
            specification.get("enabled", True)
        )

        self.name = (
            specification.get("name")
            or "groundwater"
        )

        self.surface_type = str(
            specification.get("type", "horizontal")
        ).lower()

        gamma_w = specification.get(
            "unit_weight_water"
        )

        if gamma_w is None:
            raise ValueError(
                "Groundwater requires <UnitWeightWater>. "
                "Use units consistent with the model force/length system."
            )

        self.unit_weight_water = float(
            gamma_w
        )

        if self.unit_weight_water <= 0.0:
            raise ValueError(
                "UnitWeightWater must be positive."
            )

        self.elevation = None
        self.points = None

        if self.surface_type == "horizontal":
            elevation = specification.get(
                "elevation"
            )

            if elevation is None:
                raise ValueError(
                    'Groundwater type="horizontal" requires <Elevation>.'
                )

            self.elevation = float(
                elevation
            )

        elif self.surface_type == "polyline":
            points = specification.get(
                "points",
                [],
            )

            if len(points) < 2:
                raise ValueError(
                    'Groundwater type="polyline" requires at least two <Point> entries.'
                )

            array = np.asarray(
                points,
                dtype=float,
            )

            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError(
                    "Groundwater polyline points must be (x, y) pairs."
                )

            order = np.argsort(
                array[:, 0]
            )
            array = array[order]

            if np.any(
                np.diff(array[:, 0]) <= 0.0
            ):
                raise ValueError(
                    "Groundwater polyline x coordinates must be strictly increasing."
                )

            self.points = array

        else:
            raise NotImplementedError(
                f'Unsupported groundwater type "{self.surface_type}". '
                'Supported types: "horizontal", "polyline".'
            )

    def water_level(self, x):
        """Return phreatic-surface elevation y_w at horizontal coordinate x."""
        x = float(x)

        if self.surface_type == "horizontal":
            return self.elevation

        x_values = self.points[:, 0]
        y_values = self.points[:, 1]

        tolerance = 1.0e-12 * max(
            1.0,
            abs(x_values[0]),
            abs(x_values[-1]),
        )

        if x < x_values[0] - tolerance or x > x_values[-1] + tolerance:
            raise ValueError(
                f"Groundwater polyline does not cover x={x:.6g}. "
                f"Its x-range is [{x_values[0]:.6g}, {x_values[-1]:.6g}]."
            )

        x_clamped = min(
            max(x, x_values[0]),
            x_values[-1],
        )

        return float(
            np.interp(
                x_clamped,
                x_values,
                y_values,
            )
        )

    def pressure_head(self, x, y):
        """Return positive hydrostatic pressure head below the phreatic line."""
        if not self.enabled:
            return 0.0

        return max(
            self.water_level(x) - float(y),
            0.0,
        )

    def pore_pressure(self, x, y):
        """Return hydrostatic pore-water pressure u = gamma_w * pressure_head."""
        return (
            self.unit_weight_water
            * self.pressure_head(x, y)
        )


def initialize_hydrostatic_pore_pressure(
    nodes,
    mesh_elements,
    state_manager,
    groundwater_specification,
):
    """
    Initialize hydrostatic pore pressure at every material Gauss point.

    This operation modifies BOTH committed and trial Gauss-point states so the
    hydraulic initial condition survives subsequent reset_trial()/commit()
    operations.

    Important
    ---------
    This first groundwater implementation stores pore pressure only.  It does
    NOT yet alter the mechanical stress field, internal-force vector, or
    constitutive effective stress.  Those couplings are introduced later by
    the K0/effective-stress implementation.
    """
    surface = PhreaticSurface(
        groundwater_specification
    )

    total_points = 0
    submerged_points = 0
    minimum_pressure = None
    maximum_pressure = None

    for element in mesh_elements:
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
            :,
        ]

        shape = element_info[
            "shape"
        ]

        gauss_points, _ = element_info[
            "gauss"
        ]()

        element_id = int(
            element["id"]
        )

        if state_manager.number_of_gauss_points(
            element_id
        ) != len(gauss_points):
            raise ValueError(
                f"Groundwater initialization found a Gauss-point count mismatch "
                f"in element {element_id}."
            )

        for gp_index, q in enumerate(
            gauss_points
        ):
            N = shape(q)
            position = (
                N @ element_coordinates
            )

            x = float(position[0])
            y = float(position[1])

            pore_pressure = surface.pore_pressure(
                x,
                y,
            )

            water_level = surface.water_level(
                x
            )

            pressure_head = surface.pressure_head(
                x,
                y,
            )

            for state in (
                state_manager.get_committed(
                    element_id,
                    gp_index,
                ),
                state_manager.get_trial(
                    element_id,
                    gp_index,
                ),
            ):
                state.pore_pressure = float(
                    pore_pressure
                )
                state.internal_variables[
                    "water_level"
                ] = float(
                    water_level
                )
                state.internal_variables[
                    "pressure_head"
                ] = float(
                    pressure_head
                )

            total_points += 1

            if pore_pressure > 0.0:
                submerged_points += 1

            if minimum_pressure is None:
                minimum_pressure = pore_pressure
                maximum_pressure = pore_pressure
            else:
                minimum_pressure = min(
                    minimum_pressure,
                    pore_pressure,
                )
                maximum_pressure = max(
                    maximum_pressure,
                    pore_pressure,
                )

    if minimum_pressure is None:
        minimum_pressure = 0.0
        maximum_pressure = 0.0

    return {
        "name": surface.name,
        "type": surface.surface_type,
        "unit_weight_water": surface.unit_weight_water,
        "total_gauss_points": total_points,
        "submerged_gauss_points": submerged_points,
        "minimum_pore_pressure": float(minimum_pressure),
        "maximum_pore_pressure": float(maximum_pressure),
    }
