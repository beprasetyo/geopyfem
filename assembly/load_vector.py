import numpy as np

from elements.elemtype import get_element_type


def assemble_load_vector(
    nodes,
    mesh_elements,
    physical_group_elements,
    material_models,
    loads,
    gravity=None,
    units=None,
    active_element_ids=None,
    ndof_node=2,
    verbose=True,
):
    """
    Assemble the global external load vector.

    The current implementation supports:

        1. Boundary traction
               fe = integral_Gamma (N^T t dGamma)

        2. Gravity / self weight
               fe = integral_Omega (N_u^T b dOmega)

           where
               b = [rho*gx, rho*gy]^T

    Parameters
    ----------
    nodes : ndarray, shape (nnode, 2)
        Global nodal coordinates.

    mesh_elements : list of dict
        2-D domain elements returned by gmsh_parser.py.

    physical_group_elements : dict
        Elements grouped by Gmsh Physical Group name.

    material_models : dict
        Material-model objects indexed by material region.  For gravity,
        each material model must expose:

            material.density

        Density is interpreted as kg/m^3.

    loads : list of dict
        Boundary load definitions returned by problem_reader.py.

    gravity : dict or None
        Example:
            {
                "enabled": True,
                "gx": 0.0,
                "gy": -9.81,
            }

        gx and gy are acceleration components in m/s^2.

    units : dict or None
        Global problem units from XML, for example:
            {
                "length": "m",
                "force": "kN",
                "time": "s",
            }

        Gravity currently supports length="m" and force="N" or "kN".

    ndof_node : int, optional
        Current 2-D continuum solver uses 2 DOFs/node: ux, uy.

    verbose : bool, optional
        Print short assembly summaries.

    Returns
    -------
    f : ndarray
        Global external load vector.

    Notes
    -----
    In 2-D plane strain this is a per-unit-out-of-plane-thickness formulation.
    """
    if ndof_node != 2:
        raise NotImplementedError(
            "The current load-vector assembly supports 2 DOFs per node."
        )

    num_nodes = len(nodes)
    f = np.zeros(ndof_node * num_nodes)

    active_element_ids = (
        None
        if active_element_ids is None
        else {
            int(element_id)
            for element_id in active_element_ids
        }
    )

    active_node_ids = None

    if active_element_ids is not None:
        active_node_ids = set()

        for element in mesh_elements:
            if int(
                element["id"]
            ) not in active_element_ids:
                continue

            active_node_ids.update(
                int(node_id)
                for node_id in element[
                    "connectivity"
                ]
            )

    # =====================================================================
    # 1. BOUNDARY TRACTION
    # =====================================================================
    for load in loads:
        group_name = load["group"]
        load_type = load["type"]

        if group_name not in physical_group_elements:
            raise ValueError(
                f'Load group "{group_name}" from XML does not exist '
                "in the Gmsh mesh."
            )

        time_function = load.get("time_function")

        if time_function not in (None, "constant"):
            raise NotImplementedError(
                "Only constant loads are supported in the current static solver."
            )

        if load_type != "traction":
            raise NotImplementedError(
                f'Load type "{load_type}" is not yet supported. '
                'The current implementation supports type="traction".'
            )

        tx = float(load["tx"])
        ty = float(load["ty"])

        for boundary_element in physical_group_elements[group_name]:
            if boundary_element["dimension"] != 1:
                continue

            boundary_type = boundary_element["type"]
            boundary_info = get_element_type(boundary_type)

            if boundary_info["dimension"] != 1:
                raise ValueError(
                    f'Boundary element "{boundary_type}" is not a 1-D element.'
                )

            c_edge = boundary_element["connectivity"]
            nnode_edge = boundary_info["nnode"]

            if len(c_edge) != nnode_edge:
                raise ValueError(
                    f'Boundary element "{boundary_type}" contains '
                    f'{len(c_edge)} nodes instead of {nnode_edge}.'
                )

            if active_node_ids is not None:
                inactive_boundary_nodes = [
                    int(node_id)
                    for node_id in c_edge
                    if int(node_id) not in active_node_ids
                ]

                if inactive_boundary_nodes:
                    raise ValueError(
                        f'Active traction load "{load["name"]}" uses boundary '
                        f'group "{group_name}", but boundary element contains '
                        "nodes that are not connected to the active mechanical "
                        f'domain: {inactive_boundary_nodes}. Activate the '
                        "corresponding region before activating this load."
                    )

            shape = boundary_info["shape"]
            gradshape = boundary_info["gradshape"]
            gauss_rule = boundary_info["gauss"]

            gauss_points, gauss_weights = gauss_rule()
            x_edge = nodes[c_edge, :]

            for q, w in zip(gauss_points, gauss_weights):
                N = shape(q)
                dN = gradshape(q)

                tangent = np.dot(dN, x_edge).reshape(-1)
                J_edge = np.linalg.norm(tangent)

                if J_edge <= 0.0:
                    raise ValueError(
                        f'Boundary element "{boundary_type}" in group '
                        f'"{group_name}" has non-positive boundary Jacobian.'
                    )

                for a, node_id in enumerate(c_edge):
                    f[2*node_id] += N[a] * tx * J_edge * w
                    f[2*node_id + 1] += N[a] * ty * J_edge * w

        if verbose:
            print(
                f'  traction "{load["name"]}": '
                f'group={group_name}, tx={tx}, ty={ty}'
            )

    # =====================================================================
    # 2. GRAVITY / SELF WEIGHT
    # =====================================================================
    gravity = gravity or {}

    if gravity.get("enabled", False):
        gx = gravity.get("gx")
        gy = gravity.get("gy")

        gx = 0.0 if gx is None else float(gx)
        gy = -9.81 if gy is None else float(gy)

        # rho [kg/m^3] * g [m/s^2] gives body force [N/m^3].
        # Convert N to the global force unit declared in XML.
        force_scale = _newton_to_problem_force_scale(units)

        f_gravity = np.zeros_like(f)

        for element in mesh_elements:
            element_id = int(
                element["id"]
            )

            if (
                active_element_ids is not None
                and element_id not in active_element_ids
            ):
                continue

            element_type = element["type"]
            element_info = get_element_type(element_type)

            if element_info["dimension"] != 2:
                continue

            c = element["connectivity"]
            nnode = element_info["nnode"]

            if len(c) != nnode:
                raise ValueError(
                    f'Element {element["id"]} is {element_type} but contains '
                    f'{len(c)} nodes instead of {nnode}.'
                )

            material_region = element["material"]

            if material_region not in material_models:
                raise ValueError(
                    f'Element {element["id"]} uses material region '
                    f'"{material_region}", but no material model exists.'
                )

            material = material_models[material_region]
            density = getattr(material, "density", None)

            if density is None:
                raise ValueError(
                    f'Gravity is enabled, but material region '
                    f'"{material_region}" does not define density.'
                )

            density = float(density)

            shape = element_info["shape"]
            gradshape = element_info["gradshape"]
            gauss_rule = element_info["gauss"]

            gauss_points, gauss_weights = gauss_rule()
            xIe = nodes[c, :]

            # Body force per unit volume in the declared force unit / m^3.
            body_force = np.array([
                density * gx * force_scale,
                density * gy * force_scale,
            ])

            for q, w in zip(gauss_points, gauss_weights):
                N = shape(q)
                dN = gradshape(q)

                J = np.dot(dN, xIe)
                detJ = np.linalg.det(J)

                if detJ <= 0.0:
                    raise ValueError(
                        f'Element {element["id"]} has non-positive Jacobian '
                        f'det(J)={detJ:.6e} during gravity assembly.'
                    )

                for a, node_id in enumerate(c):
                    f_gravity[2*node_id] += (
                        N[a] * body_force[0] * detJ * w
                    )

                    f_gravity[2*node_id + 1] += (
                        N[a] * body_force[1] * detJ * w
                    )

        f += f_gravity

        if verbose:
            force_unit = (units or {}).get("force", "N")
            print(
                f"  gravity/self weight: gx={gx}, gy={gy} m/s^2, "
                f"sum(Fx)={np.sum(f_gravity[0::2]):.6g} {force_unit}, "
                f"sum(Fy)={np.sum(f_gravity[1::2]):.6g} {force_unit}"
            )

    return f


def _newton_to_problem_force_scale(units):
    """
    Convert a force expressed in N to the force unit declared in the XML.

    Density remains kg/m^3 and gravity remains m/s^2.

    Supported combinations
    ----------------------
    Length:
        m

    Force:
        N
        kN
    """
    units = units or {}

    length_unit = str(
        units.get("length", "m")
    ).strip().lower()

    force_unit = str(
        units.get("force", "N")
    ).strip().lower()

    if length_unit != "m":
        raise NotImplementedError(
            "Gravity loading currently assumes Length='m' because density "
            "is interpreted as kg/m^3 and gravity as m/s^2."
        )

    if force_unit == "n":
        return 1.0

    if force_unit == "kn":
        return 1.0e-3

    raise NotImplementedError(
        f'Gravity loading does not yet support Force="{units.get("force")}". '
        'Currently supported: "N" and "kN".'
    )
