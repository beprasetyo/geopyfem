def get_fixed_dofs(
    physical_groups,
    boundaries,
    ndof_node=2,
    verbose=True,
):
    """
    Convert XML displacement boundary conditions into global constrained DOFs.

    Parameters
    ----------
    physical_groups : dict
        Gmsh Physical Group names mapped to zero-based node IDs.

    boundaries : list of dict
        Boundary-condition definitions returned by problem_reader.py.

    ndof_node : int, optional
        Number of global DOFs per node.
        Current 2-D solver uses:
            DOF 0 -> ux
            DOF 1 -> uy

    verbose : bool, optional
        Print a summary of each boundary condition.

    Returns
    -------
    fixed_dofs : list of int
        Sorted global DOF numbers constrained to zero displacement.

    Notes
    -----
    Current XML syntax supports:

        fixed
        free

    for ux and uy.

    Future prescribed non-zero displacement can be added here without
    changing main.py.
    """
    if ndof_node != 2:
        raise NotImplementedError(
            "The current boundary-condition module supports 2 DOFs per node."
        )

    fixed_dofs = set()

    for bc in boundaries:
        group_name = bc["group"]

        if group_name not in physical_groups:
            raise ValueError(
                f'Boundary group "{group_name}" from XML does not exist '
                "in the Gmsh mesh."
            )

        ux_condition = bc["ux"].lower()
        uy_condition = bc["uy"].lower()

        if ux_condition not in ("fixed", "free"):
            raise ValueError(
                f'Unsupported ux boundary condition "{bc["ux"]}". '
                'Use "fixed" or "free".'
            )

        if uy_condition not in ("fixed", "free"):
            raise ValueError(
                f'Unsupported uy boundary condition "{bc["uy"]}". '
                'Use "fixed" or "free".'
            )

        boundary_nodes = physical_groups[group_name]

        for node_id in boundary_nodes:
            if ux_condition == "fixed":
                fixed_dofs.add(
                    ndof_node * node_id
                )

            if uy_condition == "fixed":
                fixed_dofs.add(
                    ndof_node * node_id + 1
                )

        if verbose:
            print(
                f'  boundary "{bc["name"]}": '
                f'group={group_name}, '
                f'ux={ux_condition}, '
                f'uy={uy_condition}, '
                f'nodes={len(boundary_nodes)}'
            )

    return sorted(fixed_dofs)


def apply_displacement_boundary_conditions(
    K,
    f,
    physical_groups,
    boundaries,
    ndof_node=2,
    verbose=True,
):
    """
    Apply zero displacement constraints to the global linear system.

    Parameters
    ----------
    K : ndarray
        Global stiffness matrix.

    f : ndarray
        Global external load vector.

    physical_groups : dict
        Gmsh Physical Groups mapped to node IDs.

    boundaries : list of dict
        XML displacement boundary conditions.

    ndof_node : int, optional
        Number of DOFs per node.

    verbose : bool, optional
        Print boundary-condition summaries.

    Returns
    -------
    K : ndarray
        Stiffness matrix after constraint application.

    f : ndarray
        Load vector after constraint application.

    fixed_dofs : list of int
        Constrained global DOFs.

    Notes
    -----
    This intentionally preserves the row-replacement approach used in the
    original educational FEM code:

        K[dof, :] = 0
        K[dof, dof] = 1
        f[dof] = 0

    It is suitable for the current zero-prescribed-displacement linear solver.

    A future implementation may use symmetric row/column modification,
    elimination, penalty methods, or more general prescribed displacements.
    """
    fixed_dofs = get_fixed_dofs(
        physical_groups,
        boundaries,
        ndof_node=ndof_node,
        verbose=verbose,
    )

    for dof in fixed_dofs:
        K[dof, :] = 0.0
        K[dof, dof] = 1.0
        f[dof] = 0.0

    return K, f, fixed_dofs
