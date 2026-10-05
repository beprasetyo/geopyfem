import numpy as np

from assembly.internal_force import assemble_internal_force


def build_excavation_release_path(
    nodes,
    mesh_elements,
    state_manager,
    active_element_ids,
    f_physical_target,
    free_dofs,
):
    """
    Build the temporary holding-force state used to release excavation stress.

    Parameters
    ----------
    nodes, mesh_elements
        Current GeoPyFEM mesh data.
    state_manager
        Contains the committed stress state from the previously converged
        construction stage. Elements selected for excavation may already have
        been marked inactive; only ``active_element_ids`` are assembled here.
    active_element_ids
        Remaining mechanical domain after the excavation selection has been
        removed.
    f_physical_target
        External-load vector that should remain after excavation is complete:
        gravity of remaining elements + all still-active boundary loads.
    free_dofs
        Current free mechanical DOFs after inactive-only nodes have been
        suppressed.

    Returns
    -------
    dict
        ``f_external_start`` is an equilibrium-preserving numerical start
        vector. ``holding_force`` is gradually released to zero as the normal
        GeoPyFEM load factor advances from 0 to 1.

    Notes
    -----
    At the old displacement/state, after the excavation elements are removed,
    the remaining domain has the committed internal force

        f_int_remaining.

    We introduce a temporary numerical force only on FREE DOFs:

        f_hold = f_int_remaining - f_physical_target

    Therefore

        f_start = f_physical_target + f_hold

    satisfies free-DOF equilibrium exactly at lambda = 0. The standard staged
    load path then gives

        f(lambda) = f_start + lambda (f_target - f_start)
                  = f_physical_target + (1-lambda) f_hold.

    At lambda = 1 the temporary holding force is zero and only the actual
    post-excavation physical loads remain.

    No artificial force is inserted on prescribed/suppressed DOFs. This keeps
    the holding load confined to the equilibrium equations that are actually
    solved and avoids polluting physical support-load bookkeeping.
    """
    f_physical_target = np.asarray(
        f_physical_target,
        dtype=float,
    ).copy()

    free_dofs = np.asarray(
        free_dofs,
        dtype=int,
    )

    f_internal_remaining = assemble_internal_force(
        nodes=nodes,
        mesh_elements=mesh_elements,
        gauss_states=state_manager.committed,
        active_element_ids=active_element_ids,
    )

    holding_force = np.zeros_like(
        f_physical_target
    )

    holding_force[
        free_dofs
    ] = (
        f_internal_remaining[
            free_dofs
        ]
        - f_physical_target[
            free_dofs
        ]
    )

    f_external_start = (
        f_physical_target
        + holding_force
    )

    initial_residual = (
        f_external_start
        - f_internal_remaining
    )

    return {
        "f_external_start": f_external_start,
        "f_physical_target": f_physical_target,
        "holding_force": holding_force,
        "f_internal_remaining_start": f_internal_remaining,
        "holding_force_norm": float(
            np.linalg.norm(
                holding_force[
                    free_dofs
                ]
            )
        ),
        "initial_free_residual_norm": float(
            np.linalg.norm(
                initial_residual[
                    free_dofs
                ]
            )
        ),
    }
