from copy import deepcopy

import numpy as np

from elements.elemtype import get_element_type
from state.gauss_point_state import GaussPointState


def _copy_state_map(
    state_map,
):
    """Deep-copy all element/Gauss-point material states."""
    return {
        element_id: [
            state.copy()
            for state in states
        ]
        for element_id, states in state_map.items()
    }


def _copy_reference_map(
    reference_map,
):
    """Deep-copy the element/Gauss-point activation-strain references."""
    return {
        element_id: [
            np.asarray(
                strain,
                dtype=float,
            ).copy()
            for strain in strains
        ]
        for element_id, strains in reference_map.items()
    }


def _fresh_mechanical_state(
    old_state,
):
    """
    Return a zero mechanical material state while preserving environmental
    quantities that are not part of the constitutive stress history.

    Currently preserved
    -------------------
    - pore_pressure
    - groundwater water_level
    - groundwater pressure_head

    This is appropriate for element birth/death:
    a removed/re-installed material region does not inherit its old stress,
    strain, plastic strain, hardening, or K0 history, while the static
    groundwater field can remain available at that location.
    """
    preserved_internal = {}

    for key in (
        "water_level",
        "pressure_head",
    ):
        if key in old_state.internal_variables:
            preserved_internal[
                key
            ] = deepcopy(
                old_state.internal_variables[
                    key
                ]
            )

    return GaussPointState(
        pore_pressure=float(
            old_state.pore_pressure
        ),
        internal_variables=preserved_internal,
    )


def _copy_tangent_map(
    tangent_map,
):
    """Deep-copy Gauss-point algorithmic tangent matrices."""
    return {
        element_id: [
            None
            if tangent is None
            else np.asarray(
                tangent,
                dtype=float,
            ).reshape(3, 3).copy()
            for tangent in tangents
        ]
        for element_id, tangents in tangent_map.items()
    }


class GaussPointStateManager:
    """
    Manage committed/trial Gauss-point state and construction activity.

    In addition to constitutive state, the manager stores one activation
    strain reference at every Gauss point.

    For an element that exists from the beginning:

        epsilon_ref = 0

    For a newly activated element, GeoPyFEM first gives previously inactive
    nodes a cumulative geometric birth displacement extended from the old
    active interface.  Then:

        epsilon_ref = B @ u_birth

    The constitutive model therefore receives:

        epsilon_material
            = B @ u_current - epsilon_ref

    so a newly installed element is stress-free at its activation stage while
    its nodal U field remains cumulative and geometrically compatible with the
    construction state that already exists below/around it.

    Deactivation
    ------------
    Deactivated elements are removed from stiffness, internal-force and
    self-weight assembly by the solver/assembly active-element filters.

    Their mechanical material history is cleared immediately. If that region
    is activated again later, GeoPyFEM treats it as newly installed material
    and creates a new stress-free activation reference.

    This first implementation therefore represents excavation/removal and
    fresh re-installation/backfill, not temporary hiding with history
    preservation.
    """

    def __init__(
        self,
        mesh_elements,
    ):
        self._committed = (
            self._initialize_zero_states(
                mesh_elements
            )
        )

        self._trial = _copy_state_map(
            self._committed
        )

        self._committed_tangents = {
            element_id: [
                None
                for _ in states
            ]
            for element_id, states in self._committed.items()
        }

        self._trial_tangents = _copy_tangent_map(
            self._committed_tangents
        )

        self._all_element_ids = set(
            self._committed.keys()
        )

        self._active_element_ids = set(
            self._all_element_ids
        )

        self._activation_strain_reference = {
            element_id: [
                np.zeros(
                    3,
                    dtype=float,
                )
                for _ in states
            ]
            for element_id, states in self._committed.items()
        }

    @staticmethod
    def _initialize_zero_states(
        mesh_elements,
    ):
        """Create exactly one state object per integration point."""
        states = {}

        for element in mesh_elements:
            element_info = get_element_type(
                element[
                    "type"
                ]
            )

            if element_info[
                "dimension"
            ] != 2:
                continue

            gauss_rule = element_info[
                "gauss"
            ]

            gauss_points, _ = (
                gauss_rule()
            )

            element_id = int(
                element[
                    "id"
                ]
            )

            if element_id in states:
                raise ValueError(
                    f"Duplicate element ID {element_id} "
                    "while initializing Gauss-point states."
                )

            states[
                element_id
            ] = [
                GaussPointState()
                for _ in gauss_points
            ]

        return states

    @property
    def committed(
        self,
    ):
        return self._committed

    @property
    def trial(
        self,
    ):
        return self._trial

    @property
    def committed_tangents(
        self,
    ):
        return self._committed_tangents

    @property
    def trial_tangents(
        self,
    ):
        return self._trial_tangents

    @property
    def active_element_ids(
        self,
    ):
        return set(
            self._active_element_ids
        )

    @property
    def inactive_element_ids(
        self,
    ):
        return (
            self._all_element_ids
            - self._active_element_ids
        )

    @property
    def number_of_states(
        self,
    ):
        return sum(
            len(states)
            for states in self._committed.values()
        )

    def snapshot_committed(
        self,
    ):
        return _copy_state_map(
            self._committed
        )

    def snapshot_activation_references(
        self,
    ):
        return _copy_reference_map(
            self._activation_strain_reference
        )

    def number_of_gauss_points(
        self,
        element_id,
    ):
        return len(
            self._committed[
                int(
                    element_id
                )
            ]
        )

    def get_committed(
        self,
        element_id,
        gauss_point_index,
    ):
        return self._committed[
            int(
                element_id
            )
        ][
            int(
                gauss_point_index
            )
        ]

    def get_trial(
        self,
        element_id,
        gauss_point_index,
    ):
        return self._trial[
            int(
                element_id
            )
        ][
            int(
                gauss_point_index
            )
        ]

    def set_trial(
        self,
        element_id,
        gauss_point_index,
        state,
    ):
        element_id = int(element_id)
        gauss_point_index = int(gauss_point_index)

        self._trial[element_id][gauss_point_index] = state.copy()
        self._trial_tangents[element_id][gauss_point_index] = None

    def set_trial_response(
        self,
        element_id,
        gauss_point_index,
        response,
    ):
        """Store one complete constitutive response for the trial state."""
        element_id = int(element_id)
        gauss_point_index = int(gauss_point_index)

        self._trial[element_id][gauss_point_index] = (
            response.state.copy()
        )

        self._trial_tangents[element_id][gauss_point_index] = (
            np.asarray(
                response.tangent,
                dtype=float,
            ).reshape(3, 3).copy()
        )

    def set_committed_tangent(
        self,
        element_id,
        gauss_point_index,
        tangent,
    ):
        """Set the tangent associated with an already committed state."""
        element_id = int(element_id)
        gauss_point_index = int(gauss_point_index)

        matrix = np.asarray(
            tangent,
            dtype=float,
        ).reshape(3, 3).copy()

        self._committed_tangents[element_id][gauss_point_index] = matrix
        self._trial_tangents[element_id][gauss_point_index] = matrix.copy()

    def get_activation_strain_reference(
        self,
        element_id,
        gauss_point_index,
    ):
        return np.asarray(
            self._activation_strain_reference[
                int(
                    element_id
                )
            ][
                int(
                    gauss_point_index
                )
            ],
            dtype=float,
        ).copy()

    def is_active(
        self,
        element_id,
    ):
        return (
            int(
                element_id
            )
            in self._active_element_ids
        )

    def reset_trial(
        self,
    ):
        self._trial = _copy_state_map(
            self._committed
        )

        self._trial_tangents = _copy_tangent_map(
            self._committed_tangents
        )

    def revert(
        self,
    ):
        self.reset_trial()

    def commit(
        self,
    ):
        self._committed = _copy_state_map(
            self._trial
        )

        self._committed_tangents = _copy_tangent_map(
            self._trial_tangents
        )

        self._trial = _copy_state_map(
            self._committed
        )

        self._trial_tangents = _copy_tangent_map(
            self._committed_tangents
        )

    def deactivate_elements(
        self,
        element_ids,
    ):
        """
        Remove elements from the active mechanical domain and clear their
        mechanical history.

        Environmental groundwater quantities are retained.
        """
        for element_id in element_ids:
            element_id = int(
                element_id
            )

            if element_id not in self._all_element_ids:
                raise ValueError(
                    f"Cannot deactivate unknown element ID {element_id}."
                )

            self._active_element_ids.discard(
                element_id
            )

            self._committed[
                element_id
            ] = [
                _fresh_mechanical_state(
                    state
                )
                for state in self._committed[
                    element_id
                ]
            ]

            self._trial[
                element_id
            ] = [
                state.copy()
                for state in self._committed[
                    element_id
                ]
            ]

            self._committed_tangents[element_id] = [
                None
                for _ in self._committed[element_id]
            ]
            self._trial_tangents[element_id] = [
                None
                for _ in self._committed[element_id]
            ]

            self._activation_strain_reference[
                element_id
            ] = [
                np.zeros(
                    3,
                    dtype=float,
                )
                for _ in self._committed[
                    element_id
                ]
            ]

    def activate_elements(
        self,
        element_reference_strains,
    ):
        """
        Activate elements as newly installed stress-free material.

        Parameters
        ----------
        element_reference_strains : dict
            Mapping:

                element_id -> [epsilon_ref_gp0, epsilon_ref_gp1, ...]

            where each epsilon_ref is the current B @ u vector at the instant
            of activation.
        """
        for (
            element_id,
            reference_strains,
        ) in element_reference_strains.items():
            element_id = int(
                element_id
            )

            if element_id not in self._all_element_ids:
                raise ValueError(
                    f"Cannot activate unknown element ID {element_id}."
                )

            expected = len(
                self._committed[
                    element_id
                ]
            )

            if len(
                reference_strains
            ) != expected:
                raise ValueError(
                    f"Element {element_id} requires {expected} activation "
                    f"strain references but received "
                    f"{len(reference_strains)}."
                )

            # Reactivation means fresh construction material. Clear old
            # mechanical history but preserve groundwater quantities.
            self._committed[
                element_id
            ] = [
                _fresh_mechanical_state(
                    state
                )
                for state in self._committed[
                    element_id
                ]
            ]

            self._trial[
                element_id
            ] = [
                state.copy()
                for state in self._committed[
                    element_id
                ]
            ]

            self._committed_tangents[element_id] = [
                None
                for _ in self._committed[element_id]
            ]
            self._trial_tangents[element_id] = [
                None
                for _ in self._committed[element_id]
            ]

            self._activation_strain_reference[
                element_id
            ] = [
                np.asarray(
                    strain,
                    dtype=float,
                ).reshape(
                    3
                ).copy()
                for strain in reference_strains
            ]

            self._active_element_ids.add(
                element_id
            )
