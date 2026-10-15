"""Reusable small-strain element kinematics cache.

GeoPyFEM currently uses a fixed reference mesh with small-strain kinematics.
For that formulation the following quantities are invariant during Newton
iterations and construction stages as long as the mesh topology/coordinates do
not change:

- element global DOF map,
- Gauss-point natural coordinates and weights,
- B matrix,
- det(J),
- global shape-function gradients.

Computing these quantities once avoids repeatedly rebuilding Jacobians and
B-matrices during constitutive updates, internal-force assembly, and tangent
assembly.
"""

from dataclasses import dataclass

import numpy as np

from elements.elemtype import get_element_type
from elements.kinematics import build_B_matrix


def _element_dof_indices(connectivity, ndof_node=2):
    connectivity = np.asarray(connectivity, dtype=int).reshape(-1)
    dofs = np.empty(connectivity.size * int(ndof_node), dtype=int)
    for local_dof in range(int(ndof_node)):
        dofs[local_dof::int(ndof_node)] = int(ndof_node) * connectivity + local_dof
    return dofs


def _readonly(array, dtype=None):
    value = np.asarray(array, dtype=dtype).copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class GaussPointKinematics:
    natural_coordinates: np.ndarray
    weight: float
    B: np.ndarray
    detJ: float
    dN_global: np.ndarray


@dataclass(frozen=True)
class ElementKinematics:
    element_id: int
    element_type: str
    connectivity: np.ndarray
    dofs: np.ndarray
    gauss_points: tuple

    @property
    def number_of_gauss_points(self):
        return len(self.gauss_points)

    @property
    def number_of_nodes(self):
        return int(self.connectivity.size)

    @property
    def number_of_dofs(self):
        return int(self.dofs.size)


def build_element_kinematics(nodes, element, ndof_node=2):
    """Build immutable kinematics for one 2-D continuum element."""
    element_info = get_element_type(element["type"])
    if element_info["dimension"] != 2:
        return None

    element_ndof_node = int(element_info["ndof_node"])
    if element_ndof_node != int(ndof_node):
        raise NotImplementedError(
            f'Element "{element["type"]}" uses {element_ndof_node} DOFs/node, '
            f"while the kinematics cache is configured for {ndof_node}."
        )

    connectivity = np.asarray(element["connectivity"], dtype=int).reshape(-1)
    nnode = int(element_info["nnode"])
    if connectivity.size != nnode:
        raise ValueError(
            f'Element {element["id"]} is {element["type"]}, but its '
            f"connectivity contains {connectivity.size} nodes instead of "
            f"{nnode}."
        )

    coordinates = np.asarray(nodes, dtype=float)[connectivity, :]
    gradshape = element_info["gradshape"]
    gauss_points, gauss_weights = element_info["gauss"]()

    gp_kinematics = []
    for q, weight in zip(gauss_points, gauss_weights):
        B, detJ, dN_global = build_B_matrix(
            element_coordinates=coordinates,
            gradshape=gradshape,
            natural_coordinates=q,
        )
        gp_kinematics.append(
            GaussPointKinematics(
                natural_coordinates=_readonly(q, dtype=float),
                weight=float(weight),
                B=_readonly(B, dtype=float),
                detJ=float(detJ),
                dN_global=_readonly(dN_global, dtype=float),
            )
        )

    return ElementKinematics(
        element_id=int(element["id"]),
        element_type=str(element["type"]),
        connectivity=_readonly(connectivity, dtype=int),
        dofs=_readonly(
            _element_dof_indices(connectivity, ndof_node=ndof_node),
            dtype=int,
        ),
        gauss_points=tuple(gp_kinematics),
    )


class ElementKinematicsCache:
    """Immutable kinematic data for all 2-D continuum elements in one mesh."""

    def __init__(self, nodes, mesh_elements, ndof_node=2):
        self.ndof_node = int(ndof_node)
        self.number_of_nodes = int(len(nodes))
        self._elements = {}

        total_gauss_points = 0
        for element in mesh_elements:
            kinematics = build_element_kinematics(
                nodes=nodes,
                element=element,
                ndof_node=self.ndof_node,
            )
            if kinematics is None:
                continue

            element_id = kinematics.element_id
            if element_id in self._elements:
                raise ValueError(
                    f"Duplicate element ID {element_id} in kinematics cache."
                )

            self._elements[element_id] = kinematics
            total_gauss_points += kinematics.number_of_gauss_points

        self.number_of_elements = len(self._elements)
        self.number_of_gauss_points = int(total_gauss_points)

    def __contains__(self, element_id):
        return int(element_id) in self._elements

    def __getitem__(self, element_id):
        element_id = int(element_id)
        try:
            return self._elements[element_id]
        except KeyError as error:
            raise KeyError(
                f"Element {element_id} is not present in the 2-D kinematics cache."
            ) from error

    def get(self, element_id, default=None):
        return self._elements.get(int(element_id), default)

    def element_ids(self):
        return tuple(self._elements.keys())


def get_element_kinematics(
    nodes,
    element,
    ndof_node=2,
    kinematics_cache=None,
):
    """Return cached element kinematics or build a backward-compatible copy."""
    if kinematics_cache is not None:
        return kinematics_cache.get(int(element["id"]))
    return build_element_kinematics(
        nodes=nodes,
        element=element,
        ndof_node=ndof_node,
    )
