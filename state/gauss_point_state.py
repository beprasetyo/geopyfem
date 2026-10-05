from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np


# =============================================================================
# GeoPyFEM stress convention
# =============================================================================
#
# Sign convention: tension positive.
# Positive pore-water pressure u is compressive.
#
# Soil constitutive state is EFFECTIVE stress:
#
#     sigma_total = sigma_effective - u I
#     sigma_effective = sigma_total + u I
#
# Full 3-D Voigt ordering:
#
#     stress = [xx, yy, zz, xy, yz, xz]
#     strain = [exx, eyy, ezz, gamma_xy, gamma_yz, gamma_xz]
#
# Engineering shear strains are used.
# =============================================================================

PORE_PRESSURE_DIRECTION_VOIGT6 = np.array(
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    dtype=float,
)


def plane_strain_2d_to_voigt6(strain_2d):
    strain_2d = np.asarray(strain_2d, dtype=float).reshape(-1)
    if strain_2d.size != 3:
        raise ValueError(
            "plane_strain_2d_to_voigt6() requires "
            "[epsilon_xx, epsilon_yy, gamma_xy]."
        )
    return np.array([
        strain_2d[0], strain_2d[1], 0.0,
        strain_2d[2], 0.0, 0.0,
    ], dtype=float)


def stress_tensor_to_voigt6(stress_tensor):
    stress_tensor = np.asarray(stress_tensor, dtype=float)
    if stress_tensor.shape != (3, 3):
        raise ValueError(
            "stress_tensor_to_voigt6() requires a 3x3 tensor."
        )
    return np.array([
        stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[2, 2],
        stress_tensor[0, 1], stress_tensor[1, 2], stress_tensor[0, 2],
    ], dtype=float)


def stress_voigt6_to_tensor(stress):
    stress = np.asarray(stress, dtype=float).reshape(-1)
    if stress.size != 6:
        raise ValueError(
            "stress_voigt6_to_tensor() requires six stress components."
        )
    sxx, syy, szz, txy, tyz, txz = stress
    return np.array([
        [sxx, txy, txz],
        [txy, syy, tyz],
        [txz, tyz, szz],
    ], dtype=float)


def strain_voigt6_to_tensor(strain):
    strain = np.asarray(strain, dtype=float).reshape(-1)
    if strain.size != 6:
        raise ValueError(
            "strain_voigt6_to_tensor() requires six strain components."
        )
    exx, eyy, ezz, gxy, gyz, gxz = strain
    return np.array([
        [exx, 0.5*gxy, 0.5*gxz],
        [0.5*gxy, eyy, 0.5*gyz],
        [0.5*gxz, 0.5*gyz, ezz],
    ], dtype=float)


def total_stress_from_effective(effective_stress, pore_pressure):
    """Return tension-positive total stress from effective stress and u."""
    effective_stress = np.asarray(
        effective_stress,
        dtype=float,
    ).reshape(6)
    return (
        effective_stress
        - float(pore_pressure) * PORE_PRESSURE_DIRECTION_VOIGT6
    )


def effective_stress_from_total(total_stress, pore_pressure):
    """Return tension-positive effective stress from total stress and u."""
    total_stress = np.asarray(
        total_stress,
        dtype=float,
    ).reshape(6)
    return (
        total_stress
        + float(pore_pressure) * PORE_PRESSURE_DIRECTION_VOIGT6
    )


@dataclass
class GaussPointState:
    """
    Material state at one integration point.

    Fundamental stress contract
    ---------------------------
    ``effective_stress`` is the constitutive stress state.  Pore pressure is
    stored independently in ``pore_pressure``.  Total Cauchy stress is derived
    whenever equilibrium/output needs it:

        sigma_total = sigma_effective - u I

    The legacy name ``state.stress`` remains as a backward-compatible alias
    for ``state.effective_stress`` so existing constitutive models continue to
    run while new soil models can use the explicit name.
    """

    strain: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=float)
    )

    effective_stress: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=float)
    )

    plastic_strain: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=float)
    )

    equivalent_plastic_strain: float = 0.0
    pore_pressure: float = 0.0
    yielded: bool = False
    internal_variables: dict = field(default_factory=dict)

    def copy(self):
        return GaussPointState(
            strain=np.asarray(self.strain, dtype=float).copy(),
            effective_stress=np.asarray(
                self.effective_stress,
                dtype=float,
            ).copy(),
            plastic_strain=np.asarray(
                self.plastic_strain,
                dtype=float,
            ).copy(),
            equivalent_plastic_strain=float(
                self.equivalent_plastic_strain
            ),
            pore_pressure=float(self.pore_pressure),
            yielded=bool(self.yielded),
            internal_variables=deepcopy(self.internal_variables),
        )

    # -----------------------------------------------------------------
    # Backward-compatible constitutive stress alias.
    # -----------------------------------------------------------------
    @property
    def stress(self):
        """Legacy alias for effective_stress."""
        return self.effective_stress

    @stress.setter
    def stress(self, value):
        self.effective_stress[:] = np.asarray(
            value,
            dtype=float,
        ).reshape(6)

    def set_plane_strain(self, strain_2d):
        self.strain[:] = plane_strain_2d_to_voigt6(strain_2d)

    def set_effective_stress_tensor(self, stress_tensor):
        self.effective_stress[:] = stress_tensor_to_voigt6(stress_tensor)

    def set_stress_tensor(self, stress_tensor):
        """Backward-compatible alias: constitutive stress is effective."""
        self.set_effective_stress_tensor(stress_tensor)

    @property
    def strain_tensor(self):
        return strain_voigt6_to_tensor(self.strain)

    @property
    def effective_stress_tensor(self):
        return stress_voigt6_to_tensor(self.effective_stress)

    @property
    def total_stress(self):
        return total_stress_from_effective(
            self.effective_stress,
            self.pore_pressure,
        )

    @property
    def total_stress_tensor(self):
        return stress_voigt6_to_tensor(self.total_stress)

    @property
    def stress_tensor(self):
        """Legacy alias for the constitutive/effective stress tensor."""
        return self.effective_stress_tensor

    @property
    def plastic_strain_tensor(self):
        return strain_voigt6_to_tensor(self.plastic_strain)

    @property
    def effective_stress_2d(self):
        return np.array([
            self.effective_stress[0],
            self.effective_stress[1],
            self.effective_stress[3],
        ], dtype=float)

    @property
    def total_stress_2d(self):
        sigma = self.total_stress
        return np.array([
            sigma[0], sigma[1], sigma[3]
        ], dtype=float)

    @property
    def stress_2d(self):
        """Legacy alias for effective_stress_2d."""
        return self.effective_stress_2d

    @property
    def strain_2d(self):
        return np.array([
            self.strain[0],
            self.strain[1],
            self.strain[3],
        ], dtype=float)
