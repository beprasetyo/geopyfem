from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class MaterialContext:
    """
    Read-only metadata supplied by the global solver to one constitutive call.

    The constitutive model must not use this object to reach back into the FEM
    solver.  It is only contextual information, analogous to the bookkeeping
    arguments passed to a UMAT (element number, integration-point number,
    step/increment number, time, etc.).

    Most rate-independent models can ignore most fields.
    """

    element_id: Optional[int] = None
    gauss_point_index: Optional[int] = None
    stage_id: Optional[int] = None
    stage_name: Optional[str] = None
    load_factor: Optional[float] = None
    increment_number: Optional[int] = None
    iteration_number: Optional[int] = None
    load_path_kind: Optional[str] = None
    time: Optional[float] = None
    dt: Optional[float] = None
    coordinates: Optional[tuple] = None


@dataclass
class ConstitutiveResponse:
    """
    Complete local constitutive result returned to the GeoPyFEM core.

    This is the GeoPyFEM analogue of the essential UMAT outputs:

        state.effective_stress  ~ STRESS   (GeoPyFEM uses effective stress)
        tangent                 ~ DDSDDE
        state.internal_variables~ STATEV

    The global solver/assembly does not need to know which constitutive model
    produced the response.
    """

    state: Any
    tangent: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    suggested_step_scale: Optional[float] = None

    def __post_init__(self):
        self.tangent = np.asarray(
            self.tangent,
            dtype=float,
        ).reshape(3, 3).copy()

        if not np.all(np.isfinite(self.tangent)):
            raise ValueError(
                "ConstitutiveResponse tangent contains non-finite values."
            )

        if not hasattr(self.state, "effective_stress"):
            raise TypeError(
                "ConstitutiveResponse.state must expose effective_stress."
            )

        stress = np.asarray(
            self.state.effective_stress,
            dtype=float,
        )

        if stress.shape != (6,):
            raise ValueError(
                "ConstitutiveResponse effective stress must have six Voigt "
                "components [xx, yy, zz, xy, yz, xz]."
            )

        if not np.all(np.isfinite(stress)):
            raise ValueError(
                "ConstitutiveResponse effective stress contains non-finite "
                "values."
            )

        self.diagnostics = deepcopy(
            self.diagnostics
        )

        if self.suggested_step_scale is not None:
            scale = float(
                self.suggested_step_scale
            )

            if scale <= 0.0:
                raise ValueError(
                    "suggested_step_scale must be positive when supplied."
                )

            self.suggested_step_scale = scale

    @property
    def effective_stress(self):
        return np.asarray(
            self.state.effective_stress,
            dtype=float,
        ).copy()

    def copy(self):
        return ConstitutiveResponse(
            state=self.state.copy(),
            tangent=self.tangent.copy(),
            diagnostics=deepcopy(
                self.diagnostics
            ),
            suggested_step_scale=(
                None
                if self.suggested_step_scale is None
                else float(self.suggested_step_scale)
            ),
        )


class ConstitutiveModel(ABC):
    """
    Formal GeoPyFEM constitutive-model contract.

    A new nonlinear material model only has to implement ``integrate`` and be
    registered in ``material_model/materialtype.py``.

    Inputs
    ------
    total_strain_2d
        Current material strain [exx, eyy, gamma_xy].
    strain_increment_2d
        Increment from the committed state to the current trial state.
    committed_state
        Read-only converged state from the previous accepted load increment.
    context
        Optional solver metadata.  Rate-independent models may ignore it.

    Output
    ------
    ConstitutiveResponse
        New trial state + algorithmic tangent + optional diagnostics.

    Important
    ---------
    GeoPyFEM soil constitutive models evolve EFFECTIVE stress.  Pore pressure
    stays outside the constitutive model and total stress is assembled by the
    global equilibrium layer.
    """

    model_name = "ConstitutiveModel"

    @abstractmethod
    def integrate(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
        context=None,
    ):
        raise NotImplementedError

    def update_state(
        self,
        total_strain_2d,
        committed_state,
    ):
        """
        Backward-compatible adapter for older GeoPyFEM call sites.

        New solver code must use ``integrate`` so stress and tangent are
        generated by the same local constitutive operation.
        """
        total_strain_2d = np.asarray(
            total_strain_2d,
            dtype=float,
        ).reshape(3)

        strain_increment_2d = (
            total_strain_2d
            - committed_state.strain_2d
        )

        response = self.integrate(
            total_strain_2d=total_strain_2d,
            strain_increment_2d=strain_increment_2d,
            committed_state=committed_state,
            context=MaterialContext(),
        )

        return response.state
