"""
GeoPyFEM constitutive-model template.

Copy this file to e.g. ``mohr_coulomb.py``, rename the class, implement the
local stress-integration algorithm, and register the class in
``material_model/materialtype.py``.

This file is intentionally NOT registered as a usable material model.
"""

import numpy as np

from material_model.constitutive import (
    ConstitutiveModel,
    ConstitutiveResponse,
)


class NewSoilModelTemplate(ConstitutiveModel):
    model_name = "NewSoilModelTemplate"

    def __init__(self, parameters, formulation="plane_strain"):
        self.parameters = parameters
        self.formulation = formulation

        # Read and validate model parameters here.
        # Example:
        # self.E = float(parameters["E"])
        # self.nu = float(parameters["nu"])
        # self.cohesion = float(parameters["cohesion"])
        # self.friction_angle = float(parameters["friction_angle"])

        density = parameters.get("density")
        self.density = (
            None
            if density is None
            else float(density)
        )

    def integrate(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
        context=None,
    ):
        """
        Perform ONE local constitutive integration.

        Inputs roughly correspond to a UMAT-style contract:

            total_strain_2d      ~ current STRAN
            strain_increment_2d  ~ DSTRAN
            committed_state      ~ old STRESS + STATEV
            context              ~ NOEL/NPT/KSTEP/KINC/etc.

        Return:

            response.state.effective_stress  ~ new STRESS
            response.state.internal_variables~ new STATEV
            response.tangent                 ~ DDSDDE

        GeoPyFEM constitutive stress is EFFECTIVE stress. Pore pressure must
        not be subtracted here; the global equilibrium layer constructs total
        stress separately.
        """
        total_strain_2d = np.asarray(
            total_strain_2d,
            dtype=float,
        ).reshape(3)

        strain_increment_2d = np.asarray(
            strain_increment_2d,
            dtype=float,
        ).reshape(3)

        trial_state = committed_state.copy()

        # -------------------------------------------------------------
        # MODEL-SPECIFIC LOCAL INTEGRATION GOES HERE
        # -------------------------------------------------------------
        # Typical nonlinear soil-model sequence:
        #
        # 1. elastic predictor using strain_increment_2d
        # 2. evaluate yield function in EFFECTIVE stress
        # 3. if elastic: accept predictor
        # 4. if plastic: return-map to the yield surface
        # 5. update plastic strain / hardening variables / other STATEV
        # 6. calculate the algorithmic consistent tangent C_alg
        #
        # trial_state.effective_stress[:] = ...
        # trial_state.plastic_strain[:] = ...
        # trial_state.internal_variables["..."] = ...
        # C_alg = ...

        raise NotImplementedError(
            "Implement the local stress integration and algorithmic tangent."
        )

        # Example return structure after implementation:
        # return ConstitutiveResponse(
        #     state=trial_state,
        #     tangent=C_alg,
        #     diagnostics={
        #         "yielded": bool(trial_state.yielded),
        #         "yield_function": ...,
        #     },
        # )
