import numpy as np

from material_model.constitutive import (
    ConstitutiveModel,
    ConstitutiveResponse,
)


class LinearElastic(ConstitutiveModel):
    """
    Small-strain isotropic linear-elastic material model.

    The current FEM solver uses the tangent constitutive matrix returned by
    tangent_matrix().  The class interface is intentionally kept simple so
    future constitutive models can expose the same method.

    Parameters
    ----------
    parameters : dict
        Material parameters read from XML.
        Required:
            E
            nu
    formulation : str
        Currently supported:
            plane_strain
    """

    model_name = "LinearElastic"

    def __init__(self, parameters, formulation="plane_strain"):
        self.parameters = parameters
        self.formulation = formulation

        if "E" not in parameters:
            raise ValueError("LinearElastic material requires parameter E.")

        if "nu" not in parameters:
            raise ValueError("LinearElastic material requires parameter nu.")

        self.E = float(parameters["E"])
        self.nu = float(parameters["nu"])

        # Mass density is kept in SI mass units (kg/m^3), independent of
        # whether the global force unit is N or kN.  Gravity-load assembly
        # performs the required force-unit conversion.
        density = parameters.get("density")
        self.density = None if density is None else float(density)

        self._validate_parameters()

    def _validate_parameters(self):
        if self.E <= 0.0:
            raise ValueError("LinearElastic parameter E must be positive.")

        if not (-1.0 < self.nu < 0.5):
            raise ValueError(
                "LinearElastic Poisson ratio nu must satisfy -1 < nu < 0.5."
            )

        if self.density is not None and self.density < 0.0:
            raise ValueError(
                "LinearElastic density must be non-negative."
            )

    def tangent_matrix(self, state=None):
        """
        Return the material tangent matrix C.

        Parameters
        ----------
        state : optional
            Reserved for future compatibility with nonlinear material models.
            Linear elasticity does not need material state.

        Returns
        -------
        C : ndarray, shape (3, 3)
            Constitutive matrix for 2-D small-strain analysis.
        """
        if self.formulation == "plane_strain":
            E = self.E
            nu = self.nu

            C = E / (1.0 + nu) / (1.0 - 2.0 * nu) * np.array([
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, 0.5 - nu],
            ])

            return C

        raise NotImplementedError(
            f'LinearElastic formulation "{self.formulation}" '
            "is not yet implemented."
        )

    def stress(self, strain, state=None):
        """
        Return constitutive EFFECTIVE stress for a total strain vector.

        This helper is not required by the current stiffness assembly but is
        useful for future stress post-processing.
        """
        C = self.tangent_matrix(state=state)
        return C @ np.asarray(strain)

    def stress_tensor(self, strain, state=None):
        """
        Return the complete 3-D stress tensor for the current 2-D formulation.

        For plane strain:

            epsilon_zz = 0

        but generally:

            sigma_zz != 0

        The input engineering-strain vector is:

            [epsilon_xx, epsilon_yy, gamma_xy]

        and the returned tensor is:

            [[sigma_xx, tau_xy,   0],
             [tau_xy,   sigma_yy, 0],
             [0,        0,        sigma_zz]]

        Sign convention
        ---------------
        The current GeoPyFEM elastic formulation is tension-positive.

        Future nonlinear constitutive models should provide converged stress
        tensors from their Gauss-point state rather than reconstructing stress
        from total strain during post-processing.
        """
        strain = np.asarray(strain, dtype=float)

        if self.formulation != "plane_strain":
            raise NotImplementedError(
                f'Complete stress tensor for formulation '
                f'"{self.formulation}" is not yet implemented.'
            )

        sigma_xx, sigma_yy, tau_xy = self.stress(
            strain,
            state=state,
        )

        # For isotropic linear elasticity under plane strain:
        #
        #     epsilon_zz = 0
        #     sigma_zz = nu * (sigma_xx + sigma_yy)
        sigma_zz = self.nu * (
            sigma_xx + sigma_yy
        )

        return np.array([
            [sigma_xx, tau_xy, 0.0],
            [tau_xy, sigma_yy, 0.0],
            [0.0, 0.0, sigma_zz],
        ])

    def integrate(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
        context=None,
    ):
        """
        Integrate one small-strain linear-elastic constitutive increment.

        GeoPyFEM may initialize a non-zero effective stress independently of
        displacement/strain (for example a K0 geostatic state), so the local
        update is written incrementally:

            sigma'_(n+1) = sigma'_n + C : Delta epsilon

        The returned ``ConstitutiveResponse`` carries BOTH the updated state
        and the tangent used by the same global Newton iteration.
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

        effective_stress_increment_tensor = (
            self.stress_tensor(
                strain_increment_2d
            )
        )

        trial_state.set_plane_strain(
            total_strain_2d
        )

        trial_state.set_effective_stress_tensor(
            committed_state.effective_stress_tensor
            + effective_stress_increment_tensor
        )

        trial_state.plastic_strain[:] = (
            committed_state.plastic_strain
        )

        trial_state.equivalent_plastic_strain = float(
            committed_state.equivalent_plastic_strain
        )

        trial_state.yielded = False

        return ConstitutiveResponse(
            state=trial_state,
            tangent=self.tangent_matrix(),
            diagnostics={
                "elastic": True,
                "yielded": False,
            },
        )
