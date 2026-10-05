import numpy as np

from material_model.constitutive import (
    ConstitutiveModel,
    ConstitutiveResponse,
)
from state.gauss_point_state import plane_strain_2d_to_voigt6


class VonMises(ConstitutiveModel):
    """
    Small-strain J2 / Von Mises elastoplastic material for GeoPyFEM.

    Current formulation
    -------------------
    - 3-D constitutive stress integration
    - 2-D plane-strain finite-element kinematics
    - isotropic linear elasticity
    - associated J2 flow rule
    - optional linear isotropic hardening
    - radial-return stress integration

    Stress sign convention
    ----------------------
    GeoPyFEM currently uses tension-positive stress.  The Von Mises criterion
    is insensitive to hydrostatic stress sign because yielding depends on the
    deviatoric stress invariant J2.

    Required XML parameters
    -----------------------
    E
        Young's modulus.

    nu
        Poisson ratio.

    yield_stress
        Initial uniaxial yield stress in the same stress unit as E.

    Optional XML parameters
    -----------------------
    hardening_modulus
        Linear isotropic hardening modulus H.

        H = 0 -> perfect plasticity.

    density
        Mass density [kg/m^3] for gravity loading.

    Constitutive equations
    ----------------------
    Yield function:

        f = q - sigma_y

    where:

        q = sqrt(3/2 * s:s)

        sigma_y = sigma_y0 + H * epbar

    and ``epbar`` is the equivalent plastic strain.

    For a plastic trial state, radial return gives:

        delta_lambda = f_trial / (3G + H)

    and:

        s_new = (1 - 3G*delta_lambda/q_trial) * s_trial

    with the mean stress unchanged.

    Tangent used by Newton-Raphson
    ------------------------------
    The first GeoPyFEM implementation evaluates the algorithmic 2-D tangent
    numerically from the SAME local radial-return algorithm.  This gives a
    consistent benchmark implementation without introducing a second,
    separately-derived closed-form tangent that could hide constitutive bugs.

    A future optimization can replace this numerical differentiation with the
    analytical consistent elastoplastic tangent after the implementation has
    been thoroughly benchmarked.
    """

    # GeoPyFEM constitutive stress contract: this model evolves effective
    # stress.  Pore pressure is applied separately by global equilibrium.
    model_name = "VonMises"

    def __init__(self, parameters, formulation="plane_strain"):
        self.parameters = parameters
        self.formulation = formulation

        if "E" not in parameters:
            raise ValueError(
                "VonMises material requires parameter E."
            )

        if "nu" not in parameters:
            raise ValueError(
                "VonMises material requires parameter nu."
            )

        if "yield_stress" not in parameters:
            raise ValueError(
                "VonMises material requires parameter yield_stress."
            )

        self.E = float(parameters["E"])
        self.nu = float(parameters["nu"])
        self.yield_stress = float(
            parameters["yield_stress"]
        )

        self.hardening_modulus = float(
            parameters.get(
                "hardening_modulus",
                0.0,
            )
        )

        density = parameters.get("density")
        self.density = (
            None
            if density is None
            else float(density)
        )

        # Numerical differentiation step for the first benchmark
        # implementation of the algorithmic tangent.
        self.tangent_perturbation = float(
            parameters.get(
                "tangent_perturbation",
                1.0e-8,
            )
        )

        self._validate_parameters()

        self.G = (
            self.E
            / (2.0 * (1.0 + self.nu))
        )

        self.bulk_modulus = (
            self.E
            / (3.0 * (1.0 - 2.0 * self.nu))
        )

        self.lame_lambda = (
            self.E
            * self.nu
            / (
                (1.0 + self.nu)
                * (1.0 - 2.0 * self.nu)
            )
        )

        self._C6 = self._build_elastic_matrix_3d()
        self._C_plane_strain = self._build_elastic_matrix_plane_strain()

    def _validate_parameters(self):
        if self.formulation != "plane_strain":
            raise NotImplementedError(
                "VonMises currently supports only plane_strain."
            )

        if self.E <= 0.0:
            raise ValueError(
                "VonMises parameter E must be positive."
            )

        if not (-1.0 < self.nu < 0.5):
            raise ValueError(
                "VonMises Poisson ratio nu must satisfy -1 < nu < 0.5."
            )

        if self.yield_stress <= 0.0:
            raise ValueError(
                "VonMises yield_stress must be positive."
            )

        if self.hardening_modulus < 0.0:
            raise ValueError(
                "VonMises hardening_modulus must be non-negative."
            )

        if self.density is not None and self.density < 0.0:
            raise ValueError(
                "VonMises density must be non-negative."
            )

        if self.tangent_perturbation <= 0.0:
            raise ValueError(
                "VonMises tangent_perturbation must be positive."
            )

    # =========================================================================
    # ELASTIC MATRICES
    # =========================================================================

    def _build_elastic_matrix_3d(self):
        """
        Return the 6x6 isotropic elastic matrix using GeoPyFEM Voigt order:

            [xx, yy, zz, xy, yz, xz]

        and engineering shear strains:

            [exx, eyy, ezz, gamma_xy, gamma_yz, gamma_xz].
        """
        lam = self.lame_lambda
        G = self.G

        C = np.zeros((6, 6), dtype=float)

        C[0, 0] = lam + 2.0*G
        C[1, 1] = lam + 2.0*G
        C[2, 2] = lam + 2.0*G

        C[0, 1] = lam
        C[0, 2] = lam
        C[1, 0] = lam
        C[1, 2] = lam
        C[2, 0] = lam
        C[2, 1] = lam

        C[3, 3] = G
        C[4, 4] = G
        C[5, 5] = G

        return C

    def _build_elastic_matrix_plane_strain(self):
        E = self.E
        nu = self.nu

        return (
            E
            / (1.0 + nu)
            / (1.0 - 2.0*nu)
            * np.array([
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, 0.5 - nu],
            ])
        )

    def tangent_matrix(self, state=None):
        """
        Return the current 2-D algorithmic tangent matrix.

        Before any constitutive update, the elastic tangent is returned.
        After update_state(), the trial state carries the tangent associated
        with that exact stress-integration result.
        """
        if state is not None:
            tangent = state.internal_variables.get(
                "algorithmic_tangent_2d"
            )

            if tangent is not None:
                return np.asarray(
                    tangent,
                    dtype=float,
                ).copy()

        return self._C_plane_strain.copy()

    # =========================================================================
    # J2 / VON MISES HELPERS
    # =========================================================================

    @staticmethod
    def _deviatoric_stress(stress):
        stress = np.asarray(
            stress,
            dtype=float,
        ).reshape(6)

        mean_stress = (
            stress[0]
            + stress[1]
            + stress[2]
        ) / 3.0

        deviator = stress.copy()
        deviator[0] -= mean_stress
        deviator[1] -= mean_stress
        deviator[2] -= mean_stress

        return mean_stress, deviator

    @staticmethod
    def _von_mises_q_from_deviator(deviator):
        s = np.asarray(
            deviator,
            dtype=float,
        ).reshape(6)

        # Tensor contraction s:s. Shear terms appear twice because the
        # symmetric stress tensor contains both xy and yx entries.
        s_contract_s = (
            s[0]**2
            + s[1]**2
            + s[2]**2
            + 2.0*(
                s[3]**2
                + s[4]**2
                + s[5]**2
            )
        )

        return np.sqrt(
            1.5 * s_contract_s
        )

    @staticmethod
    def _stress_2d_from_state(state):
        return np.array([
            state.effective_stress[0],
            state.effective_stress[1],
            state.effective_stress[3],
        ], dtype=float)

    def _current_yield_stress(self, equivalent_plastic_strain):
        return (
            self.yield_stress
            + self.hardening_modulus
            * float(equivalent_plastic_strain)
        )

    # =========================================================================
    # LOCAL RETURN MAPPING
    # =========================================================================

    def _integrate_state(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
    ):
        """
        Perform one local elastic predictor / radial return operation.

        This function does NOT calculate the algorithmic tangent.  Keeping the
        stress integration separate allows the tangent to be differentiated
        from this exact constitutive map without recursion.
        """
        total_strain_2d = np.asarray(
            total_strain_2d,
            dtype=float,
        ).reshape(-1)

        if total_strain_2d.size != 3:
            raise ValueError(
                "VonMises update requires "
                "[epsilon_xx, epsilon_yy, gamma_xy]."
            )

        total_strain_6 = plane_strain_2d_to_voigt6(
            total_strain_2d
        )

        strain_increment_2d = np.asarray(
            strain_increment_2d,
            dtype=float,
        ).reshape(3)

        strain_increment = plane_strain_2d_to_voigt6(
            strain_increment_2d
        )

        # ------------------------------------------------------------------
        # Elastic predictor
        # ------------------------------------------------------------------
        stress_trial = (
            committed_state.effective_stress
            + self._C6 @ strain_increment
        )

        mean_stress_trial, deviator_trial = (
            self._deviatoric_stress(
                stress_trial
            )
        )

        q_trial = self._von_mises_q_from_deviator(
            deviator_trial
        )

        epbar_n = float(
            committed_state.equivalent_plastic_strain
        )

        yield_stress_n = self._current_yield_stress(
            epbar_n
        )

        yield_function_trial = (
            q_trial
            - yield_stress_n
        )

        yield_tolerance = (
            1.0e-10
            * max(
                self.yield_stress,
                1.0,
            )
        )

        trial_state = committed_state.copy()
        trial_state.strain[:] = total_strain_6

        # ------------------------------------------------------------------
        # Elastic step
        # ------------------------------------------------------------------
        if yield_function_trial <= yield_tolerance:
            trial_state.effective_stress[:] = stress_trial
            trial_state.yielded = False

            trial_state.internal_variables[
                "plastic_multiplier"
            ] = 0.0

            trial_state.internal_variables[
                "q"
            ] = float(q_trial)

            trial_state.internal_variables[
                "mean_stress"
            ] = float(mean_stress_trial)

            trial_state.internal_variables[
                "current_yield_stress"
            ] = float(yield_stress_n)

            trial_state.internal_variables[
                "yield_function"
            ] = float(yield_function_trial)

            return trial_state

        # ------------------------------------------------------------------
        # Plastic radial return
        # ------------------------------------------------------------------
        if q_trial <= 0.0:
            raise RuntimeError(
                "VonMises plastic correction encountered q_trial <= 0."
            )

        delta_lambda = (
            yield_function_trial
            / (
                3.0*self.G
                + self.hardening_modulus
            )
        )

        radial_scale = (
            1.0
            - 3.0*self.G
            * delta_lambda
            / q_trial
        )

        deviator_new = (
            radial_scale
            * deviator_trial
        )

        stress_new = deviator_new.copy()
        stress_new[0] += mean_stress_trial
        stress_new[1] += mean_stress_trial
        stress_new[2] += mean_stress_trial

        # Associated J2 plastic-flow direction in GeoPyFEM engineering-strain
        # Voigt convention.
        flow_direction = np.array([
            1.5 * deviator_trial[0] / q_trial,
            1.5 * deviator_trial[1] / q_trial,
            1.5 * deviator_trial[2] / q_trial,
            3.0 * deviator_trial[3] / q_trial,
            3.0 * deviator_trial[4] / q_trial,
            3.0 * deviator_trial[5] / q_trial,
        ], dtype=float)

        plastic_strain_new = (
            committed_state.plastic_strain
            + delta_lambda
            * flow_direction
        )

        epbar_new = (
            epbar_n
            + delta_lambda
        )

        yield_stress_new = self._current_yield_stress(
            epbar_new
        )

        _, deviator_check = self._deviatoric_stress(
            stress_new
        )

        q_new = self._von_mises_q_from_deviator(
            deviator_check
        )

        trial_state.effective_stress[:] = stress_new
        trial_state.plastic_strain[:] = plastic_strain_new
        trial_state.equivalent_plastic_strain = float(
            epbar_new
        )
        trial_state.yielded = True

        trial_state.internal_variables[
            "plastic_multiplier"
        ] = float(delta_lambda)

        trial_state.internal_variables[
            "q"
        ] = float(q_new)

        trial_state.internal_variables[
            "mean_stress"
        ] = float(mean_stress_trial)

        trial_state.internal_variables[
            "current_yield_stress"
        ] = float(yield_stress_new)

        trial_state.internal_variables[
            "yield_function"
        ] = float(
            q_new
            - yield_stress_new
        )

        return trial_state

    # =========================================================================
    # ALGORITHMIC TANGENT
    # =========================================================================

    def _numerical_algorithmic_tangent(
        self,
        total_strain_2d,
        committed_state,
    ):
        """
        Differentiate the local constitutive map numerically.

        The committed state is held fixed while the CURRENT total strain is
        perturbed.  This is exactly the derivative required by the global
        Newton iteration:

            C_alg = d sigma_(n+1) / d epsilon_(n+1)

        for the current load step.
        """
        epsilon = np.asarray(
            total_strain_2d,
            dtype=float,
        ).reshape(3)

        C_alg = np.zeros((3, 3), dtype=float)

        for j in range(3):
            h = (
                self.tangent_perturbation
                * max(
                    1.0,
                    abs(epsilon[j]),
                )
            )

            epsilon_plus = epsilon.copy()
            epsilon_minus = epsilon.copy()

            epsilon_plus[j] += h
            epsilon_minus[j] -= h

            state_plus = self._integrate_state(
                epsilon_plus,
                epsilon_plus - committed_state.strain_2d,
                committed_state,
            )

            state_minus = self._integrate_state(
                epsilon_minus,
                epsilon_minus - committed_state.strain_2d,
                committed_state,
            )

            sigma_plus = self._stress_2d_from_state(
                state_plus
            )

            sigma_minus = self._stress_2d_from_state(
                state_minus
            )

            C_alg[:, j] = (
                sigma_plus
                - sigma_minus
            ) / (2.0*h)

        # Associated J2 plasticity has a symmetric consistent tangent.
        # Numerical differentiation can introduce tiny asymmetry, so remove
        # only that numerical noise.
        return 0.5 * (
            C_alg
            + C_alg.T
        )

    # =========================================================================
    # PUBLIC CONSTITUTIVE UPDATE
    # =========================================================================

    def integrate(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
        context=None,
    ):
        """
        Return one complete J2 constitutive response.

        Stress integration and the algorithmic tangent are generated from the
        SAME local return-mapping operation.  The global solver therefore does
        not need a second material-specific tangent call.
        """
        total_strain_2d = np.asarray(
            total_strain_2d,
            dtype=float,
        ).reshape(3)

        strain_increment_2d = np.asarray(
            strain_increment_2d,
            dtype=float,
        ).reshape(3)

        trial_state = self._integrate_state(
            total_strain_2d,
            strain_increment_2d,
            committed_state,
        )

        if trial_state.yielded:
            C_alg = self._numerical_algorithmic_tangent(
                total_strain_2d,
                committed_state,
            )
        else:
            C_alg = self._C_plane_strain.copy()

        return ConstitutiveResponse(
            state=trial_state,
            tangent=C_alg,
            diagnostics={
                "yielded": bool(trial_state.yielded),
                "q": trial_state.internal_variables.get("q"),
                "yield_function": trial_state.internal_variables.get(
                    "yield_function"
                ),
                "plastic_multiplier": trial_state.internal_variables.get(
                    "plastic_multiplier"
                ),
            },
        )

    def update_state(
        self,
        total_strain_2d,
        committed_state,
    ):
        """Backward-compatible adapter for pre-interface GeoPyFEM code."""
        total_strain_2d = np.asarray(
            total_strain_2d,
            dtype=float,
        ).reshape(3)

        response = self.integrate(
            total_strain_2d=total_strain_2d,
            strain_increment_2d=(
                total_strain_2d
                - committed_state.strain_2d
            ),
            committed_state=committed_state,
            context=None,
        )

        # Preserve the old tangent_matrix(state=...) behaviour for external
        # scripts that have not yet migrated to ConstitutiveResponse.
        response.state.internal_variables[
            "algorithmic_tangent_2d"
        ] = response.tangent.copy()

        return response.state
