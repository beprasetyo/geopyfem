import numpy as np

from material_model.constitutive import (
    ConstitutiveModel,
    ConstitutiveResponse,
)
from state.gauss_point_state import plane_strain_2d_to_voigt6


class MohrCoulomb(ConstitutiveModel):
    """
    Small-strain elastic-perfectly-plastic Mohr-Coulomb model for GeoPyFEM.

    GeoPyFEM interface
    ------------------
    - 2-D plane-strain finite-element kinematics.
    - Full 3-D stress integration at each Gauss point.
    - Effective stress is the constitutive stress.
    - GeoPyFEM external storage uses tension-positive stress/strain.
    - The internal Mohr-Coulomb return mapping is evaluated with a temporary
      compression-positive principal-stress convention because this is the
      convention used by the verified pedagogical derivation.
    - Non-associated flow is supported through an independent dilation angle.

    Yield function in the internal compression-positive principal space
    --------------------------------------------------------------------
        sigma1 >= sigma2 >= sigma3

        f13 = (1 - sin(phi))*sigma1
              - (1 + sin(phi))*sigma3
              - 2*c*cos(phi)

    Plastic potential uses the same form with phi replaced by psi, without
    the cohesion constant.

    Current return regions
    ----------------------
    - elastic
    - face
    - compression_edge: sigma1 > sigma2 = sigma3
    - extension_edge:   sigma1 = sigma2 > sigma3
    - apex_projection (optional pedagogical mode only)

    Tangent
    -------
    - elastic: exact plane-strain elastic tangent
    - plastic, distinct trial principal stresses: analytic spectral tangent
    - repeated/nearly repeated trial principal stresses: numerical 2-D
      tangent fallback from the same local constitutive map

    Important limitations
    ---------------------
    - The non-associated apex return is not yet considered production-ready.
      By default apex entry raises NotImplementedError.  Set
      ``apex_mode='projection'`` only for controlled pedagogical testing.
    - The repeated-eigenvalue tangent uses numerical differentiation rather
      than the closed-form limiting spectral derivative.
    - Perfect plasticity only: no cohesion/friction hardening or softening.
    - Current formulation is plane strain only.
    """

    model_name = "MohrCoulomb"

    _PLANE_STRAIN_INDICES = np.array([0, 1, 3], dtype=int)

    def __init__(self, parameters, formulation="plane_strain"):
        self.parameters = parameters
        self.formulation = formulation

        self.E = self._required_float(parameters, "E")
        self.nu = self._required_float(parameters, "nu")

        self.cohesion = self._required_float_alias(
            parameters,
            "cohesion",
            "c",
        )
        self.friction_angle = self._required_float_alias(
            parameters,
            "friction_angle",
            "phi",
        )
        self.dilation_angle = self._optional_float_alias(
            parameters,
            "dilation_angle",
            "psi",
            default=0.0,
        )

        density = parameters.get("density")
        self.density = None if density is None else float(density)

        self.tangent_perturbation = float(
            parameters.get("tangent_perturbation", 1.0e-8)
        )
        self.repeated_eigenvalue_tolerance = float(
            parameters.get("repeated_eigenvalue_tolerance", 1.0e-8)
        )
        self.yield_tolerance_factor = float(
            parameters.get("yield_tolerance_factor", 1.0e-10)
        )

        self.apex_mode = str(
            parameters.get("apex_mode", "error")
        ).strip().lower()

        self._validate_parameters()

        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.lame_lambda = (
            self.E
            * self.nu
            / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        )

        self._C6 = self._build_elastic_matrix_3d()
        self._Cp = self._C6[:3, :3].copy()
        self._C_plane_strain = self._C6[
            np.ix_(self._PLANE_STRAIN_INDICES, self._PLANE_STRAIN_INDICES)
        ].copy()

    # =========================================================================
    # PARAMETERS / VALIDATION
    # =========================================================================

    @staticmethod
    def _required_float(parameters, name):
        if name not in parameters:
            raise ValueError(
                f"MohrCoulomb material requires parameter {name}."
            )
        return float(parameters[name])

    @staticmethod
    def _required_float_alias(parameters, primary, alias):
        if primary in parameters:
            return float(parameters[primary])
        if alias in parameters:
            return float(parameters[alias])
        raise ValueError(
            "MohrCoulomb material requires parameter "
            f"{primary} (alias: {alias})."
        )

    @staticmethod
    def _optional_float_alias(parameters, primary, alias, default):
        if primary in parameters:
            return float(parameters[primary])
        if alias in parameters:
            return float(parameters[alias])
        return float(default)

    def _validate_parameters(self):
        if self.formulation != "plane_strain":
            raise NotImplementedError(
                "MohrCoulomb currently supports only plane_strain."
            )

        if self.E <= 0.0:
            raise ValueError("MohrCoulomb parameter E must be positive.")

        if not (-1.0 < self.nu < 0.5):
            raise ValueError(
                "MohrCoulomb Poisson ratio nu must satisfy -1 < nu < 0.5."
            )

        if self.cohesion < 0.0:
            raise ValueError("MohrCoulomb cohesion must be non-negative.")

        if not (0.0 <= self.friction_angle < 90.0):
            raise ValueError(
                "MohrCoulomb friction_angle must satisfy 0 <= phi < 90 deg."
            )

        if not (-89.0 < self.dilation_angle < 89.0):
            raise ValueError(
                "MohrCoulomb dilation_angle must satisfy -89 < psi < 89 deg."
            )

        if self.density is not None and self.density < 0.0:
            raise ValueError("MohrCoulomb density must be non-negative.")

        if self.tangent_perturbation <= 0.0:
            raise ValueError(
                "MohrCoulomb tangent_perturbation must be positive."
            )

        if self.repeated_eigenvalue_tolerance <= 0.0:
            raise ValueError(
                "MohrCoulomb repeated_eigenvalue_tolerance must be positive."
            )

        if self.yield_tolerance_factor <= 0.0:
            raise ValueError(
                "MohrCoulomb yield_tolerance_factor must be positive."
            )

        if self.apex_mode not in {"error", "projection"}:
            raise ValueError(
                "MohrCoulomb apex_mode must be 'error' or 'projection'."
            )

    # =========================================================================
    # COMMON GEOpyFEM STATE HELPERS
    # =========================================================================

    @staticmethod
    def _effective_stress_from_state(state):
        """Read the explicit effective stress, with legacy fallback."""
        if hasattr(state, "effective_stress"):
            return np.asarray(state.effective_stress, dtype=float).reshape(6)
        return np.asarray(state.stress, dtype=float).reshape(6)

    @staticmethod
    def _set_effective_stress(state, stress):
        """Write effective stress using the current state API or legacy field."""
        stress = np.asarray(stress, dtype=float).reshape(6)
        if hasattr(state, "effective_stress"):
            state.effective_stress[:] = stress
        else:
            state.stress[:] = stress

    @staticmethod
    def _stress_2d_from_state(state):
        stress = MohrCoulomb._effective_stress_from_state(state)
        return np.array([stress[0], stress[1], stress[3]], dtype=float)

    # =========================================================================
    # ELASTICITY / TENSOR HELPERS
    # =========================================================================

    def _build_elastic_matrix_3d(self):
        lam = self.lame_lambda
        G = self.G

        C = np.zeros((6, 6), dtype=float)
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * G
        C[0, 1] = C[0, 2] = lam
        C[1, 0] = C[1, 2] = lam
        C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = G
        return C

    @staticmethod
    def _stress_voigt6_to_tensor(stress):
        sxx, syy, szz, txy, tyz, txz = np.asarray(
            stress, dtype=float
        ).reshape(6)
        return np.array(
            [
                [sxx, txy, txz],
                [txy, syy, tyz],
                [txz, tyz, szz],
            ],
            dtype=float,
        )

    @staticmethod
    def _stress_tensor_to_voigt6(tensor):
        tensor = np.asarray(tensor, dtype=float).reshape(3, 3)
        return np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[0, 1],
                tensor[1, 2],
                tensor[0, 2],
            ],
            dtype=float,
        )

    @staticmethod
    def _strain_tensor_to_voigt6(tensor):
        """Tensor strain -> engineering-shear Voigt strain."""
        tensor = np.asarray(tensor, dtype=float).reshape(3, 3)
        return np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                2.0 * tensor[0, 1],
                2.0 * tensor[1, 2],
                2.0 * tensor[0, 2],
            ],
            dtype=float,
        )

    @staticmethod
    def _principal_decomposition_compression_positive(stress_cp):
        tensor = MohrCoulomb._stress_voigt6_to_tensor(stress_cp)
        values, vectors = np.linalg.eigh(tensor)
        order = np.argsort(values)[::-1]
        return values[order], vectors[:, order]

    @staticmethod
    def _principal_to_cartesian(principal_stress, directions):
        tensor = directions @ np.diag(principal_stress) @ directions.T
        return MohrCoulomb._stress_tensor_to_voigt6(tensor)

    @staticmethod
    def _q_from_principal(principal_stress):
        s1, s2, s3 = np.asarray(principal_stress, dtype=float).reshape(3)
        return float(
            np.sqrt(
                0.5
                * (
                    (s1 - s2) ** 2
                    + (s2 - s3) ** 2
                    + (s3 - s1) ** 2
                )
            )
        )

    # =========================================================================
    # MOHR-COULOMB GEOMETRY IN COMPRESSION-POSITIVE PRINCIPAL SPACE
    # =========================================================================

    def _mc_plane_vectors(self):
        phi = np.radians(self.friction_angle)
        psi = np.radians(self.dilation_angle)

        A = 1.0 - np.sin(phi)
        B = 1.0 + np.sin(phi)
        Ap = 1.0 - np.sin(psi)
        Bp = 1.0 + np.sin(psi)

        a13 = np.array([A, 0.0, -B], dtype=float)
        a12 = np.array([A, -B, 0.0], dtype=float)
        a23 = np.array([0.0, A, -B], dtype=float)

        b13 = np.array([Ap, 0.0, -Bp], dtype=float)
        b12 = np.array([Ap, -Bp, 0.0], dtype=float)
        b23 = np.array([0.0, Ap, -Bp], dtype=float)

        return a13, a12, a23, b13, b12, b23

    def _cohesion_term(self):
        phi = np.radians(self.friction_angle)
        return 2.0 * self.cohesion * np.cos(phi)

    @staticmethod
    def _yield_value(stress_p, a, cohesion_term):
        return float(a @ stress_p - cohesion_term)

    def _yield_tolerance(self):
        stress_scale = max(
            self.E,
            self.cohesion,
            1.0,
        )
        return self.yield_tolerance_factor * stress_scale

    @staticmethod
    def _is_ordered(stress, tol):
        s1, s2, s3 = stress
        return (s1 >= s2 - tol) and (s2 >= s3 - tol)

    @staticmethod
    def _valid_compression_edge(stress, dlambda, tol):
        s1, s2, s3 = stress
        return (
            s1 >= s2 - tol
            and abs(s2 - s3) <= tol
            and np.all(dlambda >= -tol)
        )

    @staticmethod
    def _valid_extension_edge(stress, dlambda, tol):
        s1, s2, s3 = stress
        return (
            abs(s1 - s2) <= tol
            and s2 >= s3 - tol
            and np.all(dlambda >= -tol)
        )

    def _active_set_return(self, stress_trial, A, B, cohesion_term):
        n_active = A.shape[1]
        f_trial = A.T @ stress_trial - cohesion_term * np.ones(n_active)

        M = A.T @ self._Cp @ B
        dlambda = np.linalg.solve(M, f_trial)
        dstrain_p = B @ dlambda
        stress_new = stress_trial - self._Cp @ dstrain_p

        # H = d sigma_new_principal / d sigma_trial_principal
        Y = np.linalg.solve(M, A.T)
        H = np.eye(3) - self._Cp @ B @ Y
        Dalg_principal = H @ self._Cp

        return stress_new, dstrain_p, dlambda, Dalg_principal, H

    def _principal_return(self, stress_trial):
        (
            a13,
            a12,
            a23,
            b13,
            b12,
            b23,
        ) = self._mc_plane_vectors()

        C = self._cohesion_term()
        tol = self._yield_tolerance()

        f13_trial = self._yield_value(stress_trial, a13, C)

        if f13_trial <= tol:
            return {
                "stress": stress_trial.copy(),
                "region": "elastic",
                "yielded": False,
                "plastic_strain_increment_principal": np.zeros(3),
                "dlambda": np.zeros(0, dtype=float),
                "tangent_principal": self._Cp.copy(),
                "principal_jacobian": np.eye(3),
                "yield_function_trial": float(f13_trial),
            }

        A_face = a13.reshape(3, 1)
        B_face = b13.reshape(3, 1)
        (
            stress_face,
            deps_face,
            dlambda_face,
            Dalg_face,
            H_face,
        ) = self._active_set_return(stress_trial, A_face, B_face, C)

        if self._is_ordered(stress_face, tol) and np.all(dlambda_face >= -tol):
            return {
                "stress": stress_face,
                "region": "face",
                "yielded": True,
                "plastic_strain_increment_principal": deps_face,
                "dlambda": dlambda_face,
                "tangent_principal": Dalg_face,
                "principal_jacobian": H_face,
                "yield_function_trial": float(f13_trial),
            }

        A_ce = np.column_stack([a13, a12])
        B_ce = np.column_stack([b13, b12])
        (
            stress_ce,
            deps_ce,
            dlambda_ce,
            Dalg_ce,
            H_ce,
        ) = self._active_set_return(stress_trial, A_ce, B_ce, C)

        if self._valid_compression_edge(stress_ce, dlambda_ce, tol):
            return {
                "stress": stress_ce,
                "region": "compression_edge",
                "yielded": True,
                "plastic_strain_increment_principal": deps_ce,
                "dlambda": dlambda_ce,
                "tangent_principal": Dalg_ce,
                "principal_jacobian": H_ce,
                "yield_function_trial": float(f13_trial),
            }

        A_ee = np.column_stack([a13, a23])
        B_ee = np.column_stack([b13, b23])
        (
            stress_ee,
            deps_ee,
            dlambda_ee,
            Dalg_ee,
            H_ee,
        ) = self._active_set_return(stress_trial, A_ee, B_ee, C)

        if self._valid_extension_edge(stress_ee, dlambda_ee, tol):
            return {
                "stress": stress_ee,
                "region": "extension_edge",
                "yielded": True,
                "plastic_strain_increment_principal": deps_ee,
                "dlambda": dlambda_ee,
                "tangent_principal": Dalg_ee,
                "principal_jacobian": H_ee,
                "yield_function_trial": float(f13_trial),
            }

        # Pedagogical geometric apex projection.
        if self.friction_angle <= 0.0:
            raise NotImplementedError(
                "MohrCoulomb reached the apex branch with phi=0. "
                "No finite cohesive MC apex exists for phi=0."
            )

        phi = np.radians(self.friction_angle)
        p_apex = -self.cohesion / np.tan(phi)

        return {
            "stress": np.array([p_apex, p_apex, p_apex], dtype=float),
            "region": "apex_projection",
            "yielded": True,
            "plastic_strain_increment_principal": None,
            "dlambda": np.zeros(0, dtype=float),
            "tangent_principal": np.zeros((3, 3), dtype=float),
            "principal_jacobian": np.zeros((3, 3), dtype=float),
            "yield_function_trial": float(f13_trial),
        }

    # =========================================================================
    # SPECTRAL TANGENT
    # =========================================================================

    def _has_repeated_principal_stress(self, principal_stress):
        principal_stress = np.asarray(principal_stress, dtype=float).reshape(3)
        scale = max(1.0, float(np.max(np.abs(principal_stress))))
        tol = self.repeated_eigenvalue_tolerance * scale

        s1, s2, s3 = principal_stress
        return (
            abs(s1 - s2) < tol
            or abs(s2 - s3) < tol
            or abs(s1 - s3) < tol
        )

    def _analytic_cartesian_tangent_6d(
        self,
        principal_trial,
        principal_new,
        directions,
        H,
    ):
        """
        Full 6x6 tangent in the same numerical convention as GeoPyFEM.

        The internal compression-positive transformation changes both stress
        and strain signs, so d(sigma_tension)/d(epsilon_tension) has the same
        numerical tangent as the compression-positive derivative.
        """
        Dcart = np.zeros((6, 6), dtype=float)

        for j in range(6):
            deps = np.zeros(6, dtype=float)
            deps[j] = 1.0

            dstress_trial_vec = self._C6 @ deps
            dstress_trial_tensor = self._stress_voigt6_to_tensor(
                dstress_trial_vec
            )

            dtrial_p = directions.T @ dstress_trial_tensor @ directions
            dnew_p = np.zeros((3, 3), dtype=float)

            dnew_diag = H @ np.diag(dtrial_p)
            for i in range(3):
                dnew_p[i, i] = dnew_diag[i]

            for i in range(3):
                for k in range(i + 1, 3):
                    denominator = principal_trial[i] - principal_trial[k]
                    scale = max(
                        1.0,
                        abs(principal_trial[i]),
                        abs(principal_trial[k]),
                    )
                    if abs(denominator) < self.repeated_eigenvalue_tolerance * scale:
                        raise ValueError(
                            "Repeated trial principal stresses encountered "
                            "inside analytic spectral tangent."
                        )

                    beta = (
                        principal_new[i] - principal_new[k]
                    ) / denominator
                    value = beta * dtrial_p[i, k]
                    dnew_p[i, k] = value
                    dnew_p[k, i] = value

            dnew_cart = directions @ dnew_p @ directions.T
            Dcart[:, j] = self._stress_tensor_to_voigt6(dnew_cart)

        return Dcart

    # =========================================================================
    # LOCAL STRESS INTEGRATION
    # =========================================================================

    def _integrate_state(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
    ):
        """Perform one local stress update without computing the 2-D tangent."""
        total_strain_2d = np.asarray(total_strain_2d, dtype=float).reshape(3)
        strain_increment_2d = np.asarray(
            strain_increment_2d, dtype=float
        ).reshape(3)

        total_strain_6 = plane_strain_2d_to_voigt6(total_strain_2d)
        dstrain_6 = plane_strain_2d_to_voigt6(strain_increment_2d)

        stress_old_tp = self._effective_stress_from_state(committed_state)
        stress_trial_tp = stress_old_tp + self._C6 @ dstrain_6

        # GeoPyFEM is tension-positive.  The verified MC return is internally
        # compression-positive; convert only inside the local integration.
        stress_trial_cp = -stress_trial_tp

        principal_trial_cp, directions = (
            self._principal_decomposition_compression_positive(stress_trial_cp)
        )

        result = self._principal_return(principal_trial_cp)

        if result["region"] == "apex_projection" and self.apex_mode == "error":
            raise NotImplementedError(
                "MohrCoulomb entered the non-associated apex region. "
                "The current GeoPyFEM MC implementation intentionally stops "
                "here because the production apex algorithm has not yet been "
                "implemented. Use apex_mode='projection' only for controlled "
                "pedagogical tests."
            )

        principal_new_cp = result["stress"]
        stress_new_cp = self._principal_to_cartesian(
            principal_new_cp,
            directions,
        )
        stress_new_tp = -stress_new_cp

        trial_state = committed_state.copy()
        trial_state.strain[:] = total_strain_6
        self._set_effective_stress(trial_state, stress_new_tp)
        trial_state.yielded = bool(result["yielded"])

        # Accumulate plastic strain only for branches with a defined
        # non-associated flow increment.  The pedagogical apex projection is
        # deliberately not assigned a fictitious flow-rule plastic strain.
        deps_p_principal_cp = result["plastic_strain_increment_principal"]
        if result["yielded"] and deps_p_principal_cp is not None:
            deps_p_tensor_cp = (
                directions
                @ np.diag(deps_p_principal_cp)
                @ directions.T
            )
            # tension-positive strain convention is the negative of the
            # temporary compression-positive constitutive convention.
            deps_p_tensor_tp = -deps_p_tensor_cp
            deps_p_voigt_tp = self._strain_tensor_to_voigt6(deps_p_tensor_tp)
            trial_state.plastic_strain[:] = (
                np.asarray(committed_state.plastic_strain, dtype=float)
                + deps_p_voigt_tp
            )

        a13, _, _, _, _, _ = self._mc_plane_vectors()
        C = self._cohesion_term()
        yield_function_new = self._yield_value(
            principal_new_cp,
            a13,
            C,
        )

        q_new = self._q_from_principal(principal_new_cp)
        p_new_cp = float(np.mean(principal_new_cp))
        dlambda = np.asarray(result["dlambda"], dtype=float).reshape(-1)

        trial_state.internal_variables["mc_region"] = result["region"]
        trial_state.internal_variables["yield_function_trial"] = float(
            result["yield_function_trial"]
        )
        trial_state.internal_variables["yield_function"] = float(
            yield_function_new
        )
        trial_state.internal_variables["plastic_multiplier"] = float(
            np.sum(dlambda) if dlambda.size else 0.0
        )
        trial_state.internal_variables["plastic_multipliers"] = dlambda.copy()
        trial_state.internal_variables["q"] = float(q_new)
        trial_state.internal_variables["p_compression_positive"] = p_new_cp
        trial_state.internal_variables["principal_trial_compression_positive"] = (
            principal_trial_cp.copy()
        )
        trial_state.internal_variables["principal_new_compression_positive"] = (
            principal_new_cp.copy()
        )
        trial_state.internal_variables["principal_directions"] = directions.copy()
        trial_state.internal_variables["principal_jacobian"] = result[
            "principal_jacobian"
        ].copy()

        return trial_state

    # =========================================================================
    # NUMERICAL 2-D TANGENT FALLBACK
    # =========================================================================

    def _numerical_algorithmic_tangent_2d(
        self,
        total_strain_2d,
        committed_state,
    ):
        """Differentiate the exact 2-D local constitutive map numerically."""
        epsilon = np.asarray(total_strain_2d, dtype=float).reshape(3)
        C_alg = np.zeros((3, 3), dtype=float)

        for j in range(3):
            h = self.tangent_perturbation * max(1.0, abs(epsilon[j]))

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

            sigma_plus = self._stress_2d_from_state(state_plus)
            sigma_minus = self._stress_2d_from_state(state_minus)

            C_alg[:, j] = (sigma_plus - sigma_minus) / (2.0 * h)

        # Do NOT symmetrize: non-associated Mohr-Coulomb generally has a
        # genuinely non-symmetric algorithmic tangent.
        return C_alg

    # =========================================================================
    # PUBLIC CONSTITUTIVE INTERFACE
    # =========================================================================

    def integrate(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
        context=None,
    ):
        """Return one complete GeoPyFEM Mohr-Coulomb constitutive response."""
        total_strain_2d = np.asarray(total_strain_2d, dtype=float).reshape(3)
        strain_increment_2d = np.asarray(
            strain_increment_2d, dtype=float
        ).reshape(3)

        trial_state = self._integrate_state(
            total_strain_2d,
            strain_increment_2d,
            committed_state,
        )

        region = trial_state.internal_variables.get("mc_region", "elastic")
        tangent_mode = "elastic"

        if not trial_state.yielded:
            C_alg_2d = self._C_plane_strain.copy()

        elif region == "apex_projection":
            # Only reachable when apex_mode='projection'.
            C_alg_2d = np.zeros((3, 3), dtype=float)
            tangent_mode = "apex_zero_pedagogical"

        else:
            principal_trial_cp = np.asarray(
                trial_state.internal_variables[
                    "principal_trial_compression_positive"
                ],
                dtype=float,
            )
            principal_new_cp = np.asarray(
                trial_state.internal_variables[
                    "principal_new_compression_positive"
                ],
                dtype=float,
            )
            directions = np.asarray(
                trial_state.internal_variables["principal_directions"],
                dtype=float,
            )
            H = np.asarray(
                trial_state.internal_variables["principal_jacobian"],
                dtype=float,
            )

            if self._has_repeated_principal_stress(principal_trial_cp):
                C_alg_2d = self._numerical_algorithmic_tangent_2d(
                    total_strain_2d,
                    committed_state,
                )
                tangent_mode = "numerical_repeated_eigenvalue_fallback"
            else:
                C_alg_6d = self._analytic_cartesian_tangent_6d(
                    principal_trial_cp,
                    principal_new_cp,
                    directions,
                    H,
                )
                C_alg_2d = C_alg_6d[
                    np.ix_(
                        self._PLANE_STRAIN_INDICES,
                        self._PLANE_STRAIN_INDICES,
                    )
                ].copy()
                tangent_mode = "analytic_spectral"

        trial_state.internal_variables["tangent_mode"] = tangent_mode

        return ConstitutiveResponse(
            state=trial_state,
            tangent=C_alg_2d,
            diagnostics={
                "yielded": bool(trial_state.yielded),
                "region": region,
                "tangent_mode": tangent_mode,
                "yield_function": trial_state.internal_variables.get(
                    "yield_function"
                ),
                "yield_function_trial": trial_state.internal_variables.get(
                    "yield_function_trial"
                ),
                "plastic_multiplier": trial_state.internal_variables.get(
                    "plastic_multiplier"
                ),
                "q": trial_state.internal_variables.get("q"),
                "p_compression_positive": trial_state.internal_variables.get(
                    "p_compression_positive"
                ),
            },
        )

    # =========================================================================
    # BACKWARD-COMPATIBILITY HELPERS
    # =========================================================================

    def update_state(self, total_strain_2d, committed_state):
        """Adapter for older GeoPyFEM code paths."""
        total_strain_2d = np.asarray(total_strain_2d, dtype=float).reshape(3)

        response = self.integrate(
            total_strain_2d=total_strain_2d,
            strain_increment_2d=(
                total_strain_2d - committed_state.strain_2d
            ),
            committed_state=committed_state,
            context=None,
        )

        response.state.internal_variables[
            "algorithmic_tangent_2d"
        ] = response.tangent.copy()

        return response.state

    def tangent_matrix(self, state=None):
        """Return cached algorithmic tangent, otherwise elastic tangent."""
        if state is not None:
            tangent = state.internal_variables.get("algorithmic_tangent_2d")
            if tangent is not None:
                return np.asarray(tangent, dtype=float).copy()
        return self._C_plane_strain.copy()
