from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np

from material_model.constitutive import (
    ConstitutiveModel,
    ConstitutiveResponse,
)
from state.gauss_point_state import plane_strain_2d_to_voigt6



@dataclass(slots=True)
class _MohrCoulombLocalResult:
    """One local MC update with transient spectral data kept off the state.

    ``state`` contains only history/output variables that must survive a
    committed increment.  Principal stresses/directions/Jacobians are local
    work data required only while constructing the algorithmic tangent.
    Keeping them here avoids copying several small NumPy arrays through every
    trial/commit/revert operation.
    """

    state: object
    region: str
    yield_function_trial: float
    yield_function: float
    plastic_multiplier: float
    q: float
    p_compression_positive: float
    principal_trial_compression_positive: np.ndarray
    principal_new_compression_positive: np.ndarray
    principal_directions: Optional[np.ndarray]
    principal_jacobian: Optional[np.ndarray]


def _mc_profile_start(profiler):
    return perf_counter() if profiler is not None else None


def _mc_profile_add(profiler, name, start, count_name=None, count=1):
    if profiler is None or start is None:
        return
    profiler.add_time(name, perf_counter() - start)
    if count_name is not None:
        profiler.increment(count_name, count)


def _mc_profile_increment(profiler, name, count=1):
    if profiler is not None:
        profiler.increment(name, count)


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
    - apex_fortran: direct apex projection with the modified single-singular
      apex tangent used by the reference Fortran implementation
    - apex_projection: retained as a legacy zero-tangent diagnostic mode

    Elastic fast path
    -----------------
    - plane-strain trial principal VALUES are evaluated analytically;
    - clearly elastic trials return without ``np.linalg.eigh``;
    - full principal directions are computed only for plastic/near-yield
      states, so the verified plastic face/edge/apex return remains unchanged.

    Tangent
    -------
    - elastic: exact plane-strain elastic tangent
    - plastic, distinct trial principal stresses: analytic spectral tangent
    - apex_fortran: modified single-singular apex tangent (Fortran style)
    - repeated/nearly repeated trial principal stresses: numerical 2-D
      tangent fallback from the same local constitutive map

    Important limitations
    ---------------------
    - The Fortran-style apex treatment is a modified tangent regularisation,
      not an exact three-surface non-associated apex active-set solution.
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

        # Reference strength is immutable input data for phi-c reduction.
        # Safety factors are always applied relative to these ORIGINAL values,
        # never compounded from an already reduced material state.
        self.reference_cohesion = float(self.cohesion)
        self.reference_friction_angle = float(self.friction_angle)
        self.reference_dilation_angle = float(self.dilation_angle)
        self.strength_reduction_factor = 1.0

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
            parameters.get("apex_mode", "fortran")
        ).strip().lower()
        self.apex_alpha = float(
            parameters.get("apex_alpha", 17.0)
        )

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

        if self.apex_mode not in {"fortran", "error", "projection"}:
            raise ValueError(
                "MohrCoulomb apex_mode must be 'fortran', 'error', "
                "or 'projection'."
            )

        if self.apex_alpha <= 0.0:
            raise ValueError(
                "MohrCoulomb apex_alpha must be positive."
            )

    # =========================================================================
    # PHI-C STRENGTH REDUCTION INTERFACE
    # =========================================================================

    def set_strength_reduction_factor(self, factor):
        """Apply one absolute phi-c reduction factor.

        GeoPyFEM follows the conventional strength-reduction definition:

            c_reduced = c_input / F
            tan(phi_reduced) = tan(phi_input) / F

        The dilation angle is intentionally left unchanged.  In particular,
        the common geotechnical choice psi = 0 therefore remains zero during
        the safety phase.
        """
        factor = float(factor)

        if factor < 1.0:
            raise ValueError(
                "MohrCoulomb strength reduction factor must be >= 1.0."
            )

        phi0 = np.radians(self.reference_friction_angle)

        self.cohesion = self.reference_cohesion / factor
        self.friction_angle = np.degrees(
            np.arctan(np.tan(phi0) / factor)
        )
        self.dilation_angle = self.reference_dilation_angle
        self.strength_reduction_factor = factor

    def restore_reference_strength(self):
        """Restore the original input Mohr-Coulomb strength."""
        self.cohesion = float(self.reference_cohesion)
        self.friction_angle = float(self.reference_friction_angle)
        self.dilation_angle = float(self.reference_dilation_angle)
        self.strength_reduction_factor = 1.0

    def strength_reduction_state(self):
        """Return current/reference strength values for safety diagnostics."""
        return {
            "model": self.model_name,
            "factor": float(self.strength_reduction_factor),
            "reference_cohesion": float(self.reference_cohesion),
            "reference_friction_angle": float(
                self.reference_friction_angle
            ),
            "reference_dilation_angle": float(
                self.reference_dilation_angle
            ),
            "cohesion": float(self.cohesion),
            "friction_angle": float(self.friction_angle),
            "dilation_angle": float(self.dilation_angle),
        }

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
    def _principal_decomposition_compression_positive(
        stress_cp,
        profiler=None,
    ):
        profile_start = _mc_profile_start(profiler)
        tensor = MohrCoulomb._stress_voigt6_to_tensor(stress_cp)
        values, vectors = np.linalg.eigh(tensor)
        order = np.argsort(values)[::-1]
        result = values[order], vectors[:, order]
        _mc_profile_add(
            profiler,
            "mc_principal_decomposition",
            profile_start,
            count_name="mc_principal_decompositions",
        )
        return result

    @staticmethod
    def _plane_strain_principal_values_compression_positive(
        stress_cp,
        profiler=None,
    ):
        """Return ordered principal values without computing eigenvectors.

        For the current plane-strain constitutive state the stress tensor has
        the block form

            [ sxx  txy   0 ]
            [ txy  syy   0 ]
            [  0    0   szz]

        so the two in-plane eigenvalues are available analytically and the
        third principal value is simply ``szz``.  This is the hot-path
        operation used for the elastic Mohr-Coulomb yield check.
        """
        profile_start = _mc_profile_start(profiler)
        try:
            sxx, syy, szz, txy, tyz, txz = np.asarray(
                stress_cp, dtype=float
            ).reshape(6)

            # Plane strain should not generate yz/xz shear.  If a future
            # extension supplies such components, preserve correctness by
            # declining the fast path and letting the caller use full eigh.
            shear_scale = max(
                1.0,
                abs(sxx),
                abs(syy),
                abs(szz),
                abs(txy),
            )
            shear_tol = 64.0 * np.finfo(float).eps * shear_scale
            if abs(tyz) > shear_tol or abs(txz) > shear_tol:
                return None

            mean_xy = 0.5 * (sxx + syy)
            radius_xy = float(
                np.hypot(0.5 * (sxx - syy), txy)
            )
            sa = mean_xy + radius_xy
            sb = mean_xy - radius_xy
            sc = szz

            # Manual ordering avoids a general eigensolver and keeps the
            # internal compression-positive convention sigma1 >= sigma2 >=
            # sigma3.
            s1 = max(sa, sb, sc)
            s3 = min(sa, sb, sc)
            s2 = sa + sb + sc - s1 - s3

            return np.array([s1, s2, s3], dtype=float)
        finally:
            _mc_profile_add(
                profiler,
                "mc_analytic_principal_values",
                profile_start,
                count_name="mc_analytic_principal_value_evaluations",
            )

    def _elastic_fast_trial_check(self, stress_trial_cp, profiler=None):
        """Check the MC trial yield condition without full eigenvectors.

        Returns ``(principal_values, f_trial, definitely_elastic)``.  A tiny
        floating-point guard around the yield tolerance deliberately falls
        back to the original full spectral path so near-yield decisions retain
        the previously validated behaviour.
        """
        profile_start = _mc_profile_start(profiler)
        try:
            principal = self._plane_strain_principal_values_compression_positive(
                stress_trial_cp,
                profiler=profiler,
            )
            if principal is None:
                _mc_profile_increment(
                    profiler,
                    "mc_fast_path_non_plane_strain_fallbacks",
                )
                return None, None, False

            phi = np.radians(self.friction_angle)
            sin_phi = np.sin(phi)
            A = 1.0 - sin_phi
            B = 1.0 + sin_phi
            C = 2.0 * self.cohesion * np.cos(phi)

            f_trial = float(A * principal[0] - B * principal[2] - C)
            tol = self._yield_tolerance()

            scale = max(
                1.0,
                abs(A * principal[0]),
                abs(B * principal[2]),
                abs(C),
            )
            roundoff_guard = 64.0 * np.finfo(float).eps * scale
            definitely_elastic = f_trial <= (tol - roundoff_guard)

            if definitely_elastic:
                _mc_profile_increment(profiler, "mc_elastic_fast_returns")
            elif f_trial <= tol + roundoff_guard:
                _mc_profile_increment(
                    profiler,
                    "mc_fast_path_yield_boundary_fallbacks",
                )
            else:
                _mc_profile_increment(
                    profiler,
                    "mc_fast_path_plastic_candidates",
                )

            return principal, f_trial, definitely_elastic
        finally:
            _mc_profile_add(
                profiler,
                "mc_elastic_fast_yield_check",
                profile_start,
                count_name="mc_elastic_fast_yield_checks",
            )

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

    def _active_set_return(
        self,
        stress_trial,
        A,
        B,
        cohesion_term,
        profiler=None,
    ):
        n_active = A.shape[1]
        f_trial = A.T @ stress_trial - cohesion_term * np.ones(n_active)

        M = A.T @ self._Cp @ B
        solve_start = _mc_profile_start(profiler)
        dlambda = np.linalg.solve(M, f_trial)
        Y = np.linalg.solve(M, A.T)
        _mc_profile_add(
            profiler,
            "mc_active_set_linear_solve",
            solve_start,
            count_name="mc_active_set_linear_solve_calls",
            count=2,
        )

        dstrain_p = B @ dlambda
        stress_new = stress_trial - self._Cp @ dstrain_p

        # H = d sigma_new_principal / d sigma_trial_principal
        H = np.eye(3) - self._Cp @ B @ Y
        Dalg_principal = H @ self._Cp

        return stress_new, dstrain_p, dlambda, Dalg_principal, H

    @staticmethod
    def _form_dep_perfect(D, normal, flow_direction):
        """Perfect-plastic tangent for one active direction.

        This is the GeoPyFEM equivalent of the reference Fortran
        ``FormDepPerfect`` routine.  For ``normal == flow_direction`` it
        produces a symmetric tangent that is singular in that direction.
        """
        D = np.asarray(D, dtype=float)
        normal = np.asarray(normal, dtype=float).reshape(-1)
        flow_direction = np.asarray(flow_direction, dtype=float).reshape(-1)

        Dg = D @ flow_direction
        fD = normal @ D
        denominator = float(normal @ Dg)

        scale = max(
            1.0,
            float(np.linalg.norm(normal)),
            float(np.linalg.norm(flow_direction)),
            float(np.linalg.norm(D, ord=2)),
        )
        if abs(denominator) <= np.finfo(float).eps * scale:
            raise np.linalg.LinAlgError(
                "Degenerate direction in MohrCoulomb._form_dep_perfect()."
            )

        return D - np.outer(Dg, fD) / denominator

    def _fortran_apex_return(self, stress_trial, p_apex, f13_trial):
        """Direct apex projection plus modified single-singular tangent.

        Mirrors the default apex treatment of the reference Fortran code:

        - project all three principal stresses to the apex;
        - form the plastic-corrector stress ``SiPla``;
        - map it through the elastic normal compliance to obtain the Koiter
          direction;
        - form a single-singular perfect-plastic tangent in that direction;
        - divide the tangent by ``apex_alpha`` (Fortran default: 17).

        No unique set of apex plastic multipliers exists in this modified
        treatment, so ``dlambda`` remains empty.
        """
        stress_trial = np.asarray(stress_trial, dtype=float).reshape(3)
        stress_new = np.full(3, float(p_apex), dtype=float)

        # Fortran: SiPla = SigP - SigP_up
        correction_stress = stress_trial - stress_new

        Cp_inv = np.linalg.inv(self._Cp)

        # Fortran: KoitDir = Dninv * SiPla
        koiter_direction = Cp_inv @ correction_stress

        # Near an already-corrected apex, roundoff can leave an extremely
        # small but nonzero Koiter direction.  Treat that state as the exact
        # zero-correction branch instead of sending a numerically degenerate
        # direction to FormDepPerfect.  This is especially important during
        # repeated safety-factor continuation close to collapse.
        koiter_norm = float(np.linalg.norm(koiter_direction))
        koiter_scale = max(
            1.0,
            float(np.linalg.norm(Cp_inv, ord=2))
            * max(
                float(np.linalg.norm(stress_trial)),
                float(np.linalg.norm(stress_new)),
                1.0,
            ),
        )
        koiter_tolerance = np.sqrt(np.finfo(float).eps) * koiter_scale

        if koiter_norm > koiter_tolerance:
            Dnorm = self._form_dep_perfect(
                self._Cp,
                koiter_direction,
                koiter_direction,
            )
            Dalg_principal = Dnorm / self.apex_alpha
            H = Dalg_principal @ Cp_inv
        else:
            Dalg_principal = np.zeros((3, 3), dtype=float)
            H = np.zeros((3, 3), dtype=float)

        return {
            "stress": stress_new,
            "region": "apex_fortran",
            "yielded": True,
            "plastic_strain_increment_principal": koiter_direction,
            "dlambda": np.zeros(0, dtype=float),
            "tangent_principal": Dalg_principal,
            "principal_jacobian": H,
            "yield_function_trial": float(f13_trial),
            "apex_correction_stress_principal": correction_stress,
            "apex_koiter_direction_principal": koiter_direction,
        }

    def _principal_return(self, stress_trial, profiler=None):
        profile_start = _mc_profile_start(profiler)
        try:
            setup_start = _mc_profile_start(profiler)
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
            _mc_profile_add(
                profiler,
                "mc_yield_setup",
                setup_start,
                count_name="mc_yield_checks",
            )

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

            face_start = _mc_profile_start(profiler)
            A_face = a13.reshape(3, 1)
            B_face = b13.reshape(3, 1)
            (
                stress_face,
                deps_face,
                dlambda_face,
                Dalg_face,
                H_face,
            ) = self._active_set_return(
                stress_trial,
                A_face,
                B_face,
                C,
                profiler=profiler,
            )
            face_valid = (
                self._is_ordered(stress_face, tol)
                and np.all(dlambda_face >= -tol)
            )
            _mc_profile_add(
                profiler,
                "mc_face_candidate",
                face_start,
                count_name="mc_face_candidate_evaluations",
            )

            if face_valid:
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

            compression_edge_start = _mc_profile_start(profiler)
            A_ce = np.column_stack([a13, a12])
            B_ce = np.column_stack([b13, b12])
            (
                stress_ce,
                deps_ce,
                dlambda_ce,
                Dalg_ce,
                H_ce,
            ) = self._active_set_return(
                stress_trial,
                A_ce,
                B_ce,
                C,
                profiler=profiler,
            )
            compression_edge_valid = self._valid_compression_edge(
                stress_ce,
                dlambda_ce,
                tol,
            )
            _mc_profile_add(
                profiler,
                "mc_compression_edge_candidate",
                compression_edge_start,
                count_name="mc_compression_edge_candidate_evaluations",
            )

            if compression_edge_valid:
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

            extension_edge_start = _mc_profile_start(profiler)
            A_ee = np.column_stack([a13, a23])
            B_ee = np.column_stack([b13, b23])
            (
                stress_ee,
                deps_ee,
                dlambda_ee,
                Dalg_ee,
                H_ee,
            ) = self._active_set_return(
                stress_trial,
                A_ee,
                B_ee,
                C,
                profiler=profiler,
            )
            extension_edge_valid = self._valid_extension_edge(
                stress_ee,
                dlambda_ee,
                tol,
            )
            _mc_profile_add(
                profiler,
                "mc_extension_edge_candidate",
                extension_edge_start,
                count_name="mc_extension_edge_candidate_evaluations",
            )

            if extension_edge_valid:
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

            apex_start = _mc_profile_start(profiler)
            try:
                # Apex return.  The stress projection is the same closed-form
                # apex projection used by the reference Fortran implementation.
                if self.friction_angle <= 0.0:
                    raise NotImplementedError(
                        "MohrCoulomb reached the apex branch with phi=0. "
                        "No finite cohesive MC apex exists for phi=0."
                    )

                phi = np.radians(self.friction_angle)
                p_apex = -self.cohesion / np.tan(phi)

                if self.apex_mode == "error":
                    raise NotImplementedError(
                        "MohrCoulomb entered the apex region while "
                        "apex_mode='error'."
                    )

                if self.apex_mode == "projection":
                    return {
                        "stress": np.array(
                            [p_apex, p_apex, p_apex],
                            dtype=float,
                        ),
                        "region": "apex_projection",
                        "yielded": True,
                        "plastic_strain_increment_principal": None,
                        "dlambda": np.zeros(0, dtype=float),
                        "tangent_principal": np.zeros((3, 3), dtype=float),
                        "principal_jacobian": np.zeros((3, 3), dtype=float),
                        "yield_function_trial": float(f13_trial),
                    }

                return self._fortran_apex_return(
                    stress_trial=stress_trial,
                    p_apex=p_apex,
                    f13_trial=f13_trial,
                )
            finally:
                _mc_profile_add(
                    profiler,
                    "mc_apex_return",
                    apex_start,
                    count_name="mc_apex_return_evaluations",
                )
        finally:
            _mc_profile_add(
                profiler,
                "mc_principal_return",
                profile_start,
                count_name="mc_principal_return_calls",
            )

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
        profiler=None,
    ):
        """
        Full 6x6 tangent in the same numerical convention as GeoPyFEM.

        The internal compression-positive transformation changes both stress
        and strain signs, so d(sigma_tension)/d(epsilon_tension) has the same
        numerical tangent as the compression-positive derivative.
        """
        profile_start = _mc_profile_start(profiler)
        try:
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
                        if (
                            abs(denominator)
                            < self.repeated_eigenvalue_tolerance * scale
                        ):
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
        finally:
            _mc_profile_add(
                profiler,
                "mc_analytic_spectral_tangent",
                profile_start,
                count_name="mc_analytic_spectral_tangent_calls",
            )

    def _apex_cartesian_tangent_6d(
        self,
        directions,
        H,
        profiler=None,
    ):
        """Rotate the modified Fortran-style apex tangent to Cartesian form.

        At the apex all updated principal stresses are equal.  The reference
        Fortran algorithm therefore nulls the spectral shear contribution via
        its principal-space modification matrix ``T``.  This routine applies
        only the modified principal normal tangent and sets those spectral
        shear terms to zero, avoiding divisions by repeated principal stresses.
        """
        profile_start = _mc_profile_start(profiler)
        try:
            directions = np.asarray(directions, dtype=float).reshape(3, 3)
            H = np.asarray(H, dtype=float).reshape(3, 3)
            Dcart = np.zeros((6, 6), dtype=float)

            for j in range(6):
                deps = np.zeros(6, dtype=float)
                deps[j] = 1.0

                dstress_trial_vec = self._C6 @ deps
                dstress_trial_tensor = self._stress_voigt6_to_tensor(
                    dstress_trial_vec
                )
                dtrial_tensor_p = (
                    directions.T @ dstress_trial_tensor @ directions
                )

                dnew_diag = H @ np.diag(dtrial_tensor_p)
                dnew_tensor_p = np.diag(dnew_diag)

                dnew_cart = directions @ dnew_tensor_p @ directions.T
                Dcart[:, j] = self._stress_tensor_to_voigt6(dnew_cart)

            return Dcart
        finally:
            _mc_profile_add(
                profiler,
                "mc_apex_tangent",
                profile_start,
                count_name="mc_apex_tangent_calls",
            )

    # =========================================================================
    # LOCAL STRESS INTEGRATION
    # =========================================================================

    def _integrate_local_result(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
        profiler=None,
    ):
        """Perform one local stress update without building the 2-D tangent.

        The dominant elastic path first evaluates only the three principal
        *values* analytically.  Full ``np.linalg.eigh`` (and therefore
        principal directions) is deferred until the trial state is plastic or
        lies inside a tiny roundoff guard around the yield tolerance.

        Persistent constitutive/output data are written to ``state``.
        Spectral quantities needed only by a plastic tangent remain transient
        in the returned local-result object.
        """
        profile_start = _mc_profile_start(profiler)
        _mc_profile_increment(profiler, "mc_state_integrations")
        try:
            predictor_start = _mc_profile_start(profiler)
            total_strain_2d = np.asarray(
                total_strain_2d,
                dtype=float,
            ).reshape(3)
            strain_increment_2d = np.asarray(
                strain_increment_2d,
                dtype=float,
            ).reshape(3)

            total_strain_6 = plane_strain_2d_to_voigt6(total_strain_2d)
            dstrain_6 = plane_strain_2d_to_voigt6(strain_increment_2d)

            stress_old_tp = self._effective_stress_from_state(committed_state)
            stress_trial_tp = stress_old_tp + self._C6 @ dstrain_6

            # GeoPyFEM is tension-positive.  The verified MC return is
            # internally compression-positive; convert only inside the local
            # integration.
            stress_trial_cp = -stress_trial_tp
            _mc_profile_add(
                profiler,
                "mc_elastic_predictor",
                predictor_start,
            )

            # -------------------------------------------------------------
            # Elastic fast path: principal values only, no eigenvectors.
            # -------------------------------------------------------------
            (
                principal_fast_cp,
                f_trial_fast,
                definitely_elastic,
            ) = self._elastic_fast_trial_check(
                stress_trial_cp,
                profiler=profiler,
            )

            directions = None
            if definitely_elastic:
                principal_trial_cp = principal_fast_cp
                principal_new_cp = principal_fast_cp
                stress_new_tp = stress_trial_tp
                result = {
                    "stress": principal_new_cp,
                    "region": "elastic",
                    "yielded": False,
                    "plastic_strain_increment_principal": None,
                    "dlambda": np.zeros(0, dtype=float),
                    "principal_jacobian": None,
                    "yield_function_trial": float(f_trial_fast),
                }
            else:
                # Plastic (or near-yield) states retain the previously
                # validated full spectral return mapping unchanged.
                principal_trial_cp, directions = (
                    self._principal_decomposition_compression_positive(
                        stress_trial_cp,
                        profiler=profiler,
                    )
                )
                result = self._principal_return(
                    principal_trial_cp,
                    profiler=profiler,
                )
                principal_new_cp = result["stress"]

                if result["yielded"]:
                    reconstruction_start = _mc_profile_start(profiler)
                    stress_new_cp = self._principal_to_cartesian(
                        principal_new_cp,
                        directions,
                    )
                    stress_new_tp = -stress_new_cp
                    _mc_profile_add(
                        profiler,
                        "mc_stress_reconstruction",
                        reconstruction_start,
                    )
                else:
                    # Near-yield fallback classified elastic by the original
                    # full spectral path.  The Cartesian trial stress is exact
                    # and needs no reconstruction.
                    stress_new_tp = stress_trial_tp
                    directions = None

            state_update_start = _mc_profile_start(profiler)
            trial_state = committed_state.copy()
            trial_state.strain[:] = total_strain_6
            self._set_effective_stress(trial_state, stress_new_tp)
            trial_state.yielded = bool(result["yielded"])
            _mc_profile_add(
                profiler,
                "mc_state_copy_update",
                state_update_start,
            )

            # Accumulate plastic strain only for yielded branches.
            plastic_start = _mc_profile_start(profiler)
            deps_p_principal_cp = result[
                "plastic_strain_increment_principal"
            ]
            if result["yielded"] and deps_p_principal_cp is not None:
                deps_p_tensor_cp = (
                    directions
                    @ np.diag(deps_p_principal_cp)
                    @ directions.T
                )
                deps_p_tensor_tp = -deps_p_tensor_cp
                deps_p_voigt_tp = self._strain_tensor_to_voigt6(
                    deps_p_tensor_tp
                )
                trial_state.plastic_strain[:] = (
                    np.asarray(
                        committed_state.plastic_strain,
                        dtype=float,
                    )
                    + deps_p_voigt_tp
                )
            _mc_profile_add(
                profiler,
                "mc_plastic_strain_update",
                plastic_start,
            )

            diagnostics_start = _mc_profile_start(profiler)
            if result["yielded"]:
                a13, _, _, _, _, _ = self._mc_plane_vectors()
                C = self._cohesion_term()
                yield_function_new = self._yield_value(
                    principal_new_cp,
                    a13,
                    C,
                )
            else:
                # For an elastic return the updated state is the trial state,
                # so no second MC geometry evaluation is required.
                yield_function_new = float(result["yield_function_trial"])

            q_new = self._q_from_principal(principal_new_cp)
            p_new_cp = float(np.mean(principal_new_cp))
            dlambda = np.asarray(result["dlambda"], dtype=float).reshape(-1)
            plastic_multiplier = float(
                np.sum(dlambda) if dlambda.size else 0.0
            )

            trial_state.internal_variables["mc_region"] = result["region"]
            trial_state.internal_variables["yield_function_trial"] = float(
                result["yield_function_trial"]
            )
            trial_state.internal_variables["yield_function"] = float(
                yield_function_new
            )
            trial_state.internal_variables["plastic_multiplier"] = (
                plastic_multiplier
            )
            trial_state.internal_variables["q"] = float(q_new)
            trial_state.internal_variables["p_compression_positive"] = (
                p_new_cp
            )
            _mc_profile_add(
                profiler,
                "mc_state_diagnostics",
                diagnostics_start,
            )

            return _MohrCoulombLocalResult(
                state=trial_state,
                region=str(result["region"]),
                yield_function_trial=float(result["yield_function_trial"]),
                yield_function=float(yield_function_new),
                plastic_multiplier=plastic_multiplier,
                q=float(q_new),
                p_compression_positive=p_new_cp,
                principal_trial_compression_positive=principal_trial_cp,
                principal_new_compression_positive=principal_new_cp,
                principal_directions=directions,
                principal_jacobian=result["principal_jacobian"],
            )
        finally:
            _mc_profile_add(
                profiler,
                "mc_state_integration",
                profile_start,
            )

    def _integrate_state(
        self,
        total_strain_2d,
        strain_increment_2d,
        committed_state,
        profiler=None,
    ):
        """Backward-compatible local-map helper returning only the state."""
        return self._integrate_local_result(
            total_strain_2d=total_strain_2d,
            strain_increment_2d=strain_increment_2d,
            committed_state=committed_state,
            profiler=profiler,
        ).state

    # =========================================================================
    # NUMERICAL 2-D TANGENT FALLBACK
    # =========================================================================

    def _numerical_algorithmic_tangent_2d(
        self,
        total_strain_2d,
        committed_state,
        profiler=None,
    ):
        """Differentiate the exact 2-D local constitutive map numerically."""
        profile_start = _mc_profile_start(profiler)
        _mc_profile_increment(profiler, "mc_numerical_tangent_fallbacks")
        try:
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
                    profiler=profiler,
                )
                state_minus = self._integrate_state(
                    epsilon_minus,
                    epsilon_minus - committed_state.strain_2d,
                    committed_state,
                    profiler=profiler,
                )

                sigma_plus = self._stress_2d_from_state(state_plus)
                sigma_minus = self._stress_2d_from_state(state_minus)

                C_alg[:, j] = (sigma_plus - sigma_minus) / (2.0 * h)

            # Do NOT symmetrize: non-associated Mohr-Coulomb generally has a
            # genuinely non-symmetric algorithmic tangent.
            return C_alg
        finally:
            _mc_profile_add(
                profiler,
                "mc_numerical_tangent_fallback",
                profile_start,
            )

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
        profiler = (
            None
            if context is None
            else getattr(context, "performance_profiler", None)
        )
        profile_start = _mc_profile_start(profiler)
        _mc_profile_increment(profiler, "mc_constitutive_calls")

        try:
            total_strain_2d = np.asarray(
                total_strain_2d,
                dtype=float,
            ).reshape(3)
            strain_increment_2d = np.asarray(
                strain_increment_2d,
                dtype=float,
            ).reshape(3)

            local_result = self._integrate_local_result(
                total_strain_2d,
                strain_increment_2d,
                committed_state,
                profiler=profiler,
            )
            trial_state = local_result.state
            region = local_result.region
            _mc_profile_increment(profiler, f"mc_region_{region}")
            if trial_state.yielded:
                _mc_profile_increment(profiler, "mc_plastic_calls")
            else:
                _mc_profile_increment(profiler, "mc_elastic_calls")

            tangent_start = _mc_profile_start(profiler)
            tangent_mode = "elastic"

            if not trial_state.yielded:
                C_alg_2d = self._C_plane_strain.copy()

            elif region == "apex_projection":
                # Legacy diagnostic mode only.
                C_alg_2d = np.zeros((3, 3), dtype=float)
                tangent_mode = "apex_zero_legacy_projection"

            elif region == "apex_fortran":
                directions = local_result.principal_directions
                H = local_result.principal_jacobian
                C_alg_6d = self._apex_cartesian_tangent_6d(
                    directions=directions,
                    H=H,
                    profiler=profiler,
                )
                C_alg_2d = C_alg_6d[
                    np.ix_(
                        self._PLANE_STRAIN_INDICES,
                        self._PLANE_STRAIN_INDICES,
                    )
                ].copy()
                tangent_mode = "fortran_modified_apex_single_singular"

            else:
                principal_trial_cp = (
                    local_result.principal_trial_compression_positive
                )
                principal_new_cp = (
                    local_result.principal_new_compression_positive
                )
                directions = local_result.principal_directions
                H = local_result.principal_jacobian

                if self._has_repeated_principal_stress(principal_trial_cp):
                    C_alg_2d = self._numerical_algorithmic_tangent_2d(
                        total_strain_2d,
                        committed_state,
                        profiler=profiler,
                    )
                    tangent_mode = "numerical_repeated_eigenvalue_fallback"
                else:
                    C_alg_6d = self._analytic_cartesian_tangent_6d(
                        principal_trial_cp,
                        principal_new_cp,
                        directions,
                        H,
                        profiler=profiler,
                    )
                    C_alg_2d = C_alg_6d[
                        np.ix_(
                            self._PLANE_STRAIN_INDICES,
                            self._PLANE_STRAIN_INDICES,
                        )
                    ].copy()
                    tangent_mode = "analytic_spectral"

            _mc_profile_add(
                profiler,
                "mc_tangent_total",
                tangent_start,
            )
            _mc_profile_increment(
                profiler,
                f"mc_tangent_mode_{tangent_mode}",
            )

            # Keep the lightweight mode label for backward-compatible
            # post-run inspection; the heavy spectral work arrays remain
            # transient in ``local_result``.
            trial_state.internal_variables["tangent_mode"] = tangent_mode

            response_start = _mc_profile_start(profiler)
            response = ConstitutiveResponse(
                state=trial_state,
                tangent=C_alg_2d,
                diagnostics={
                    "yielded": bool(trial_state.yielded),
                    "region": region,
                    "tangent_mode": tangent_mode,
                    "yield_function": local_result.yield_function,
                    "yield_function_trial": (
                        local_result.yield_function_trial
                    ),
                    "plastic_multiplier": local_result.plastic_multiplier,
                    "q": local_result.q,
                    "p_compression_positive": (
                        local_result.p_compression_positive
                    ),
                },
            )
            _mc_profile_add(
                profiler,
                "mc_response_construction",
                response_start,
            )
            return response
        finally:
            _mc_profile_add(
                profiler,
                "mc_integrate_total",
                profile_start,
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
