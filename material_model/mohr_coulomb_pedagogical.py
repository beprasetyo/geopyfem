import numpy as np


# ============================================================
# PEDAGOGICAL MOHR-COULOMB CONSTITUTIVE MODEL
#
# Sign convention:
#   compression stress = positive
#   compression strain = positive
#
# Cartesian Voigt ordering:
#   stress = [xx, yy, zz, xy, yz, xz]
#   strain = [xx, yy, zz, gamma_xy, gamma_yz, gamma_xz]
#
# Principal ordering:
#   sigma1 >= sigma2 >= sigma3
#
# Features:
#   - 3D principal-stress return mapping
#   - non-associated flow allowed: psi != phi
#   - face return
#   - compression-edge return
#   - extension-edge return
#   - pedagogical fixed-apex projection
#   - analytic principal tangent
#   - analytic Cartesian spectral tangent
#   - numerical tangent verification
#   - repeated-eigenvalue numerical fallback
#
# IMPORTANT:
#   This is still a pedagogical / production-candidate model.
#
#   The following are NOT yet considered final production
#   formulations:
#
#       1. non-associated apex return
#       2. repeated-eigenvalue analytic spectral limit
#
# ============================================================


# ============================================================
# 1. ELASTICITY
# ============================================================

def elastic_matrix_3d(E, nu):
    """
    3D isotropic elastic constitutive matrix.

    Stress Voigt ordering:
        [xx, yy, zz, xy, yz, xz]

    Strain uses engineering shear strains:
        [exx, eyy, ezz, gamma_xy, gamma_yz, gamma_xz]
    """

    G = E / (
        2.0 * (1.0 + nu)
    )

    lam = (
        E * nu
        / (
            (1.0 + nu)
            * (1.0 - 2.0 * nu)
        )
    )

    D = np.array([
        [
            lam + 2.0 * G,
            lam,
            lam,
            0.0,
            0.0,
            0.0,
        ],

        [
            lam,
            lam + 2.0 * G,
            lam,
            0.0,
            0.0,
            0.0,
        ],

        [
            lam,
            lam,
            lam + 2.0 * G,
            0.0,
            0.0,
            0.0,
        ],

        [
            0.0,
            0.0,
            0.0,
            G,
            0.0,
            0.0,
        ],

        [
            0.0,
            0.0,
            0.0,
            0.0,
            G,
            0.0,
        ],

        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            G,
        ],
    ], dtype=float)

    return D


def principal_elastic_matrix(
    E,
    nu,
):
    """
    Normal elastic matrix in principal space.

    [sigma1]   [    ] [epsilon1]
    [sigma2] = [ Dp ] [epsilon2]
    [sigma3]   [    ] [epsilon3]
    """

    D = elastic_matrix_3d(
        E,
        nu,
    )

    return D[:3, :3]


# ============================================================
# 2. STRESS REPRESENTATION
# ============================================================

def stress_vector_to_tensor(
    stress,
):
    """
    Convert:

        [sxx, syy, szz, txy, tyz, txz]

    into symmetric 3x3 stress tensor.
    """

    (
        sxx,
        syy,
        szz,
        txy,
        tyz,
        txz,
    ) = stress

    tensor = np.array([
        [
            sxx,
            txy,
            txz,
        ],

        [
            txy,
            syy,
            tyz,
        ],

        [
            txz,
            tyz,
            szz,
        ],
    ], dtype=float)

    return tensor


def stress_tensor_to_vector(
    tensor,
):
    """
    Convert symmetric 3x3 stress tensor
    into Cartesian Voigt vector.
    """

    stress = np.array([
        tensor[0, 0],
        tensor[1, 1],
        tensor[2, 2],
        tensor[0, 1],
        tensor[1, 2],
        tensor[0, 2],
    ], dtype=float)

    return stress


# ============================================================
# 3. PRINCIPAL STRESS DECOMPOSITION
# ============================================================

def principal_decomposition(
    stress,
):
    """
    Calculate principal stresses and principal directions.

    Compression-positive ordering:

        sigma1 >= sigma2 >= sigma3
    """

    stress_tensor = (
        stress_vector_to_tensor(
            stress
        )
    )

    values, vectors = (
        np.linalg.eigh(
            stress_tensor
        )
    )

    # np.linalg.eigh gives ascending eigenvalues.
    # We use descending principal stresses.

    order = np.argsort(
        values
    )[::-1]

    values = values[
        order
    ]

    vectors = vectors[
        :,
        order,
    ]

    return (
        values,
        vectors,
    )


def principal_to_cartesian(
    principal_stress,
    directions,
):
    """
    Reconstruct Cartesian stress:

        sigma =
        Q diag(sigma1,sigma2,sigma3) Q.T
    """

    tensor = (
        directions
        @ np.diag(
            principal_stress
        )
        @ directions.T
    )

    return (
        stress_tensor_to_vector(
            tensor
        )
    )


# ============================================================
# 4. MOHR-COULOMB SURFACES
# ============================================================

def mc_plane_vectors(
    phi,
    psi,
):
    """
    Construct Mohr-Coulomb yield normals
    and plastic-flow directions.

    Angles are in degrees.
    """

    phi_rad = np.radians(
        phi
    )

    psi_rad = np.radians(
        psi
    )

    # --------------------------------------------------------
    # Yield surface
    #
    # f =
    # (1-sin(phi))*sigma1
    # -(1+sin(phi))*sigma3
    # -2c*cos(phi)
    # --------------------------------------------------------

    A = (
        1.0
        - np.sin(
            phi_rad
        )
    )

    B = (
        1.0
        + np.sin(
            phi_rad
        )
    )

    # --------------------------------------------------------
    # Plastic potential
    # --------------------------------------------------------

    Ap = (
        1.0
        - np.sin(
            psi_rad
        )
    )

    Bp = (
        1.0
        + np.sin(
            psi_rad
        )
    )

    # --------------------------------------------------------
    # Yield normals
    # --------------------------------------------------------

    a13 = np.array([
        A,
        0.0,
        -B,
    ], dtype=float)

    a12 = np.array([
        A,
        -B,
        0.0,
    ], dtype=float)

    a23 = np.array([
        0.0,
        A,
        -B,
    ], dtype=float)

    # --------------------------------------------------------
    # Plastic-flow directions
    # --------------------------------------------------------

    b13 = np.array([
        Ap,
        0.0,
        -Bp,
    ], dtype=float)

    b12 = np.array([
        Ap,
        -Bp,
        0.0,
    ], dtype=float)

    b23 = np.array([
        0.0,
        Ap,
        -Bp,
    ], dtype=float)

    return (
        a13,
        a12,
        a23,
        b13,
        b12,
        b23,
    )


def cohesion_term(
    c,
    phi,
):
    """
    Mohr-Coulomb cohesion term:

        C = 2 c cos(phi)
    """

    phi_rad = np.radians(
        phi
    )

    return (
        2.0
        * c
        * np.cos(
            phi_rad
        )
    )


def yield_value(
    stress_p,
    a,
    C,
):
    """
    Evaluate one linear Mohr-Coulomb plane:

        f = a.T sigma - C
    """

    return float(
        a @ stress_p
        - C
    )


# ============================================================
# 5. GENERIC ACTIVE-SET RETURN
# ============================================================

def active_set_return(
    stress_trial,
    Dp,
    A,
    B,
    C,
):
    """
    Generic active-set return mapping.

    A:
        active yield normals

    B:
        active plastic-flow directions

    For face:
        A.shape = (3,1)
        B.shape = (3,1)

    For edge:
        A.shape = (3,2)
        B.shape = (3,2)

    Returns
    -------
    stress_new

    dstrain_p

    dlambda

    Dalg
        principal algorithmic tangent:

        d sigma_new
        -----------
        d epsilon_principal

    H
        principal stress Jacobian:

        d sigma_new
        -----------
        d sigma_trial
    """

    n_active = (
        A.shape[1]
    )

    # ========================================================
    # Trial yield values
    #
    # f_trial =
    # A.T sigma_trial - C
    # ========================================================

    f_trial = (
        A.T
        @ stress_trial
        - C
        * np.ones(
            n_active
        )
    )

    # ========================================================
    # Consistency matrix
    #
    # M = A.T D B
    # ========================================================

    M = (
        A.T
        @ Dp
        @ B
    )

    # ========================================================
    # Plastic multipliers
    #
    # M dlambda = f_trial
    # ========================================================

    dlambda = (
        np.linalg.solve(
            M,
            f_trial,
        )
    )

    # ========================================================
    # Plastic strain increment
    #
    # deps_p = B dlambda
    # ========================================================

    dstrain_p = (
        B
        @ dlambda
    )

    # ========================================================
    # Corrected principal stress
    #
    # sigma_new =
    # sigma_trial - D deps_p
    # ========================================================

    stress_new = (
        stress_trial
        - Dp
        @ dstrain_p
    )

    # ========================================================
    # PRINCIPAL-STRESS JACOBIAN
    #
    # H =
    #
    # d sigma_new
    # -----------
    # d sigma_trial
    #
    # H =
    # I - D B (A.T D B)^(-1) A.T
    # ========================================================

    Y = (
        np.linalg.solve(
            M,
            A.T,
        )
    )

    H = (
        np.eye(3)
        - Dp
        @ B
        @ Y
    )

    # ========================================================
    # PRINCIPAL ALGORITHMIC TANGENT
    #
    # Dalg =
    #
    # d sigma_new
    # -----------
    # d epsilon
    #
    # Dalg = H Dp
    # ========================================================

    Dalg = (
        H
        @ Dp
    )

    return (
        stress_new,
        dstrain_p,
        dlambda,
        Dalg,
        H,
    )


# ============================================================
# 6. ADMISSIBILITY CHECKS
# ============================================================

def is_ordered(
    stress,
    tol=1e-10,
):
    """
    Check principal ordering:

        sigma1 >= sigma2 >= sigma3
    """

    (
        s1,
        s2,
        s3,
    ) = stress

    return (
        s1 >= s2 - tol
        and
        s2 >= s3 - tol
    )


def valid_compression_edge(
    stress,
    dlambda,
    tol=1e-8,
):
    """
    Compression edge:

        sigma1 >= sigma2 = sigma3
    """

    (
        s1,
        s2,
        s3,
    ) = stress

    stress_ok = (
        s1 >= s2 - tol
        and
        abs(
            s2 - s3
        ) <= tol
    )

    multiplier_ok = (
        np.all(
            dlambda
            >= -tol
        )
    )

    return (
        stress_ok
        and
        multiplier_ok
    )


def valid_extension_edge(
    stress,
    dlambda,
    tol=1e-8,
):
    """
    Extension edge:

        sigma1 = sigma2 >= sigma3
    """

    (
        s1,
        s2,
        s3,
    ) = stress

    stress_ok = (
        abs(
            s1 - s2
        ) <= tol
        and
        s2 >= s3 - tol
    )

    multiplier_ok = (
        np.all(
            dlambda
            >= -tol
        )
    )

    return (
        stress_ok
        and
        multiplier_ok
    )


# ============================================================
# 7. PRINCIPAL-STRESS RETURN MAPPING
# ============================================================

def principal_return_mohr_coulomb(
    stress_trial,
    Dp,
    c,
    phi,
    psi,
    tol=1e-8,
):
    """
    Mohr-Coulomb return mapping
    in ordered principal-stress space.

    Possible regions:

        elastic
        face
        compression_edge
        extension_edge
        apex_projection
    """

    (
        a13,
        a12,
        a23,
        b13,
        b12,
        b23,
    ) = mc_plane_vectors(
        phi,
        psi,
    )

    C = cohesion_term(
        c,
        phi,
    )

    # ========================================================
    # 7.1 ELASTIC CHECK
    # ========================================================

    f13_trial = (
        yield_value(
            stress_trial,
            a13,
            C,
        )
    )

    if (
        f13_trial
        <= tol
    ):

        return {
            "stress":
                stress_trial.copy(),

            "region":
                "elastic",

            "yielded":
                False,

            "plastic_strain_increment":
                np.zeros(3),

            "dlambda":
                None,

            "tangent_principal":
                Dp.copy(),

            "principal_jacobian":
                np.eye(3),
        }

    # ========================================================
    # 7.2 FACE CANDIDATE
    # ========================================================

    A_face = (
        a13.reshape(
            3,
            1,
        )
    )

    B_face = (
        b13.reshape(
            3,
            1,
        )
    )

    (
        stress_face,
        deps_face,
        dlambda_face,
        Dalg_face,
        H_face,
    ) = active_set_return(
        stress_trial,
        Dp,
        A_face,
        B_face,
        C,
    )

    if is_ordered(
        stress_face,
        tol,
    ):

        return {
            "stress":
                stress_face,

            "region":
                "face",

            "yielded":
                True,

            "plastic_strain_increment":
                deps_face,

            "dlambda":
                dlambda_face,

            "tangent_principal":
                Dalg_face,

            "principal_jacobian":
                H_face,
        }

    # ========================================================
    # 7.3 COMPRESSION EDGE
    #
    # Active surfaces:
    #
    # f13 = 0
    # f12 = 0
    #
    # Corrected condition:
    #
    # sigma1 > sigma2 = sigma3
    # ========================================================

    A_ce = (
        np.column_stack([
            a13,
            a12,
        ])
    )

    B_ce = (
        np.column_stack([
            b13,
            b12,
        ])
    )

    (
        stress_ce,
        deps_ce,
        dlambda_ce,
        Dalg_ce,
        H_ce,
    ) = active_set_return(
        stress_trial,
        Dp,
        A_ce,
        B_ce,
        C,
    )

    if valid_compression_edge(
        stress_ce,
        dlambda_ce,
        tol,
    ):

        return {
            "stress":
                stress_ce,

            "region":
                "compression_edge",

            "yielded":
                True,

            "plastic_strain_increment":
                deps_ce,

            "dlambda":
                dlambda_ce,

            "tangent_principal":
                Dalg_ce,

            "principal_jacobian":
                H_ce,
        }

    # ========================================================
    # 7.4 EXTENSION EDGE
    #
    # Active surfaces:
    #
    # f13 = 0
    # f23 = 0
    #
    # Corrected condition:
    #
    # sigma1 = sigma2 > sigma3
    # ========================================================

    A_ee = (
        np.column_stack([
            a13,
            a23,
        ])
    )

    B_ee = (
        np.column_stack([
            b13,
            b23,
        ])
    )

    (
        stress_ee,
        deps_ee,
        dlambda_ee,
        Dalg_ee,
        H_ee,
    ) = active_set_return(
        stress_trial,
        Dp,
        A_ee,
        B_ee,
        C,
    )

    if valid_extension_edge(
        stress_ee,
        dlambda_ee,
        tol,
    ):

        return {
            "stress":
                stress_ee,

            "region":
                "extension_edge",

            "yielded":
                True,

            "plastic_strain_increment":
                deps_ee,

            "dlambda":
                dlambda_ee,

            "tangent_principal":
                Dalg_ee,

            "principal_jacobian":
                H_ee,
        }

    # ========================================================
    # 7.5 APEX
    #
    # Pedagogical geometric fixed-apex projection.
    #
    # IMPORTANT:
    #
    # This is NOT yet considered a rigorous final
    # non-associated Mohr-Coulomb apex algorithm.
    # ========================================================

    phi_rad = np.radians(
        phi
    )

    if (
        abs(
            np.sin(
                phi_rad
            )
        )
        < 1e-14
    ):

        raise ValueError(
            "Mohr-Coulomb apex is "
            "undefined for phi = 0."
        )

    p_apex = (
        -c
        / np.tan(
            phi_rad
        )
    )

    stress_apex = np.array([
        p_apex,
        p_apex,
        p_apex,
    ])

    return {
        "stress":
            stress_apex,

        "region":
            "apex_projection",

        "yielded":
            True,

        "plastic_strain_increment":
            None,

        "dlambda":
            None,

        "tangent_principal":
            np.zeros(
                (3, 3)
            ),

        "principal_jacobian":
            np.zeros(
                (3, 3)
            ),
    }


# ============================================================
# 8. REPEATED PRINCIPAL STRESS CHECK
# ============================================================

def has_repeated_principal_stress(
    principal_stress,
    rel_tol=1e-8,
):
    """
    Detect repeated or nearly repeated
    principal stresses.

    Important because the simple spectral derivative
    contains terms:

        (sigma_i_new - sigma_j_new)
        ---------------------------
        (sigma_i_trial - sigma_j_trial)

    which become problematic when the denominator
    approaches zero.
    """

    scale = max(
        1.0,
        float(
            np.max(
                np.abs(
                    principal_stress
                )
            )
        ),
    )

    tol = (
        rel_tol
        * scale
    )

    (
        s1,
        s2,
        s3,
    ) = principal_stress

    return (
        abs(
            s1 - s2
        ) < tol
        or
        abs(
            s2 - s3
        ) < tol
        or
        abs(
            s1 - s3
        ) < tol
    )


# ============================================================
# 9. ANALYTIC SPECTRAL CARTESIAN TANGENT
# ============================================================

def analytic_cartesian_tangent(
    D_elastic,
    principal_trial,
    principal_new,
    directions,
    H,
    eig_tol=1e-10,
):
    """
    Construct full 6x6 Cartesian algorithmic tangent
    using spectral differentiation.

    Includes:

        1. change of principal stress magnitudes
        2. rotation of principal directions

    Valid when:

        - active constitutive branch is locally fixed
        - trial principal stresses are distinct
    """

    Dcart = np.zeros(
        (6, 6)
    )

    # --------------------------------------------------------
    # Build Dcart column by column.
    #
    # Each column corresponds to one unit Cartesian
    # strain perturbation.
    # --------------------------------------------------------

    for j in range(6):

        # ====================================================
        # Unit strain perturbation
        # ====================================================

        deps = np.zeros(6)

        deps[j] = 1.0

        # ====================================================
        # Elastic trial-stress differential
        #
        # d sigma_trial =
        # D_elastic d epsilon
        # ====================================================

        dstress_trial_vec = (
            D_elastic
            @ deps
        )

        dstress_trial_tensor = (
            stress_vector_to_tensor(
                dstress_trial_vec
            )
        )

        # ====================================================
        # Transform differential into principal coordinates
        #
        # d sigma_hat =
        # Q.T d sigma_trial Q
        # ====================================================

        dtrial_p = (
            directions.T
            @ dstress_trial_tensor
            @ directions
        )

        dnew_p = np.zeros(
            (3, 3)
        )

        # ====================================================
        # PART 1:
        #
        # CHANGE OF PRINCIPAL STRESS MAGNITUDES
        #
        # d sigma_new_diag =
        # H d sigma_trial_diag
        # ====================================================

        dtrial_diag = (
            np.diag(
                dtrial_p
            )
        )

        dnew_diag = (
            H
            @ dtrial_diag
        )

        for i in range(3):

            dnew_p[
                i,
                i,
            ] = (
                dnew_diag[i]
            )

        # ====================================================
        # PART 2:
        #
        # ROTATION OF PRINCIPAL DIRECTIONS
        #
        # beta_ij =
        #
        # sigma_i_new - sigma_j_new
        # -------------------------
        # sigma_i_trial - sigma_j_trial
        # ====================================================

        for i in range(3):

            for k in range(
                i + 1,
                3,
            ):

                denominator = (
                    principal_trial[i]
                    - principal_trial[k]
                )

                scale = max(
                    1.0,
                    abs(
                        principal_trial[i]
                    ),
                    abs(
                        principal_trial[k]
                    ),
                )

                if (
                    abs(
                        denominator
                    )
                    < eig_tol
                    * scale
                ):

                    raise ValueError(
                        "Repeated or nearly repeated "
                        "trial principal stresses "
                        "encountered in analytic "
                        "spectral tangent."
                    )

                beta = (
                    principal_new[i]
                    - principal_new[k]
                ) / denominator

                value = (
                    beta
                    * dtrial_p[
                        i,
                        k,
                    ]
                )

                dnew_p[
                    i,
                    k,
                ] = value

                dnew_p[
                    k,
                    i,
                ] = value

        # ====================================================
        # Transform corrected differential
        # back to Cartesian coordinates
        # ====================================================

        dnew_cart = (
            directions
            @ dnew_p
            @ directions.T
        )

        Dcart[
            :,
            j,
        ] = (
            stress_tensor_to_vector(
                dnew_cart
            )
        )

    return Dcart


# ============================================================
# 10. STRESS UPDATE CORE
# ============================================================

def stress_update_core(
    stress_old,
    dstrain,
    E,
    nu,
    c,
    phi,
    psi,
    tol=1e-8,
):
    """
    Perform complete stress integration WITHOUT constructing
    the final Cartesian algorithmic tangent.

    The separation is intentional.

    Numerical tangent routines can call stress_update_core()
    repeatedly without causing recursive tangent evaluation.
    """

    # ========================================================
    # Elastic matrices
    # ========================================================

    D = elastic_matrix_3d(
        E,
        nu,
    )

    Dp = principal_elastic_matrix(
        E,
        nu,
    )

    # ========================================================
    # Elastic predictor
    #
    # sigma_trial =
    # sigma_old + D dstrain
    # ========================================================

    stress_trial = (
        stress_old
        + D
        @ dstrain
    )

    # ========================================================
    # Principal decomposition
    # ========================================================

    (
        principal_trial,
        directions,
    ) = (
        principal_decomposition(
            stress_trial
        )
    )

    # ========================================================
    # Mohr-Coulomb return mapping
    # ========================================================

    principal_result = (
        principal_return_mohr_coulomb(
            principal_trial,
            Dp,
            c,
            phi,
            psi,
            tol,
        )
    )

    principal_new = (
        principal_result[
            "stress"
        ]
    )

    # ========================================================
    # Transform corrected stress back to Cartesian
    # ========================================================

    stress_new = (
        principal_to_cartesian(
            principal_new,
            directions,
        )
    )

    return {
        "stress_old":
            stress_old.copy(),

        "strain_increment":
            dstrain.copy(),

        "stress_trial":
            stress_trial,

        "principal_trial":
            principal_trial,

        "principal_new":
            principal_new,

        "principal_directions":
            directions,

        "principal_jacobian":
            principal_result[
                "principal_jacobian"
            ],

        "tangent_principal":
            principal_result[
                "tangent_principal"
            ],

        "stress":
            stress_new,

        "yielded":
            principal_result[
                "yielded"
            ],

        "region":
            principal_result[
                "region"
            ],

        "plastic_strain_increment_principal":
            principal_result[
                "plastic_strain_increment"
            ],

        "dlambda":
            principal_result[
                "dlambda"
            ],
    }


# ============================================================
# 11. NUMERICAL CARTESIAN TANGENT
# ============================================================

def numerical_cartesian_tangent(
    stress_old,
    dstrain,
    E,
    nu,
    c,
    phi,
    psi,
    h=1e-7,
):
    """
    Numerical 6x6 Cartesian tangent using
    central finite differences.

    Main purposes:

        1. verification reference

        2. provisional fallback for repeated
           trial principal stresses

    stress_update_core() is used intentionally so
    there is no tangent recursion.
    """

    Dnum = np.zeros(
        (6, 6)
    )

    for j in range(6):

        dstrain_plus = (
            dstrain.copy()
        )

        dstrain_minus = (
            dstrain.copy()
        )

        dstrain_plus[j] += h

        dstrain_minus[j] -= h

        result_plus = (
            stress_update_core(
                stress_old,
                dstrain_plus,
                E,
                nu,
                c,
                phi,
                psi,
            )
        )

        result_minus = (
            stress_update_core(
                stress_old,
                dstrain_minus,
                E,
                nu,
                c,
                phi,
                psi,
            )
        )

        Dnum[
            :,
            j,
        ] = (
            result_plus[
                "stress"
            ]
            - result_minus[
                "stress"
            ]
        ) / (
            2.0
            * h
        )

    return Dnum


# ============================================================
# 12. COMPLETE STRESS UPDATE
# ============================================================

def stress_update(
    stress_old,
    dstrain,
    E,
    nu,
    c,
    phi,
    psi,
    tol=1e-8,
    repeated_eigenvalue_fallback=True,
    numerical_tangent_h=1e-7,
):
    """
    Complete pedagogical Mohr-Coulomb constitutive update.

    Input
    -----
    stress_old
        committed Cartesian stress

    dstrain
        Cartesian strain increment

    Output
    ------
    corrected stress

    algorithmic tangent

    principal information

    region

    yielded flag

    Tangent strategy
    ----------------
    elastic:
        D_elastic

    apex:
        zero tangent
        (pedagogical fixed-apex treatment only)

    plastic + distinct trial eigenvalues:
        analytic spectral Cartesian tangent

    plastic + repeated trial eigenvalues:
        numerical tangent fallback
        if enabled
    """

    result = (
        stress_update_core(
            stress_old,
            dstrain,
            E,
            nu,
            c,
            phi,
            psi,
            tol,
        )
    )

    D = elastic_matrix_3d(
        E,
        nu,
    )

    region = result[
        "region"
    ]

    # ========================================================
    # ELASTIC
    # ========================================================

    if (
        region
        == "elastic"
    ):

        Dalg_cartesian = (
            D.copy()
        )

        tangent_mode = (
            "elastic"
        )

    # ========================================================
    # APEX
    # ========================================================

    elif (
        region
        == "apex_projection"
    ):

        # ----------------------------------------------------
        # Pedagogical fixed-apex tangent.
        #
        # NOT final non-associated apex formulation.
        # ----------------------------------------------------

        Dalg_cartesian = (
            np.zeros(
                (6, 6)
            )
        )

        tangent_mode = (
            "apex_zero_pedagogical"
        )

    # ========================================================
    # REPEATED TRIAL PRINCIPAL STRESSES
    # ========================================================

    elif (
        has_repeated_principal_stress(
            result[
                "principal_trial"
            ]
        )
    ):

        if (
            not
            repeated_eigenvalue_fallback
        ):

            raise ValueError(
                "Repeated trial principal "
                "stresses require special "
                "spectral treatment or "
                "a fallback tangent."
            )

        Dalg_cartesian = (
            numerical_cartesian_tangent(
                stress_old,
                dstrain,
                E,
                nu,
                c,
                phi,
                psi,
                h=numerical_tangent_h,
            )
        )

        tangent_mode = (
            "numerical_repeated_"
            "eigenvalue_fallback"
        )

    # ========================================================
    # ANALYTIC SPECTRAL TANGENT
    # ========================================================

    else:

        Dalg_cartesian = (
            analytic_cartesian_tangent(
                D_elastic=D,

                principal_trial=
                    result[
                        "principal_trial"
                    ],

                principal_new=
                    result[
                        "principal_new"
                    ],

                directions=
                    result[
                        "principal_directions"
                    ],

                H=
                    result[
                        "principal_jacobian"
                    ],
            )
        )

        tangent_mode = (
            "analytic_spectral"
        )

    result[
        "tangent"
    ] = (
        Dalg_cartesian
    )

    result[
        "tangent_mode"
    ] = (
        tangent_mode
    )

    return result


# ============================================================
# 13. PRINCIPAL-SPACE UPDATE
#
# Used for tangent verification.
# ============================================================

def principal_stress_update(
    stress_old,
    dstrain,
    E,
    nu,
    c,
    phi,
    psi,
    tol=1e-8,
):
    """
    Constitutive update entirely in principal space.

    Used for verification of the principal
    active-set tangent.
    """

    Dp = principal_elastic_matrix(
        E,
        nu,
    )

    stress_trial = (
        stress_old
        + Dp
        @ dstrain
    )

    return (
        principal_return_mohr_coulomb(
            stress_trial,
            Dp,
            c,
            phi,
            psi,
            tol,
        )
    )


# ============================================================
# 14. NUMERICAL PRINCIPAL TANGENT
# ============================================================

def numerical_principal_tangent(
    stress_old,
    dstrain,
    E,
    nu,
    c,
    phi,
    psi,
    h=1e-7,
):
    """
    Numerical 3x3 principal tangent
    using central finite difference.
    """

    Dnum = np.zeros(
        (3, 3)
    )

    for j in range(3):

        dstrain_plus = (
            dstrain.copy()
        )

        dstrain_minus = (
            dstrain.copy()
        )

        dstrain_plus[j] += h

        dstrain_minus[j] -= h

        result_plus = (
            principal_stress_update(
                stress_old,
                dstrain_plus,
                E,
                nu,
                c,
                phi,
                psi,
            )
        )

        result_minus = (
            principal_stress_update(
                stress_old,
                dstrain_minus,
                E,
                nu,
                c,
                phi,
                psi,
            )
        )

        Dnum[
            :,
            j,
        ] = (
            result_plus[
                "stress"
            ]
            - result_minus[
                "stress"
            ]
        ) / (
            2.0
            * h
        )

    return Dnum


# ============================================================
# 15. SAME-REGION NUMERICAL CARTESIAN TANGENT
# ============================================================

def same_region_numerical_cartesian_tangent(
    stress_old,
    dstrain,
    E,
    nu,
    c,
    phi,
    psi,
    h=1e-7,
):
    """
    Numerical tangent intended for non-smooth MC regions.

    Preference:

        central difference

    if +h and -h remain in the same constitutive
    region as the base state.

    Otherwise:

        same-region one-sided difference.

    If neither perturbation remains in the same region,
    the corresponding tangent column is NaN.
    """

    base = stress_update_core(
        stress_old,
        dstrain,
        E,
        nu,
        c,
        phi,
        psi,
    )

    base_region = (
        base[
            "region"
        ]
    )

    base_stress = (
        base[
            "stress"
        ]
    )

    Dnum = np.full(
        (6, 6),
        np.nan,
    )

    for j in range(6):

        deps_plus = (
            dstrain.copy()
        )

        deps_minus = (
            dstrain.copy()
        )

        deps_plus[j] += h

        deps_minus[j] -= h

        plus = (
            stress_update_core(
                stress_old,
                deps_plus,
                E,
                nu,
                c,
                phi,
                psi,
            )
        )

        minus = (
            stress_update_core(
                stress_old,
                deps_minus,
                E,
                nu,
                c,
                phi,
                psi,
            )
        )

        same_plus = (
            plus[
                "region"
            ]
            == base_region
        )

        same_minus = (
            minus[
                "region"
            ]
            == base_region
        )

        if (
            same_plus
            and
            same_minus
        ):

            Dnum[
                :,
                j,
            ] = (
                plus[
                    "stress"
                ]
                - minus[
                    "stress"
                ]
            ) / (
                2.0
                * h
            )

        elif same_plus:

            Dnum[
                :,
                j,
            ] = (
                plus[
                    "stress"
                ]
                - base_stress
            ) / h

        elif same_minus:

            Dnum[
                :,
                j,
            ] = (
                base_stress
                - minus[
                    "stress"
                ]
            ) / h

    return Dnum


# ============================================================
# 16. VERIFY PRINCIPAL RETURN
# ============================================================

def verify_principal_return(
    result,
    c,
    phi,
    tol=1e-6,
):
    """
    Verify corrected stress satisfies
    the expected MC constraints.
    """

    stress = result[
        "stress"
    ]

    region = result[
        "region"
    ]

    (
        a13,
        a12,
        a23,
        _,
        _,
        _,
    ) = mc_plane_vectors(
        phi,
        psi=0.0,
    )

    C = cohesion_term(
        c,
        phi,
    )

    f13 = yield_value(
        stress,
        a13,
        C,
    )

    f12 = yield_value(
        stress,
        a12,
        C,
    )

    f23 = yield_value(
        stress,
        a23,
        C,
    )

    if (
        region
        == "elastic"
    ):

        assert (
            f13
            <= tol
        )

    elif (
        region
        == "face"
    ):

        assert (
            abs(
                f13
            )
            <= tol
        )

        assert (
            is_ordered(
                stress,
                tol,
            )
        )

    elif (
        region
        == "compression_edge"
    ):

        assert (
            abs(
                f13
            )
            <= tol
        )

        assert (
            abs(
                f12
            )
            <= tol
        )

        assert (
            abs(
                stress[1]
                - stress[2]
            )
            <= tol
        )

    elif (
        region
        == "extension_edge"
    ):

        assert (
            abs(
                f13
            )
            <= tol
        )

        assert (
            abs(
                f23
            )
            <= tol
        )

        assert (
            abs(
                stress[0]
                - stress[1]
            )
            <= tol
        )

    elif (
        region
        == "apex_projection"
    ):

        assert (
            abs(
                f13
            )
            <= tol
        )

        assert (
            abs(
                f12
            )
            <= tol
        )

        assert (
            abs(
                f23
            )
            <= tol
        )

        assert (
            np.allclose(
                stress,
                stress[0],
                atol=tol,
            )
        )

    else:

        raise ValueError(
            f"Unknown region: "
            f"{region}"
        )


# ============================================================
# 17. STRESS INVARIANTS
# ============================================================

def stress_invariants_from_principal(
    principal_stress,
):
    """
    Calculate:

        p = mean stress
        q = sqrt(3 J2)

    Compression positive.
    """

    (
        s1,
        s2,
        s3,
    ) = principal_stress

    p = (
        s1
        + s2
        + s3
    ) / 3.0

    q = np.sqrt(
        0.5
        * (
            (s1 - s2) ** 2
            + (s2 - s3) ** 2
            + (s3 - s1) ** 2
        )
    )

    return (
        p,
        q,
    )


# ============================================================
# 18. TEST:
# PRINCIPAL RETURN REGIONS
# ============================================================

def test_principal_regions():

    E = 30_000.0
    nu = 0.30

    c = 10.0
    phi = 30.0
    psi = 0.0

    Dp = principal_elastic_matrix(
        E,
        nu,
    )

    tests = [

        (
            "elastic",

            np.array([
                200.0,
                120.0,
                100.0,
            ]),
        ),

        (
            "face",

            np.array([
                400.0,
                120.0,
                100.0,
            ]),
        ),

        (
            "compression_edge",

            np.array([
                600.0,
                120.0,
                100.0,
            ]),
        ),

        (
            "extension_edge",

            np.array([
                400.0,
                390.0,
                100.0,
            ]),
        ),

        (
            "apex_projection",

            np.array([
                -30.0,
                -30.0,
                -30.0,
            ]),
        ),
    ]

    print()

    print(
        "=" * 70
    )

    print(
        "MOHR-COULOMB PRINCIPAL "
        "RETURN TEST"
    )

    print(
        "=" * 70
    )

    for (
        expected_region,
        stress_trial,
    ) in tests:

        result = (
            principal_return_mohr_coulomb(
                stress_trial,
                Dp,
                c,
                phi,
                psi,
            )
        )

        verify_principal_return(
            result,
            c,
            phi,
        )

        print()

        print(
            "-" * 70
        )

        print(
            "Trial stress :",
            stress_trial,
        )

        print(
            "Expected     :",
            expected_region,
        )

        print(
            "Calculated   :",
            result[
                "region"
            ],
        )

        print(
            "New stress   :",
            result[
                "stress"
            ],
        )

        assert (
            result[
                "region"
            ]
            == expected_region
        )

    print()

    print(
        "ALL PRINCIPAL RETURN "
        "TESTS PASSED"
    )


# ============================================================
# 19. TEST:
# PRINCIPAL ANALYTIC vs NUMERICAL TANGENT
# ============================================================

def test_principal_tangent():

    E = 30_000.0
    nu = 0.30

    c = 10.0
    phi = 30.0
    psi = 0.0

    stress_old = np.array([
        100.0,
        100.0,
        100.0,
    ])

    target_trial = np.array([
        400.0,
        120.0,
        100.0,
    ])

    Dp = principal_elastic_matrix(
        E,
        nu,
    )

    # --------------------------------------------------------
    # Determine strain increment that produces
    # the selected target trial stress.
    # --------------------------------------------------------

    dstrain = np.linalg.solve(
        Dp,
        target_trial
        - stress_old,
    )

    result = (
        principal_stress_update(
            stress_old,
            dstrain,
            E,
            nu,
            c,
            phi,
            psi,
        )
    )

    assert (
        result[
            "region"
        ]
        == "face"
    )

    Danalytic = (
        result[
            "tangent_principal"
        ]
    )

    Dnumerical = (
        numerical_principal_tangent(
            stress_old,
            dstrain,
            E,
            nu,
            c,
            phi,
            psi,
            h=1e-7,
        )
    )

    difference = (
        Danalytic
        - Dnumerical
    )

    print()

    print(
        "=" * 70
    )

    print(
        "PRINCIPAL TANGENT "
        "VERIFICATION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Region:"
    )

    print(
        result[
            "region"
        ]
    )

    print()

    print(
        "Analytic tangent:"
    )

    print(
        Danalytic
    )

    print()

    print(
        "Numerical tangent:"
    )

    print(
        Dnumerical
    )

    print()

    print(
        "Difference:"
    )

    print(
        difference
    )

    assert np.allclose(
        Danalytic,
        Dnumerical,
        rtol=1e-7,
        atol=1e-4,
    )

    print()

    print(
        "ANALYTIC AND NUMERICAL "
        "PRINCIPAL TANGENTS MATCH"
    )


# ============================================================
# 20. TEST:
# CARTESIAN FACE TANGENT
# ============================================================

def test_cartesian_face_tangent():

    E = 30_000.0
    nu = 0.30

    c = 10.0
    phi = 30.0
    psi = 0.0

    stress_old = np.array([
        100.0,
        100.0,
        100.0,
        0.0,
        0.0,
        0.0,
    ])

    # --------------------------------------------------------
    # Includes shear so the principal axes rotate.
    # --------------------------------------------------------

    dstrain = np.array([
         0.0100,
        -0.0020,
        -0.0030,
         0.0030,
         0.0000,
         0.0000,
    ])

    result = stress_update(
        stress_old,
        dstrain,
        E,
        nu,
        c,
        phi,
        psi,
    )

    assert (
        result[
            "region"
        ]
        == "face"
    )

    Danalytic = (
        result[
            "tangent"
        ]
    )

    Dnumerical = (
        numerical_cartesian_tangent(
            stress_old,
            dstrain,
            E,
            nu,
            c,
            phi,
            psi,
            h=1e-7,
        )
    )

    difference = (
        Danalytic
        - Dnumerical
    )

    print()

    print(
        "=" * 70
    )

    print(
        "CARTESIAN FACE "
        "TANGENT VERIFICATION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Region:"
    )

    print(
        result[
            "region"
        ]
    )

    print()

    print(
        "Tangent mode:"
    )

    print(
        result[
            "tangent_mode"
        ]
    )

    print()

    print(
        "Analytic Cartesian tangent:"
    )

    print(
        Danalytic
    )

    print()

    print(
        "Numerical Cartesian tangent:"
    )

    print(
        Dnumerical
    )

    print()

    print(
        "Difference:"
    )

    print(
        difference
    )

    print()

    print(
        "Maximum absolute error:"
    )

    print(
        np.max(
            np.abs(
                difference
            )
        )
    )

    assert np.allclose(
        Danalytic,
        Dnumerical,
        rtol=1e-6,
        atol=1e-3,
    )

    print()

    print(
        "ANALYTIC AND NUMERICAL "
        "CARTESIAN FACE "
        "TANGENTS MATCH"
    )


# ============================================================
# 21. TEST:
# COMPRESSION EDGE CARTESIAN TANGENT
# ============================================================

def test_compression_edge_cartesian_tangent():

    E = 30_000.0
    nu = 0.30

    c = 10.0
    phi = 30.0
    psi = 0.0

    stress_old = np.array([
        100.0,
        100.0,
        100.0,
        0.0,
        0.0,
        0.0,
    ])

    D = elastic_matrix_3d(
        E,
        nu,
    )

    # --------------------------------------------------------
    # Distinct TRIAL principal stresses.
    #
    # Corrected result will lie on:
    #
    # sigma2 = sigma3
    # --------------------------------------------------------

    principal_trial_target = (
        np.array([
            600.0,
            120.0,
            100.0,
        ])
    )

    # --------------------------------------------------------
    # Rotate trial principal axes by 20 degrees
    # about the z-axis.
    # --------------------------------------------------------

    theta = np.radians(
        20.0
    )

    Q = np.array([
        [
            np.cos(theta),
            -np.sin(theta),
            0.0,
        ],

        [
            np.sin(theta),
            np.cos(theta),
            0.0,
        ],

        [
            0.0,
            0.0,
            1.0,
        ],
    ])

    trial_tensor = (
        Q
        @ np.diag(
            principal_trial_target
        )
        @ Q.T
    )

    target_trial = (
        stress_tensor_to_vector(
            trial_tensor
        )
    )

    # --------------------------------------------------------
    # Determine dstrain such that:
    #
    # sigma_trial =
    # sigma_old + D dstrain
    # --------------------------------------------------------

    dstrain = np.linalg.solve(
        D,
        target_trial
        - stress_old,
    )

    result = stress_update(
        stress_old,
        dstrain,
        E,
        nu,
        c,
        phi,
        psi,
    )

    assert (
        result[
            "region"
        ]
        == "compression_edge"
    )

    Danalytic = (
        result[
            "tangent"
        ]
    )

    Dnumerical = (
        same_region_numerical_cartesian_tangent(
            stress_old,
            dstrain,
            E,
            nu,
            c,
            phi,
            psi,
            h=1e-7,
        )
    )

    valid = (
        ~np.isnan(
            Dnumerical
        )
    )

    difference = np.where(
        valid,
        Danalytic
        - Dnumerical,
        np.nan,
    )

    print()

    print(
        "=" * 70
    )

    print(
        "COMPRESSION EDGE "
        "TANGENT TEST"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Trial principal stress:"
    )

    print(
        result[
            "principal_trial"
        ]
    )

    print()

    print(
        "Corrected principal stress:"
    )

    print(
        result[
            "principal_new"
        ]
    )

    print()

    print(
        "Tangent mode:"
    )

    print(
        result[
            "tangent_mode"
        ]
    )

    print()

    print(
        "Analytic tangent:"
    )

    print(
        Danalytic
    )

    print()

    print(
        "Same-region numerical tangent:"
    )

    print(
        Dnumerical
    )

    print()

    print(
        "Difference:"
    )

    print(
        difference
    )

    if np.all(
        valid
    ):

        assert np.allclose(
            Danalytic,
            Dnumerical,
            rtol=1e-5,
            atol=1e-2,
        )

        print()

        print(
            "COMPRESSION EDGE "
            "TANGENT MATCHES"
        )

    else:

        print()

        print(
            "Some perturbation "
            "directions crossed an "
            "active-set boundary."
        )


# ============================================================
# 22. TEST:
# EXTENSION EDGE CARTESIAN TANGENT
# ============================================================

def test_extension_edge_cartesian_tangent():

    E = 30_000.0
    nu = 0.30

    c = 10.0
    phi = 30.0
    psi = 0.0

    stress_old = np.array([
        100.0,
        100.0,
        100.0,
        0.0,
        0.0,
        0.0,
    ])

    D = elastic_matrix_3d(
        E,
        nu,
    )

    # --------------------------------------------------------
    # Distinct trial principal stresses.
    #
    # Corrected result will lie on:
    #
    # sigma1 = sigma2
    # --------------------------------------------------------

    principal_trial_target = (
        np.array([
            400.0,
            390.0,
            100.0,
        ])
    )

    theta = np.radians(
        20.0
    )

    Q = np.array([
        [
            np.cos(theta),
            -np.sin(theta),
            0.0,
        ],

        [
            np.sin(theta),
            np.cos(theta),
            0.0,
        ],

        [
            0.0,
            0.0,
            1.0,
        ],
    ])

    trial_tensor = (
        Q
        @ np.diag(
            principal_trial_target
        )
        @ Q.T
    )

    target_trial = (
        stress_tensor_to_vector(
            trial_tensor
        )
    )

    dstrain = np.linalg.solve(
        D,
        target_trial
        - stress_old,
    )

    result = stress_update(
        stress_old,
        dstrain,
        E,
        nu,
        c,
        phi,
        psi,
    )

    assert (
        result[
            "region"
        ]
        == "extension_edge"
    )

    Danalytic = (
        result[
            "tangent"
        ]
    )

    Dnumerical = (
        same_region_numerical_cartesian_tangent(
            stress_old,
            dstrain,
            E,
            nu,
            c,
            phi,
            psi,
            h=1e-7,
        )
    )

    valid = (
        ~np.isnan(
            Dnumerical
        )
    )

    difference = np.where(
        valid,
        Danalytic
        - Dnumerical,
        np.nan,
    )

    print()

    print(
        "=" * 70
    )

    print(
        "EXTENSION EDGE "
        "TANGENT TEST"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Trial principal stress:"
    )

    print(
        result[
            "principal_trial"
        ]
    )

    print()

    print(
        "Corrected principal stress:"
    )

    print(
        result[
            "principal_new"
        ]
    )

    print()

    print(
        "Tangent mode:"
    )

    print(
        result[
            "tangent_mode"
        ]
    )

    print()

    print(
        "Analytic tangent:"
    )

    print(
        Danalytic
    )

    print()

    print(
        "Same-region numerical tangent:"
    )

    print(
        Dnumerical
    )

    print()

    print(
        "Difference:"
    )

    print(
        difference
    )

    if np.all(
        valid
    ):

        max_error = np.max(
            np.abs(
                difference
            )
        )

        print()

        print(
            "Maximum absolute error:"
        )

        print(
            max_error
        )

        assert np.allclose(
            Danalytic,
            Dnumerical,
            rtol=1e-5,
            atol=1e-2,
        )

        print()

        print(
            "EXTENSION EDGE "
            "TANGENT MATCHES"
        )

    else:

        print()

        print(
            "Some perturbation "
            "directions crossed an "
            "active-set boundary."
        )


# ============================================================
# 23. STRAIN-HISTORY TEST
# ============================================================

def run_strain_history():

    E = 30_000.0
    nu = 0.30

    c = 10.0
    phi = 30.0
    psi = 0.0

    # --------------------------------------------------------
    # Initial isotropic stress
    # --------------------------------------------------------

    stress = np.array([
        100.0,
        100.0,
        100.0,
        0.0,
        0.0,
        0.0,
    ])

    # --------------------------------------------------------
    # Axial strain increment
    # --------------------------------------------------------

    deps_axial = (
        0.0005
    )

    dstrain = np.array([
        deps_axial,
        -nu
        * deps_axial,
        -nu
        * deps_axial,
        0.0,
        0.0,
        0.0,
    ])

    history = []

    # --------------------------------------------------------
    # Initial condition
    # --------------------------------------------------------

    history.append({
        "step":
            0,

        "axial_strain":
            0.0,

        "sigma1":
            100.0,

        "sigma2":
            100.0,

        "sigma3":
            100.0,

        "p":
            100.0,

        "q":
            0.0,

        "region":
            "initial",
    })

    total_axial_strain = (
        0.0
    )

    # --------------------------------------------------------
    # Incremental loading
    # --------------------------------------------------------

    for step in range(
        1,
        31,
    ):

        result = stress_update(
            stress_old=
                stress,

            dstrain=
                dstrain,

            E=
                E,

            nu=
                nu,

            c=
                c,

            phi=
                phi,

            psi=
                psi,
        )

        # ----------------------------------------------------
        # Accept corrected stress
        # ----------------------------------------------------

        stress = (
            result[
                "stress"
            ].copy()
        )

        total_axial_strain += (
            deps_axial
        )

        principal = (
            result[
                "principal_new"
            ]
        )

        (
            p,
            q,
        ) = (
            stress_invariants_from_principal(
                principal
            )
        )

        history.append({
            "step":
                step,

            "axial_strain":
                total_axial_strain,

            "sigma1":
                principal[0],

            "sigma2":
                principal[1],

            "sigma3":
                principal[2],

            "p":
                p,

            "q":
                q,

            "region":
                result[
                    "region"
                ],
        })

    return history


# ============================================================
# 24. PRINT STRAIN HISTORY
# ============================================================

def print_history(
    history,
):

    print()

    print(
        "=" * 105
    )

    print(
        "STRAIN HISTORY"
    )

    print(
        "=" * 105
    )

    print(
        "Step   eps1       "
        "sigma1     sigma2     sigma3"
        "       p          q"
        "        region"
    )

    print(
        "-" * 105
    )

    for row in history:

        print(
            f"{row['step']:4d} "
            f"{row['axial_strain']:9.5f} "
            f"{row['sigma1']:10.3f} "
            f"{row['sigma2']:10.3f} "
            f"{row['sigma3']:10.3f} "
            f"{row['p']:10.3f} "
            f"{row['q']:10.3f} "
            f"{row['region']}"
        )


# ============================================================
# 25. OPTIONAL PLOTTING
# ============================================================

def plot_history(
    history,
):

    import matplotlib.pyplot as plt

    axial_strain = np.array([
        row[
            "axial_strain"
        ]
        for row
        in history
    ])

    sigma1 = np.array([
        row[
            "sigma1"
        ]
        for row
        in history
    ])

    sigma2 = np.array([
        row[
            "sigma2"
        ]
        for row
        in history
    ])

    sigma3 = np.array([
        row[
            "sigma3"
        ]
        for row
        in history
    ])

    p = np.array([
        row[
            "p"
        ]
        for row
        in history
    ])

    q = np.array([
        row[
            "q"
        ]
        for row
        in history
    ])

    # --------------------------------------------------------
    # Principal stress - axial strain
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        axial_strain,
        sigma1,
        label=r"$\sigma_1$",
    )

    plt.plot(
        axial_strain,
        sigma2,
        label=r"$\sigma_2$",
    )

    plt.plot(
        axial_strain,
        sigma3,
        label=r"$\sigma_3$",
    )

    plt.xlabel(
        "Axial strain"
    )

    plt.ylabel(
        "Principal stress [kPa]"
    )

    plt.legend()

    plt.grid()

    plt.show()

    # --------------------------------------------------------
    # p-q stress path
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        p,
        q,
        marker="o",
    )

    plt.xlabel(
        "p [kPa]"
    )

    plt.ylabel(
        "q [kPa]"
    )

    plt.grid()

    plt.show()


# ============================================================
# 26. MAIN
# ============================================================

if __name__ == "__main__":

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    # --------------------------------------------------------
    # TEST 1:
    # Elastic / face / compression edge /
    # extension edge / apex
    # --------------------------------------------------------

    test_principal_regions()

    # --------------------------------------------------------
    # TEST 2:
    # Principal analytic tangent
    # versus numerical principal tangent
    # --------------------------------------------------------

    test_principal_tangent()

    # --------------------------------------------------------
    # TEST 3:
    # Full Cartesian spectral tangent
    # for a face return
    # --------------------------------------------------------

    test_cartesian_face_tangent()

    # --------------------------------------------------------
    # TEST 4:
    # Compression-edge Cartesian tangent
    # --------------------------------------------------------

    test_compression_edge_cartesian_tangent()

    # --------------------------------------------------------
    # TEST 5:
    # Extension-edge Cartesian tangent
    # --------------------------------------------------------

    test_extension_edge_cartesian_tangent()

    # --------------------------------------------------------
    # TEST 6:
    # Incremental strain history
    # --------------------------------------------------------

    history = (
        run_strain_history()
    )

    print_history(
        history
    )

    # --------------------------------------------------------
    # OPTIONAL:
    # Uncomment for plots
    # --------------------------------------------------------

    # plot_history(
    #     history
    # )