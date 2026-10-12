import numpy as np


# ============================================================
# PEDAGOGICAL MOHR-COULOMB CONSTITUTIVE MODEL
#
# Sign convention:
#   compression stress  = positive
#   compression strain  = positive
#
# Cartesian Voigt ordering:
#   [xx, yy, zz, xy, yz, xz]
#
# Principal ordering:
#   sigma1 >= sigma2 >= sigma3
#
# This is a pedagogical implementation.
# Not yet the production GeoPyFEM material model.
# ============================================================


# ============================================================
# 1. ELASTICITY
# ============================================================

def elastic_matrix_3d(E, nu):
    """
    3D isotropic elastic constitutive matrix.
    """

    G = E / (2.0 * (1.0 + nu))

    lam = (
        E * nu
        / ((1.0 + nu) * (1.0 - 2.0 * nu))
    )

    D = np.array([
        [lam + 2.0 * G, lam,           lam,           0.0, 0.0, 0.0],
        [lam,           lam + 2.0 * G, lam,           0.0, 0.0, 0.0],
        [lam,           lam,           lam + 2.0 * G, 0.0, 0.0, 0.0],
        [0.0,           0.0,           0.0,           G,   0.0, 0.0],
        [0.0,           0.0,           0.0,           0.0, G,   0.0],
        [0.0,           0.0,           0.0,           0.0, 0.0, G],
    ])

    return D


def principal_elastic_matrix(E, nu):
    """
    Elastic matrix for normal components in principal space.

    [sigma1]   [       ] [epsilon1]
    [sigma2] = [  Dp   ] [epsilon2]
    [sigma3]   [       ] [epsilon3]
    """

    D = elastic_matrix_3d(E, nu)

    return D[:3, :3]


# ============================================================
# 2. STRESS REPRESENTATION
# ============================================================

def stress_vector_to_tensor(stress):
    """
    Voigt stress vector -> 3x3 stress tensor.

    stress =
    [sxx, syy, szz, txy, tyz, txz]
    """

    sxx, syy, szz, txy, tyz, txz = stress

    return np.array([
        [sxx, txy, txz],
        [txy, syy, tyz],
        [txz, tyz, szz],
    ])


def stress_tensor_to_vector(tensor):
    """
    3x3 stress tensor -> Voigt stress vector.
    """

    return np.array([
        tensor[0, 0],
        tensor[1, 1],
        tensor[2, 2],
        tensor[0, 1],
        tensor[1, 2],
        tensor[0, 2],
    ])


# ============================================================
# 3. PRINCIPAL STRESS DECOMPOSITION
# ============================================================

def principal_decomposition(stress):
    """
    Return ordered principal stresses and directions.

    Compression-positive ordering:

        sigma1 >= sigma2 >= sigma3
    """

    tensor = stress_vector_to_tensor(stress)

    values, vectors = np.linalg.eigh(tensor)

    # NumPy gives ascending eigenvalues.
    order = np.argsort(values)[::-1]

    values = values[order]
    vectors = vectors[:, order]

    return values, vectors


def principal_to_cartesian(
    principal_stress,
    directions,
):
    """
    Reconstruct Cartesian stress from principal stresses.

        sigma = Q diag(sigma_p) Q.T
    """

    tensor = (
        directions
        @ np.diag(principal_stress)
        @ directions.T
    )

    return stress_tensor_to_vector(tensor)


# ============================================================
# 4. MOHR-COULOMB SURFACES
# ============================================================

def mc_plane_vectors(phi, psi):
    """
    Construct yield normals and plastic-flow directions.

    Angles in degrees.
    """

    phi_rad = np.radians(phi)
    psi_rad = np.radians(psi)

    # --------------------------------------------------------
    # Yield surface
    # --------------------------------------------------------

    A = 1.0 - np.sin(phi_rad)
    B = 1.0 + np.sin(phi_rad)

    # --------------------------------------------------------
    # Plastic potential
    # --------------------------------------------------------

    Ap = 1.0 - np.sin(psi_rad)
    Bp = 1.0 + np.sin(psi_rad)

    # --------------------------------------------------------
    # Yield normals
    # --------------------------------------------------------

    a13 = np.array([
        A,
        0.0,
        -B,
    ])

    a12 = np.array([
        A,
        -B,
        0.0,
    ])

    a23 = np.array([
        0.0,
        A,
        -B,
    ])

    # --------------------------------------------------------
    # Plastic flow directions
    # --------------------------------------------------------

    b13 = np.array([
        Ap,
        0.0,
        -Bp,
    ])

    b12 = np.array([
        Ap,
        -Bp,
        0.0,
    ])

    b23 = np.array([
        0.0,
        Ap,
        -Bp,
    ])

    return (
        a13,
        a12,
        a23,
        b13,
        b12,
        b23,
    )


def cohesion_term(c, phi):
    """
    C = 2 c cos(phi)
    """

    phi_rad = np.radians(phi)

    return (
        2.0
        * c
        * np.cos(phi_rad)
    )


def yield_value(
    stress_p,
    a,
    C,
):
    """
    Linear Mohr-Coulomb surface:

        f = a.T sigma - C
    """

    return float(
        a @ stress_p - C
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
    Return mapping for a specified active set.

    Parameters
    ----------
    stress_trial : (3,) array

    Dp : (3,3) array
        Elastic matrix in principal space.

    A : (3,n) array
        Active yield normals.

    B : (3,n) array
        Active plastic-flow directions.

    Returns
    -------
    stress_new
    dstrain_p
    dlambda
    Dalg
    """

    n_active = A.shape[1]

    # ========================================================
    # Trial values of active yield surfaces
    #
    # f_trial = A.T sigma_trial - C
    # ========================================================

    f_trial = (
        A.T @ stress_trial
        - C * np.ones(n_active)
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

    dlambda = np.linalg.solve(
        M,
        f_trial,
    )

    # ========================================================
    # Plastic strain increment
    #
    # deps_p = B dlambda
    # ========================================================

    dstrain_p = (
        B @ dlambda
    )

    # ========================================================
    # Corrected stress
    #
    # sigma_new =
    # sigma_trial - D deps_p
    # ========================================================

    stress_new = (
        stress_trial
        - Dp @ dstrain_p
    )

    # ========================================================
    # ALGORITHMIC TANGENT
    #
    # Dalg =
    #
    # D - D B (A.T D B)^(-1) A.T D
    #
    # Do not explicitly invert M.
    # ========================================================

    X = np.linalg.solve(
        M,
        A.T @ Dp,
    )

    Dalg = (
        Dp
        - Dp @ B @ X
    )

    return (
        stress_new,
        dstrain_p,
        dlambda,
        Dalg,
    )


# ============================================================
# 6. ADMISSIBILITY CHECKS
# ============================================================

def is_ordered(
    stress,
    tol=1e-10,
):
    """
    Check:

        sigma1 >= sigma2 >= sigma3
    """

    s1, s2, s3 = stress

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

    s1, s2, s3 = stress

    stress_ok = (
        s1 >= s2 - tol
        and
        abs(s2 - s3) <= tol
    )

    multiplier_ok = np.all(
        dlambda >= -tol
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

    s1, s2, s3 = stress

    stress_ok = (
        abs(s1 - s2) <= tol
        and
        s2 >= s3 - tol
    )

    multiplier_ok = np.all(
        dlambda >= -tol
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
    Mohr-Coulomb return mapping in ordered principal space.

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

    f13_trial = yield_value(
        stress_trial,
        a13,
        C,
    )

    if f13_trial <= tol:

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
        }

    # ========================================================
    # 7.2 FACE CANDIDATE
    # ========================================================

    A_face = (
        a13.reshape(3, 1)
    )

    B_face = (
        b13.reshape(3, 1)
    )

    (
        stress_face,
        deps_face,
        dlambda_face,
        Dalg_face,
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
        }

    # ========================================================
    # 7.3 COMPRESSION EDGE
    #
    # Active:
    #
    # f13 = 0
    # f12 = 0
    # ========================================================

    A_ce = np.column_stack([
        a13,
        a12,
    ])

    B_ce = np.column_stack([
        b13,
        b12,
    ])

    (
        stress_ce,
        deps_ce,
        dlambda_ce,
        Dalg_ce,
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
        }

    # ========================================================
    # 7.4 EXTENSION EDGE
    #
    # Active:
    #
    # f13 = 0
    # f23 = 0
    # ========================================================

    A_ee = np.column_stack([
        a13,
        a23,
    ])

    B_ee = np.column_stack([
        b13,
        b23,
    ])

    (
        stress_ee,
        deps_ee,
        dlambda_ee,
        Dalg_ee,
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
        }

    # ========================================================
    # 7.5 APEX
    #
    # Pedagogical fixed-apex projection.
    #
    # Production non-associated MC requires
    # special apex treatment.
    # ========================================================

    phi_rad = np.radians(phi)

    if abs(np.sin(phi_rad)) < 1e-14:

        raise ValueError(
            "Mohr-Coulomb apex is undefined "
            "for phi = 0."
        )

    p_apex = (
        -c
        / np.tan(phi_rad)
    )

    stress_apex = np.array([
        p_apex,
        p_apex,
        p_apex,
    ])

    # --------------------------------------------------------
    # Pedagogical tangent for fixed-apex projection
    # --------------------------------------------------------

    Dalg_apex = np.zeros(
        (3, 3)
    )

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
            Dalg_apex,
    }


# ============================================================
# 8. FULL CARTESIAN STRESS UPDATE
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
):
    """
    Complete stress update.

    Important:
    tangent returned by principal return is still
    in PRINCIPAL SPACE.

    We are not yet constructing the production
    Cartesian 6x6 tangent.
    """

    # --------------------------------------------------------
    # Elastic matrices
    # --------------------------------------------------------

    D = elastic_matrix_3d(
        E,
        nu,
    )

    Dp = principal_elastic_matrix(
        E,
        nu,
    )

    # --------------------------------------------------------
    # Elastic predictor
    # --------------------------------------------------------

    stress_trial = (
        stress_old
        + D @ dstrain
    )

    # --------------------------------------------------------
    # Principal decomposition
    # --------------------------------------------------------

    (
        principal_trial,
        directions,
    ) = principal_decomposition(
        stress_trial
    )

    # --------------------------------------------------------
    # Mohr-Coulomb return
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Back to Cartesian coordinates
    # --------------------------------------------------------

    stress_new = principal_to_cartesian(
        principal_new,
        directions,
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

        "tangent_principal":
            principal_result[
                "tangent_principal"
            ],
    }


# ============================================================
# 9. VERIFY RETURN CONDITIONS
# ============================================================

def verify_principal_return(
    result,
    c,
    phi,
    tol=1e-6,
):
    """
    Check that corrected stress satisfies the correct
    Mohr-Coulomb constraints.
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

    if region == "elastic":

        assert f13 <= tol

    elif region == "face":

        assert abs(f13) <= tol

        assert is_ordered(
            stress,
            tol,
        )

    elif region == "compression_edge":

        assert abs(f13) <= tol
        assert abs(f12) <= tol

        assert abs(
            stress[1]
            - stress[2]
        ) <= tol

    elif region == "extension_edge":

        assert abs(f13) <= tol
        assert abs(f23) <= tol

        assert abs(
            stress[0]
            - stress[1]
        ) <= tol

    elif region == "apex_projection":

        assert abs(f13) <= tol
        assert abs(f12) <= tol
        assert abs(f23) <= tol

        assert np.allclose(
            stress,
            stress[0],
            atol=tol,
        )

    else:

        raise ValueError(
            f"Unknown region: {region}"
        )


# ============================================================
# 10. PRINCIPAL-SPACE MATERIAL UPDATE
#
# Used specifically for tangent verification.
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

    This avoids principal-direction rotation and is useful
    for verifying the active-set tangent.
    """

    Dp = principal_elastic_matrix(
        E,
        nu,
    )

    stress_trial = (
        stress_old
        + Dp @ dstrain
    )

    return principal_return_mohr_coulomb(
        stress_trial,
        Dp,
        c,
        phi,
        psi,
        tol,
    )


# ============================================================
# 11. NUMERICAL PRINCIPAL TANGENT
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
    Numerical tangent using central finite differences.

    D_ij =
        d sigma_i / d epsilon_j
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

        stress_plus = (
            result_plus[
                "stress"
            ]
        )

        stress_minus = (
            result_minus[
                "stress"
            ]
        )

        Dnum[:, j] = (
            stress_plus
            - stress_minus
        ) / (2.0 * h)

    return Dnum


# ============================================================
# 12. STRESS INVARIANTS
# ============================================================

def stress_invariants_from_principal(
    principal_stress,
):
    """
    Calculate p and q.

    Compression positive.
    """

    s1, s2, s3 = (
        principal_stress
    )

    p = (
        s1 + s2 + s3
    ) / 3.0

    q = np.sqrt(
        0.5
        * (
            (s1 - s2) ** 2
            + (s2 - s3) ** 2
            + (s3 - s1) ** 2
        )
    )

    return p, q


# ============================================================
# 13. PRINCIPAL REGION TEST
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
    print("=" * 70)
    print("MOHR-COULOMB PRINCIPAL RETURN TEST")
    print("=" * 70)

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
        print("-" * 70)

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
            result["region"],
        )

        print(
            "New stress   :",
            result["stress"],
        )

        assert (
            result["region"]
            == expected_region
        )

    print()
    print(
        "ALL PRINCIPAL RETURN TESTS PASSED"
    )


# ============================================================
# 14. ANALYTIC vs NUMERICAL TANGENT TEST
# ============================================================

def test_principal_tangent():

    E = 30_000.0
    nu = 0.30

    c = 10.0
    phi = 30.0
    psi = 0.0

    # --------------------------------------------------------
    # Old stress
    # --------------------------------------------------------

    stress_old = np.array([
        100.0,
        100.0,
        100.0,
    ])

    # --------------------------------------------------------
    # We deliberately choose a trial stress that returns
    # to the INTERIOR OF A FACE.
    #
    # This avoids edge non-smoothness.
    # --------------------------------------------------------

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
    # Find strain increment that produces target_trial
    #
    # target_trial =
    # stress_old + Dp dstrain
    # --------------------------------------------------------

    dstrain = np.linalg.solve(
        Dp,
        target_trial - stress_old,
    )

    # --------------------------------------------------------
    # Constitutive update
    # --------------------------------------------------------

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
        result["region"]
        == "face"
    )

    Danalytic = result[
        "tangent_principal"
    ]

    # --------------------------------------------------------
    # Numerical tangent
    # --------------------------------------------------------

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
    print("=" * 70)
    print(
        "PRINCIPAL TANGENT VERIFICATION"
    )
    print("=" * 70)

    print()
    print(
        "Strain increment:"
    )

    print(
        dstrain
    )

    print()
    print(
        "Region:"
    )

    print(
        result["region"]
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

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    assert np.allclose(
        Danalytic,
        Dnumerical,
        rtol=1e-7,
        atol=1e-4,
    )

    print()
    print(
        "ANALYTIC AND NUMERICAL "
        "TANGENTS MATCH"
    )


# ============================================================
# 15. STRAIN-HISTORY TEST
# ============================================================

def run_strain_history():

    E = 30_000.0
    nu = 0.30

    c = 10.0
    phi = 30.0
    psi = 0.0

    # --------------------------------------------------------
    # Initial hydrostatic stress
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

    deps_axial = 0.0005

    dstrain = np.array([
        deps_axial,
        -nu * deps_axial,
        -nu * deps_axial,
        0.0,
        0.0,
        0.0,
    ])

    history = []

    # Add initial condition
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

    total_axial_strain = 0.0

    # --------------------------------------------------------
    # Incremental loading
    # --------------------------------------------------------

    for step in range(
        1,
        31,
    ):

        result = stress_update(
            stress_old=stress,
            dstrain=dstrain,
            E=E,
            nu=nu,
            c=c,
            phi=phi,
            psi=psi,
        )

        # Accept corrected stress
        stress = (
            result["stress"].copy()
        )

        total_axial_strain += (
            deps_axial
        )

        principal = result[
            "principal_new"
        ]

        p, q = (
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
                result["region"],
        })

    return history


# ============================================================
# 16. PRINT HISTORY
# ============================================================

def print_history(
    history,
):

    print()
    print("=" * 105)
    print("STRAIN HISTORY")
    print("=" * 105)

    print(
        "Step   eps1       "
        "sigma1     sigma2     sigma3"
        "       p          q"
        "        region"
    )

    print("-" * 105)

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
# 17. OPTIONAL PLOTTING
# ============================================================

def plot_history(
    history,
):

    import matplotlib.pyplot as plt

    axial_strain = np.array([
        row["axial_strain"]
        for row in history
    ])

    sigma1 = np.array([
        row["sigma1"]
        for row in history
    ])

    sigma2 = np.array([
        row["sigma2"]
        for row in history
    ])

    sigma3 = np.array([
        row["sigma3"]
        for row in history
    ])

    p = np.array([
        row["p"]
        for row in history
    ])

    q = np.array([
        row["q"]
        for row in history
    ])

    # --------------------------------------------------------
    # Principal stresses
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
    Numerical 6x6 algorithmic tangent for the complete
    Cartesian stress-update algorithm.

    This derivative includes:
        - principal stress changes
        - principal direction changes
        - return mapping

    It is intended as a verification/reference tangent,
    not as the final production tangent.
    """

    Dnum = np.zeros((6, 6))

    for j in range(6):

        dstrain_plus = dstrain.copy()
        dstrain_minus = dstrain.copy()

        dstrain_plus[j] += h
        dstrain_minus[j] -= h

        result_plus = stress_update(
            stress_old=stress_old,
            dstrain=dstrain_plus,
            E=E,
            nu=nu,
            c=c,
            phi=phi,
            psi=psi,
        )

        result_minus = stress_update(
            stress_old=stress_old,
            dstrain=dstrain_minus,
            E=E,
            nu=nu,
            c=c,
            phi=phi,
            psi=psi,
        )

        stress_plus = result_plus["stress"]
        stress_minus = result_minus["stress"]

        Dnum[:, j] = (
            stress_plus
            - stress_minus
        ) / (2.0 * h)

    return Dnum

def test_cartesian_numerical_tangent():

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

    # A strain increment containing both normal and shear parts.
    # We want a non-trivial principal direction.
    dstrain = np.array([
         0.0100,
        -0.0020,
        -0.0030,
         0.0030,
         0.0000,
         0.0000,
    ])

    result = stress_update(
        stress_old=stress_old,
        dstrain=dstrain,
        E=E,
        nu=nu,
        c=c,
        phi=phi,
        psi=psi,
    )

    Dnum = numerical_cartesian_tangent(
        stress_old=stress_old,
        dstrain=dstrain,
        E=E,
        nu=nu,
        c=c,
        phi=phi,
        psi=psi,
        h=1e-7,
    )

    print()
    print("=" * 70)
    print("CARTESIAN NUMERICAL TANGENT")
    print("=" * 70)

    print()
    print("Region:")
    print(result["region"])

    print()
    print("Trial Cartesian stress:")
    print(result["stress_trial"])

    print()
    print("Trial principal stress:")
    print(result["principal_trial"])

    print()
    print("Corrected principal stress:")
    print(result["principal_new"])

    print()
    print("Corrected Cartesian stress:")
    print(result["stress"])

    print()
    print("Numerical Cartesian tangent:")
    print(Dnum)

# ============================================================
# 18. MAIN
# ============================================================

if __name__ == "__main__":

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    # --------------------------------------------------------
    # Test 1:
    # elastic / face / edges / apex
    # --------------------------------------------------------

    test_principal_regions()

    # --------------------------------------------------------
    # Test 2:
    # analytic vs numerical tangent
    # --------------------------------------------------------

    test_principal_tangent()

    # --------------------------------------------------------
    # Test 3:
    # incremental strain history
    # --------------------------------------------------------

    history = (
        run_strain_history()
    )

    print_history(
        history
    )

    # Optional
    plot_history(
        history
    )
    test_cartesian_numerical_tangent()