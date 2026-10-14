import numpy as np
import pytest

from material_model.mohr_coulomb import MohrCoulomb


def _material(apex_mode="fortran", apex_alpha=17.0):
    return MohrCoulomb(
        parameters={
            "E": 30000.0,
            "nu": 0.30,
            "cohesion": 10.0,
            "friction_angle": 30.0,
            "dilation_angle": 0.0,
            "apex_mode": apex_mode,
            "apex_alpha": apex_alpha,
        },
        formulation="plane_strain",
    )


def test_fortran_apex_projection_and_modified_tangent():
    material = _material()

    # Compression-positive principal trial stress intentionally beyond the
    # tensile apex. This was also used in the earlier GeoPyFEM MC benchmark.
    stress_trial = np.array([-30.0, -30.0, -30.0])

    result = material._principal_return(stress_trial)

    expected_apex = -material.cohesion / np.tan(
        np.radians(material.friction_angle)
    )

    assert result["region"] == "apex_fortran"
    assert result["yielded"] is True
    assert np.allclose(
        result["stress"],
        np.full(3, expected_apex),
        rtol=0.0,
        atol=1.0e-12,
    )

    # Fortran-style Koiter direction:
    #   SiPla   = SigP_trial - SigP_updated
    #   KoitDir = inv(Cp) @ SiPla
    expected_correction = stress_trial - result["stress"]
    expected_koiter = np.linalg.solve(
        material._Cp,
        expected_correction,
    )

    assert np.allclose(
        result["apex_correction_stress_principal"],
        expected_correction,
    )
    assert np.allclose(
        result["plastic_strain_increment_principal"],
        expected_koiter,
    )

    D_apex = result["tangent_principal"]

    # Modified Fortran apex tangent type 1 is single-singular in the
    # Koiter/plastic-correction direction.
    assert np.all(np.isfinite(D_apex))
    assert np.linalg.norm(D_apex) > 0.0
    assert np.allclose(
        D_apex @ expected_koiter,
        np.zeros(3),
        atol=1.0e-10,
    )

    eig = np.linalg.eigvalsh(D_apex)
    assert abs(eig[0]) < 1.0e-9
    assert eig[1] > 0.0
    assert eig[2] > 0.0

    # Updated apex stress must lie on the MC yield surface.
    a13 = material._mc_plane_vectors()[0]
    f_apex = material._yield_value(
        result["stress"],
        a13,
        material._cohesion_term(),
    )
    assert abs(f_apex) < 1.0e-10


def test_fortran_apex_cartesian_tangent_is_finite():
    material = _material()
    stress_trial = np.array([-30.0, -30.0, -30.0])
    result = material._principal_return(stress_trial)

    stress_trial_6 = np.array(
        [-30.0, -30.0, -30.0, 0.0, 0.0, 0.0]
    )
    _, directions = (
        material._principal_decomposition_compression_positive(
            stress_trial_6
        )
    )

    D6 = material._apex_cartesian_tangent_6d(
        directions=directions,
        H=result["principal_jacobian"],
    )

    assert D6.shape == (6, 6)
    assert np.all(np.isfinite(D6))

    indices = material._PLANE_STRAIN_INDICES
    D2 = D6[np.ix_(indices, indices)]

    assert D2.shape == (3, 3)
    assert np.all(np.isfinite(D2))
    assert np.linalg.norm(D2) > 0.0


def test_legacy_projection_mode_is_retained():
    material = _material(apex_mode="projection")
    result = material._principal_return(
        np.array([-30.0, -30.0, -30.0])
    )

    assert result["region"] == "apex_projection"
    assert result["plastic_strain_increment_principal"] is None
    assert np.allclose(
        result["tangent_principal"],
        np.zeros((3, 3)),
    )


def test_error_mode_still_stops_at_apex():
    material = _material(apex_mode="error")

    with pytest.raises(NotImplementedError):
        material._principal_return(
            np.array([-30.0, -30.0, -30.0])
        )


@pytest.mark.parametrize(
    "stress_trial, expected_region",
    [
        (np.array([400.0, 120.0, 100.0]), "face"),
        (np.array([600.0, 120.0, 100.0]), "compression_edge"),
        (np.array([400.0, 390.0, 100.0]), "extension_edge"),
        (np.array([100.0, 100.0, 100.0]), "elastic"),
    ],
)
def test_existing_return_regions_are_unchanged(
    stress_trial,
    expected_region,
):
    material = _material()
    result = material._principal_return(stress_trial)

    assert result["region"] == expected_region
