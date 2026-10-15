import numpy as np

from material_model.constitutive import MaterialContext
from material_model.mohr_coulomb import MohrCoulomb
from state.gauss_point_state import GaussPointState


def _material():
    return MohrCoulomb(
        parameters={
            "E": 30000.0,
            "nu": 0.30,
            "c": 10.0,
            "phi": 30.0,
            "psi": 5.0,
            "density": 1800.0,
        },
        formulation="plane_strain",
    )


def _integrate(strain):
    strain = np.asarray(strain, dtype=float)
    return _material().integrate(
        total_strain_2d=strain,
        strain_increment_2d=strain,
        committed_state=GaussPointState(),
        context=MaterialContext(),
    )


def test_mc_transient_spectral_work_is_not_persisted():
    response = _integrate([0.005, -0.01, 0.0])
    internal = response.state.internal_variables

    transient_keys = {
        "plastic_multipliers",
        "principal_trial_compression_positive",
        "principal_new_compression_positive",
        "principal_directions",
        "principal_jacobian",
        "apex_correction_stress_principal",
        "apex_koiter_direction_principal",
        "apex_alpha",
    }

    assert transient_keys.isdisjoint(internal)

    for key in (
        "mc_region",
        "yield_function_trial",
        "yield_function",
        "plastic_multiplier",
        "q",
        "p_compression_positive",
    ):
        assert key in internal

    assert response.diagnostics["region"] == "face"
    assert response.diagnostics["tangent_mode"] == "analytic_spectral"


def test_mc_apex_transient_work_is_not_persisted():
    response = _integrate([-0.02, 0.03, -0.03])

    assert response.diagnostics["region"] == "apex_fortran"
    assert response.diagnostics["tangent_mode"] == (
        "fortran_modified_apex_single_singular"
    )

    internal = response.state.internal_variables
    assert "principal_directions" not in internal
    assert "principal_jacobian" not in internal
    assert "apex_correction_stress_principal" not in internal
    assert "apex_koiter_direction_principal" not in internal


def test_mc_cleanup_preserves_reference_face_response():
    response = _integrate([0.005, -0.01, 0.0])

    expected_stress = np.array([
        -72.52297133298028,
        -252.20993015031837,
        -97.4198704449896,
        0.0,
        0.0,
        0.0,
    ])
    expected_plastic_strain = np.array([
        0.003921134371812928,
        -0.0032924307461024184,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
    expected_tangent = np.array([
        [11872.569657600228, 14139.687226912845, 0.0],
        [35617.70897280067, 42419.06168073854, 0.0],
        [0.0, 0.0, 5989.5652939112715],
    ])

    assert np.allclose(response.state.effective_stress, expected_stress, rtol=0.0, atol=1e-12)
    assert np.allclose(response.state.plastic_strain, expected_plastic_strain, rtol=0.0, atol=1e-12)
    assert np.allclose(response.tangent, expected_tangent, rtol=0.0, atol=1e-10)
    assert response.diagnostics["yield_function_trial"] == 184.60256884738808
