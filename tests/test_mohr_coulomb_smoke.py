"""Small repository-level smoke tests for material_model.mohr_coulomb.MohrCoulomb.

Run from the GeoPyFEM repository root after copying mohr_coulomb.py and
registering the material model.
"""

import numpy as np

from material_model.mohr_coulomb import MohrCoulomb
from state.gauss_point_state import GaussPointState


def _material(**overrides):
    parameters = {
        "E": 30000.0,
        "nu": 0.30,
        "cohesion": 10.0,
        "friction_angle": 30.0,
        "dilation_angle": 0.0,
        "apex_mode": "error",
    }
    parameters.update(overrides)
    return MohrCoulomb(parameters, formulation="plane_strain")


def _set_effective_stress(state, stress):
    if hasattr(state, "effective_stress"):
        state.effective_stress[:] = stress
    else:
        state.stress[:] = stress


def test_zero_increment_preserves_committed_stress():
    material = _material()
    state = GaussPointState()

    committed = np.array([-100.0, -100.0, -100.0, 0.0, 0.0, 0.0])
    _set_effective_stress(state, committed)

    response = material.integrate(
        total_strain_2d=np.zeros(3),
        strain_increment_2d=np.zeros(3),
        committed_state=state,
        context=None,
    )

    if hasattr(response.state, "effective_stress"):
        stress_new = response.state.effective_stress
    else:
        stress_new = response.state.stress

    assert np.allclose(stress_new, committed)
    assert response.tangent.shape == (3, 3)


def test_elastic_tangent_shape():
    material = _material()
    state = GaussPointState()

    response = material.integrate(
        total_strain_2d=np.array([1.0e-7, -3.0e-8, 2.0e-8]),
        strain_increment_2d=np.array([1.0e-7, -3.0e-8, 2.0e-8]),
        committed_state=state,
        context=None,
    )

    assert response.tangent.shape == (3, 3)
    assert np.all(np.isfinite(response.tangent))
