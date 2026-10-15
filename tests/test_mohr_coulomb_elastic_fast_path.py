import numpy as np

from material_model.constitutive import MaterialContext
from material_model.mohr_coulomb import MohrCoulomb
from solvers.profiler import SolverProfiler
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


def _integrate(material, strain, profiler=None):
    strain = np.asarray(strain, dtype=float)
    return material.integrate(
        total_strain_2d=strain,
        strain_increment_2d=strain,
        committed_state=GaussPointState(),
        context=MaterialContext(performance_profiler=profiler),
    )


def test_analytic_plane_strain_principal_values_match_full_eigh():
    material = _material()
    rng = np.random.default_rng(12031991)

    for _ in range(500):
        stress = np.zeros(6, dtype=float)
        stress[0] = rng.normal(scale=100.0)
        stress[1] = rng.normal(scale=100.0)
        stress[2] = rng.normal(scale=100.0)
        stress[3] = rng.normal(scale=60.0)

        fast = material._plane_strain_principal_values_compression_positive(
            stress
        )
        full, _ = material._principal_decomposition_compression_positive(
            stress
        )

        assert np.allclose(fast, full, rtol=2.0e-14, atol=2.0e-12)


def test_elastic_fast_path_skips_full_eigendecomposition():
    profiler = SolverProfiler()
    response = _integrate(
        _material(),
        [-1.0e-5, -1.0e-5, 0.0],
        profiler=profiler,
    )
    counts = profiler.snapshot()["counts"]

    assert response.diagnostics["region"] == "elastic"
    assert not response.state.yielded
    assert counts["mc_elastic_fast_returns"] == 1
    assert counts["mc_analytic_principal_value_evaluations"] == 1
    assert counts.get("mc_principal_decompositions", 0) == 0
    assert counts.get("mc_principal_return_calls", 0) == 0


def test_plastic_response_retains_full_spectral_return():
    profiler = SolverProfiler()
    response = _integrate(
        _material(),
        [0.005, -0.01, 0.0],
        profiler=profiler,
    )
    counts = profiler.snapshot()["counts"]

    assert response.state.yielded
    assert response.diagnostics["region"] == "face"
    assert response.diagnostics["tangent_mode"] == "analytic_spectral"
    assert counts["mc_analytic_principal_value_evaluations"] == 1
    assert counts["mc_fast_path_plastic_candidates"] == 1
    assert counts["mc_principal_decompositions"] == 1
    assert counts["mc_principal_return_calls"] == 1
    assert counts["mc_analytic_spectral_tangent_calls"] == 1


def test_non_plane_strain_shear_declines_fast_principal_path():
    material = _material()
    stress = np.array([10.0, 20.0, 30.0, 5.0, 1.0e-3, 0.0])

    assert (
        material._plane_strain_principal_values_compression_positive(stress)
        is None
    )
