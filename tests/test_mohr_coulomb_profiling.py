import numpy as np
import pytest

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


def test_mc_fine_profiler_records_elastic_path():
    profiler = SolverProfiler()
    response = _integrate(
        _material(),
        [-1.0e-5, -1.0e-5, 0.0],
        profiler=profiler,
    )
    profile = profiler.snapshot()
    timings = profile["timings"]
    counts = profile["counts"]

    assert response.diagnostics["region"] == "elastic"
    assert counts["mc_constitutive_calls"] == 1
    assert counts["mc_state_integrations"] == 1
    assert counts["mc_region_elastic"] == 1
    assert counts["mc_tangent_mode_elastic"] == 1
    assert counts["mc_elastic_fast_returns"] == 1
    assert counts["mc_analytic_principal_value_evaluations"] == 1
    assert counts.get("mc_principal_decompositions", 0) == 0

    for key in (
        "mc_integrate_total",
        "mc_state_integration",
        "mc_elastic_predictor",
        "mc_analytic_principal_values",
        "mc_elastic_fast_yield_check",
        "mc_state_copy_update",
        "mc_state_diagnostics",
        "mc_tangent_total",
        "mc_response_construction",
    ):
        assert timings[key] >= 0.0

    assert timings["mc_integrate_total"] >= timings["mc_state_integration"]


def test_mc_fine_profiler_records_plastic_face_and_analytic_tangent():
    profiler = SolverProfiler()
    response = _integrate(
        _material(),
        [0.005, -0.01, 0.0],
        profiler=profiler,
    )
    profile = profiler.snapshot()
    timings = profile["timings"]
    counts = profile["counts"]

    assert response.state.yielded
    assert response.diagnostics["region"] == "face"
    assert response.diagnostics["tangent_mode"] == "analytic_spectral"

    assert counts["mc_region_face"] == 1
    assert counts["mc_face_candidate_evaluations"] == 1
    assert counts["mc_active_set_linear_solve_calls"] == 2
    assert counts["mc_analytic_spectral_tangent_calls"] == 1
    assert counts["mc_tangent_mode_analytic_spectral"] == 1

    assert timings["mc_face_candidate"] > 0.0
    assert timings["mc_active_set_linear_solve"] > 0.0
    assert timings["mc_analytic_spectral_tangent"] > 0.0
    assert timings["mc_tangent_total"] >= timings["mc_analytic_spectral_tangent"]


def test_mc_profiling_is_numerically_transparent():
    material_a = _material()
    material_b = _material()
    strain = np.array([0.005, -0.01, 0.0], dtype=float)

    response_plain = material_a.integrate(
        total_strain_2d=strain,
        strain_increment_2d=strain,
        committed_state=GaussPointState(),
        context=MaterialContext(),
    )

    profiler = SolverProfiler()
    response_profiled = material_b.integrate(
        total_strain_2d=strain,
        strain_increment_2d=strain,
        committed_state=GaussPointState(),
        context=MaterialContext(performance_profiler=profiler),
    )

    assert response_profiled.diagnostics == response_plain.diagnostics
    assert np.array_equal(
        response_profiled.state.effective_stress,
        response_plain.state.effective_stress,
    )
    assert np.array_equal(
        response_profiled.state.plastic_strain,
        response_plain.state.plastic_strain,
    )
    assert np.array_equal(response_profiled.tangent, response_plain.tangent)
