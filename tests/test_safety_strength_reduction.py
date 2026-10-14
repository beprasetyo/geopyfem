import xml.etree.ElementTree as ET

import numpy as np
import pytest

from material_model.mohr_coulomb import MohrCoulomb
from problem_reader import _read_stages
from stage.safety import (
    normalize_strength_reduction_settings,
    validate_safety_stage_definition,
    apply_strength_reduction,
    restore_reference_strength,
    characteristic_length,
    arc_metric_squared,
    predictor_direction_sign,
    plateau_diagnostics,
)


def _mc():
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


def test_mc_strength_reduction_is_absolute_not_compounded():
    material = _mc()

    material.set_strength_reduction_factor(2.0)
    assert material.cohesion == pytest.approx(5.0)
    assert material.friction_angle == pytest.approx(
        np.degrees(np.arctan(np.tan(np.radians(30.0)) / 2.0))
    )
    assert material.dilation_angle == pytest.approx(5.0)

    material.set_strength_reduction_factor(4.0)
    assert material.cohesion == pytest.approx(2.5)
    assert material.friction_angle == pytest.approx(
        np.degrees(np.arctan(np.tan(np.radians(30.0)) / 4.0))
    )
    assert material.dilation_angle == pytest.approx(5.0)

    material.restore_reference_strength()
    assert material.strength_reduction_factor == pytest.approx(1.0)
    assert material.cohesion == pytest.approx(10.0)
    assert material.friction_angle == pytest.approx(30.0)
    assert material.dilation_angle == pytest.approx(5.0)


def test_safety_helpers_apply_and_restore_mc_strength():
    material = _mc()
    models = {"soil": material}

    summary = apply_strength_reduction(models, 1.5)
    assert summary["soil"]["factor"] == pytest.approx(1.5)
    assert material.cohesion == pytest.approx(10.0 / 1.5)

    restored = restore_reference_strength(models)
    assert restored["soil"]["factor"] == pytest.approx(1.0)
    assert material.cohesion == pytest.approx(10.0)


def test_strength_reduction_settings_require_initial_factor_one():
    with pytest.raises(ValueError):
        normalize_strength_reduction_settings({"initial_factor": 1.1})

    settings = normalize_strength_reduction_settings({
        "initial_factor_increment": 0.02,
        "maximum_factor": 2.5,
        "window_steps": 10,
    })

    assert settings["initial_factor"] == pytest.approx(1.0)
    assert settings["initial_factor_increment"] == pytest.approx(0.02)
    assert settings["maximum_factor"] == pytest.approx(2.5)
    assert settings["window_steps"] == 10


def test_legacy_strength_reduction_settings_are_translated():
    settings = normalize_strength_reduction_settings({
        "initial_increment": 0.05,
        "minimum_increment": 0.005,
        "maximum_increment": 0.10,
        "factor_tolerance": 0.002,
    })

    assert settings["initial_factor_increment"] == pytest.approx(0.05)
    assert settings["minimum_radius_factor"] == pytest.approx(0.10)
    assert settings["maximum_radius_factor"] == pytest.approx(2.0)
    assert settings["factor_range_tolerance"] == pytest.approx(0.002)


def test_safety_stage_rejects_load_or_construction_changes():
    validate_safety_stage_definition({
        "procedure": "safety",
        "activate_loads": [],
        "deactivate_loads": [],
        "activate_regions": [],
        "deactivate_regions": [],
        "gravity": None,
        "k0": None,
    })

    with pytest.raises(ValueError):
        validate_safety_stage_definition({
            "procedure": "safety",
            "activate_loads": ["new_load"],
        })


def test_problem_reader_reads_nested_arc_length_safety_block():
    root = ET.fromstring(
        """
        <Problem>
          <Stages>
            <Stage id="3" name="Safety">
              <Procedure>safety</Procedure>
              <StrengthReduction>
                <InitialFactor>1.0</InitialFactor>
                <InitialFactorIncrement>0.01</InitialFactorIncrement>
                <MaximumFactor>3.0</MaximumFactor>
                <MaximumSteps>120</MaximumSteps>
                <ArcLength>
                  <MinimumRadiusFactor>0.05</MinimumRadiusFactor>
                  <MaximumRadiusFactor>2.0</MaximumRadiusFactor>
                  <FactorWeight>1.0</FactorWeight>
                  <ConstraintTolerance>1.0e-7</ConstraintTolerance>
                </ArcLength>
                <FailureDetection>
                  <WindowSteps>12</WindowSteps>
                  <FactorRangeTolerance>0.005</FactorRangeTolerance>
                  <DisplacementGrowthRatio>0.5</DisplacementGrowthRatio>
                  <MinimumDisplacementRatio>0.0001</MinimumDisplacementRatio>
                </FailureDetection>
              </StrengthReduction>
            </Stage>
          </Stages>
        </Problem>
        """
    )

    stages = _read_stages(root)
    settings = stages[0]["strength_reduction"]

    assert stages[0]["procedure"] == "safety"
    assert settings["initial_factor"] == pytest.approx(1.0)
    assert settings["initial_factor_increment"] == pytest.approx(0.01)
    assert settings["maximum_factor"] == pytest.approx(3.0)
    assert settings["maximum_steps"] == 120
    assert settings["minimum_radius_factor"] == pytest.approx(0.05)
    assert settings["factor_weight"] == pytest.approx(1.0)
    assert settings["window_steps"] == 12
    assert settings["factor_range_tolerance"] == pytest.approx(0.005)


def test_arc_metric_is_mesh_size_normalized():
    nodes = np.array([[0.0, 0.0], [3.0, 4.0]])
    L = characteristic_length(nodes)
    assert L == pytest.approx(5.0)

    du = np.array([0.0, 0.0, 0.5, 0.0])
    free = np.array([2, 3])

    value = arc_metric_squared(
        delta_u=du,
        delta_factor=0.1,
        free_dofs=free,
        displacement_scale=L,
        factor_weight=1.0,
    )

    rms = 0.5 / np.sqrt(2.0)
    expected = (rms / 5.0) ** 2 + 0.1 ** 2
    assert value == pytest.approx(expected)


def test_predictor_direction_continues_previous_arc_branch():
    z = np.array([1.0, 0.0])
    free = np.array([0, 1])

    positive = predictor_direction_sign(
        tangent_du_per_factor=z,
        previous_delta_u=np.array([0.1, 0.0]),
        previous_delta_factor=0.05,
        free_dofs=free,
        displacement_scale=1.0,
        factor_weight=1.0,
    )
    assert positive == 1.0

    negative = predictor_direction_sign(
        tangent_du_per_factor=z,
        previous_delta_u=np.array([-0.1, 0.0]),
        previous_delta_factor=-0.05,
        free_dofs=free,
        displacement_scale=1.0,
        factor_weight=1.0,
    )
    assert negative == -1.0


def test_plateau_requires_factor_flatness_and_deformation_growth():
    settings = normalize_strength_reduction_settings({
        "window_steps": 6,
        "factor_range_tolerance": 0.005,
        "displacement_growth_ratio": 0.50,
        "minimum_displacement_ratio": 1.0e-4,
    })

    history = [{
        "step": 0,
        "strength_reduction_factor": 1.0,
        "maximum_safety_displacement": 0.0,
        "yielded_gauss_points": 0,
        "converged": True,
    }]

    factors = [1.366, 1.368, 1.369, 1.3685, 1.3675, 1.367]
    displacements = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

    for i, (factor, displacement) in enumerate(
        zip(factors, displacements), start=1
    ):
        history.append({
            "step": i,
            "strength_reduction_factor": factor,
            "maximum_safety_displacement": displacement,
            "yielded_gauss_points": 20,
            "converged": True,
        })

    result = plateau_diagnostics(
        history=history,
        settings=settings,
        characteristic_length_value=10.0,
    )

    assert result["plateau"] is True
    assert result["peak_factor"] == pytest.approx(1.369)
    assert result["factor_range"] <= 0.005
    assert result["deformation_growing"] is True
    assert result["plasticity_present"] is True


def test_rising_factor_path_is_not_a_plateau():
    settings = normalize_strength_reduction_settings({
        "window_steps": 6,
        "factor_range_tolerance": 0.005,
        "displacement_growth_ratio": 0.10,
    })

    history = []
    for i in range(1, 7):
        history.append({
            "step": i,
            "strength_reduction_factor": 1.0 + 0.02 * i,
            "maximum_safety_displacement": 0.01 * i,
            "yielded_gauss_points": 5,
            "converged": True,
        })

    result = plateau_diagnostics(
        history=history,
        settings=settings,
        characteristic_length_value=10.0,
    )

    assert result["plateau"] is False
    assert result["factor_plateau"] is False


def test_fortran_apex_tolerates_near_zero_correction_direction():
    material = _mc()
    phi = np.radians(material.friction_angle)
    p_apex = -material.cohesion / np.tan(phi)

    trial = np.full(3, p_apex + 1.0e-12)

    result = material._fortran_apex_return(
        stress_trial=trial,
        p_apex=p_apex,
        f13_trial=1.0,
    )

    assert result["region"] == "apex_fortran"
    assert np.all(np.isfinite(result["tangent_principal"]))
    assert np.allclose(
        result["stress"],
        np.full(3, p_apex),
        atol=1.0e-12,
    )
