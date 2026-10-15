import numpy as np

from assembly.global_stiffness import assemble_global_stiffness
from assembly.internal_force import assemble_internal_force
from assembly.tangent_stiffness import assemble_tangent_stiffness
from elements.kinematics_cache import ElementKinematicsCache
from material_model.linear_elastic import LinearElastic
from solvers.common import _update_trial_gauss_states
from state.state_manager import GaussPointStateManager


def _one_quad_problem():
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    elements = [{
        "id": 0,
        "type": "Quad4",
        "connectivity": np.array([0, 1, 2, 3], dtype=int),
        "material": "soil",
    }]
    materials = {
        "soil": LinearElastic({"E": 30000.0, "nu": 0.30, "density": 1800.0})
    }
    return nodes, elements, materials


def test_cache_matches_uncached_linear_stiffness():
    nodes, elements, materials = _one_quad_problem()
    cache = ElementKinematicsCache(nodes, elements)

    assert cache.number_of_elements == 1
    assert cache.number_of_gauss_points == 4
    assert cache[0].dofs.tolist() == list(range(8))

    K_plain = assemble_global_stiffness(
        nodes, elements, materials, matrix_storage="dense"
    )
    K_cache = assemble_global_stiffness(
        nodes, elements, materials, matrix_storage="dense",
        kinematics_cache=cache,
    )
    np.testing.assert_allclose(K_cache, K_plain, rtol=0.0, atol=0.0)


def test_cache_matches_uncached_constitutive_internal_force_and_tangent():
    nodes, elements, materials = _one_quad_problem()
    cache = ElementKinematicsCache(nodes, elements)

    state_plain = GaussPointStateManager(elements)
    state_cache = GaussPointStateManager(elements)

    u = np.zeros(8, dtype=float)
    u[4] = 2.0e-4
    u[5] = -1.5e-4

    for manager, cache_arg in ((state_plain, None), (state_cache, cache)):
        manager.reset_trial()
        _update_trial_gauss_states(
            nodes=nodes,
            mesh_elements=elements,
            u=u,
            material_models=materials,
            state_manager=manager,
            active_element_ids={0},
            stage_id=1,
            load_factor=1.0,
            increment_number=1,
            iteration_number=1,
            load_path_kind="test",
            kinematics_cache=cache_arg,
        )

    f_plain = assemble_internal_force(
        nodes, elements, state_plain.trial, active_element_ids={0}
    )
    f_cache = assemble_internal_force(
        nodes, elements, state_cache.trial, active_element_ids={0},
        kinematics_cache=cache,
    )
    np.testing.assert_allclose(f_cache, f_plain, rtol=0.0, atol=1.0e-14)

    K_plain = assemble_tangent_stiffness(
        nodes, elements,
        gauss_tangents=state_plain.trial_tangents,
        active_element_ids={0},
        matrix_storage="dense",
    )
    K_cache = assemble_tangent_stiffness(
        nodes, elements,
        gauss_tangents=state_cache.trial_tangents,
        active_element_ids={0},
        matrix_storage="dense",
        kinematics_cache=cache,
    )
    np.testing.assert_allclose(K_cache, K_plain, rtol=0.0, atol=0.0)


def test_cached_arrays_are_read_only():
    nodes, elements, _ = _one_quad_problem()
    cache = ElementKinematicsCache(nodes, elements)
    gp = cache[0].gauss_points[0]

    assert not cache[0].dofs.flags.writeable
    assert not cache[0].connectivity.flags.writeable
    assert not gp.B.flags.writeable
    assert not gp.dN_global.flags.writeable
    assert not gp.natural_coordinates.flags.writeable
