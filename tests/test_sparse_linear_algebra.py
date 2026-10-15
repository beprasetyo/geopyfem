import numpy as np
from scipy.sparse import csr_matrix, issparse

from assembly.boundary_condition import apply_zero_displacement_constraints
from assembly.global_stiffness import assemble_global_stiffness
from material_model.linear_elastic import LinearElastic
from solvers.common import _update_trial_gauss_states
from solvers.linear_algebra import solve_direct
from state.state_manager import GaussPointStateManager


def _one_quad_problem():
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    elements = [
        {
            "id": 0,
            "type": "Quad4",
            "connectivity": np.array([0, 1, 2, 3], dtype=int),
            "material": "soil",
        }
    ]
    materials = {
        "soil": LinearElastic(
            {
                "E": 30000.0,
                "nu": 0.30,
                "density": 1800.0,
            }
        )
    }
    return nodes, elements, materials


def test_dense_and_sparse_global_stiffness_match():
    nodes, elements, materials = _one_quad_problem()

    K_dense = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        matrix_storage="dense",
    )
    K_sparse = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        matrix_storage="sparse",
    )

    assert isinstance(K_dense, np.ndarray)
    assert issparse(K_sparse)
    np.testing.assert_allclose(K_sparse.toarray(), K_dense, rtol=0.0, atol=1.0e-12)


def test_dense_and_sparse_direct_solve_match_with_constraints():
    K = np.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=float,
    )
    rhs = np.array([0.0, 2.0, 3.0], dtype=float)
    fixed = [0]

    Kd, fd = apply_zero_displacement_constraints(K, rhs, fixed)
    Ks, fs = apply_zero_displacement_constraints(csr_matrix(K), rhs, fixed)

    ud = solve_direct(Kd, fd)
    us = solve_direct(Ks, fs)

    np.testing.assert_allclose(us, ud, rtol=1.0e-13, atol=1.0e-13)
    assert us[0] == 0.0


def test_common_trial_update_stores_constitutive_response_tangents():
    nodes, elements, materials = _one_quad_problem()
    states = GaussPointStateManager(elements)

    u = np.zeros(2 * len(nodes), dtype=float)
    u[4] = 1.0e-4
    u[5] = -2.0e-4

    states.reset_trial()
    _update_trial_gauss_states(
        nodes=nodes,
        mesh_elements=elements,
        u=u,
        material_models=materials,
        state_manager=states,
        active_element_ids={0},
        stage_id=1,
        stage_name="test",
        load_factor=0.5,
        increment_number=1,
        iteration_number=1,
        load_path_kind="test",
    )

    expected = materials["soil"].tangent_matrix()
    assert 0 in states.trial_tangents
    for tangent in states.trial_tangents[0]:
        assert tangent is not None
        np.testing.assert_allclose(tangent, expected, rtol=0.0, atol=0.0)


def test_dense_and_sparse_augmented_arc_system_match():
    from solvers.linear_algebra import solve_augmented_arc_system

    K_dense = np.array(
        [
            [12.0, 1.5],
            [0.5, 9.0],
        ],
        dtype=float,
    )
    q = np.array([2.0, -1.0], dtype=float)
    grad_u = np.array([0.2, -0.1], dtype=float)
    grad_factor = 0.7
    rhs_eq = np.array([1.0, -0.5], dtype=float)
    rhs_arc = -0.03

    dense = solve_augmented_arc_system(
        K_ff=K_dense,
        q_ff=q,
        grad_u=grad_u,
        grad_factor=grad_factor,
        rhs_equilibrium=rhs_eq,
        rhs_constraint=rhs_arc,
    )
    sparse = solve_augmented_arc_system(
        K_ff=csr_matrix(K_dense),
        q_ff=q,
        grad_u=grad_u,
        grad_factor=grad_factor,
        rhs_equilibrium=rhs_eq,
        rhs_constraint=rhs_arc,
    )

    np.testing.assert_allclose(sparse, dense, rtol=1.0e-13, atol=1.0e-13)
