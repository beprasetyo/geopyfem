import numpy as np
from scipy.sparse import issparse

from assembly.global_stiffness import assemble_global_stiffness
from assembly.matrix_builder import SparseAssemblyPattern
from elements.kinematics_cache import ElementKinematicsCache
from material_model.linear_elastic import LinearElastic
from solvers.profiler import SolverProfiler


def _two_quad_problem():
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    elements = [
        {
            "id": 10,
            "type": "Quad4",
            "connectivity": np.array([0, 1, 4, 3], dtype=int),
            "material": "soil",
        },
        {
            "id": 20,
            "type": "Quad4",
            "connectivity": np.array([1, 2, 5, 4], dtype=int),
            "material": "soil",
        },
    ]
    materials = {
        "soil": LinearElastic(
            {"E": 30000.0, "nu": 0.30, "density": 1800.0}
        )
    }
    return nodes, elements, materials


def test_fixed_csr_pattern_matches_dynamic_sparse_and_dense():
    nodes, elements, materials = _two_quad_problem()
    cache = ElementKinematicsCache(nodes, elements)
    pattern = SparseAssemblyPattern.from_kinematics_cache(
        size=2 * len(nodes),
        kinematics_cache=cache,
    )

    dense = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        matrix_storage="dense",
        kinematics_cache=cache,
    )
    dynamic = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        matrix_storage="sparse",
        kinematics_cache=cache,
    )
    fixed = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        matrix_storage="sparse",
        kinematics_cache=cache,
        sparse_pattern=pattern,
    )

    assert issparse(dynamic)
    assert issparse(fixed)
    np.testing.assert_allclose(dynamic.toarray(), dense, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(fixed.toarray(), dense, rtol=0.0, atol=1.0e-12)


def test_one_pattern_is_valid_for_different_active_element_sets():
    nodes, elements, materials = _two_quad_problem()
    cache = ElementKinematicsCache(nodes, elements)
    pattern = SparseAssemblyPattern.from_kinematics_cache(
        size=2 * len(nodes),
        kinematics_cache=cache,
    )

    first_only = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        active_element_ids={10},
        matrix_storage="sparse",
        kinematics_cache=cache,
        sparse_pattern=pattern,
    )
    second_only = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        active_element_ids={20},
        matrix_storage="sparse",
        kinematics_cache=cache,
        sparse_pattern=pattern,
    )

    # Construction-stage activity changes only data values, not CSR topology.
    np.testing.assert_array_equal(first_only.indptr, second_only.indptr)
    np.testing.assert_array_equal(first_only.indices, second_only.indices)
    assert first_only.nnz == pattern.nnz
    assert second_only.nnz == pattern.nnz
    assert not np.allclose(first_only.data, second_only.data)


def test_fixed_csr_path_is_reported_by_profiler():
    nodes, elements, materials = _two_quad_problem()
    cache = ElementKinematicsCache(nodes, elements)
    pattern = SparseAssemblyPattern.from_kinematics_cache(
        size=2 * len(nodes),
        kinematics_cache=cache,
    )
    profiler = SolverProfiler()

    assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        matrix_storage="sparse",
        kinematics_cache=cache,
        sparse_pattern=pattern,
        profiler=profiler,
    )

    snapshot = profiler.snapshot()
    assert snapshot["counts"]["fixed_csr_scatter_calls"] == 2
    assert snapshot["counts"]["fixed_csr_finalize_calls"] == 1
    assert snapshot["counts"].get("dynamic_coo_finalize_calls", 0) == 0


def test_stage_specific_pattern_matches_full_pattern_for_active_subset():
    nodes, elements, materials = _two_quad_problem()
    cache = ElementKinematicsCache(nodes, elements)
    full_pattern = SparseAssemblyPattern.from_kinematics_cache(
        size=2 * len(nodes),
        kinematics_cache=cache,
    )
    stage_pattern = SparseAssemblyPattern.from_kinematics_cache(
        size=2 * len(nodes),
        kinematics_cache=cache,
        active_element_ids={10},
    )

    full_matrix = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        active_element_ids={10},
        matrix_storage="sparse",
        kinematics_cache=cache,
        sparse_pattern=full_pattern,
    )
    stage_matrix = assemble_global_stiffness(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        active_element_ids={10},
        matrix_storage="sparse",
        kinematics_cache=cache,
        sparse_pattern=stage_pattern,
    )

    np.testing.assert_allclose(
        stage_matrix.toarray(), full_matrix.toarray(), rtol=0.0, atol=1.0e-12
    )
    assert stage_pattern.nnz <= full_pattern.nnz
    assert stage_pattern.number_of_element_maps == 1


def test_fixed_csr_preserves_nonsymmetric_element_contributions():
    from assembly.matrix_builder import GlobalMatrixBuilder

    nodes, elements, _ = _two_quad_problem()
    cache = ElementKinematicsCache(nodes, elements)
    pattern = SparseAssemblyPattern.from_kinematics_cache(
        size=2 * len(nodes),
        kinematics_cache=cache,
    )

    rng = np.random.default_rng(1234)
    element_matrices = {
        element["id"]: rng.normal(size=(8, 8))
        for element in elements
    }

    dynamic = GlobalMatrixBuilder(
        size=2 * len(nodes), matrix_storage="sparse"
    )
    fixed = GlobalMatrixBuilder(
        size=2 * len(nodes),
        matrix_storage="sparse",
        sparse_pattern=pattern,
    )

    for element in elements:
        element_id = element["id"]
        dofs = cache[element_id].dofs
        matrix = element_matrices[element_id]
        dynamic.add(dofs, matrix, element_id=element_id)
        fixed.add(dofs, matrix, element_id=element_id)

    np.testing.assert_allclose(
        fixed.finalize().toarray(),
        dynamic.finalize().toarray(),
        rtol=0.0,
        atol=1.0e-13,
    )
