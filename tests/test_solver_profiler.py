import numpy as np

from assembly.matrix_builder import GlobalMatrixBuilder
from solvers.linear_algebra import solve_direct
from solvers.profiler import SolverProfiler


def test_profiler_snapshot_primary_and_other_time():
    profiler = SolverProfiler()
    profiler.add_time("constitutive_update", 1.25)
    profiler.add_time("internal_force_assembly", 0.50)
    profiler.add_time("constitutive_integration", 1.00)
    profiler.increment("constitutive_calls", 12)

    snapshot = profiler.snapshot(total_elapsed=2.50)

    assert snapshot["timings"]["constitutive_update"] == 1.25
    assert snapshot["timings"]["constitutive_integration"] == 1.00
    assert snapshot["counts"]["constitutive_calls"] == 12
    assert snapshot["primary_profiled_time"] == 1.75
    assert snapshot["other_time"] == 0.75


def test_sparse_matrix_and_linear_solve_are_profiled():
    profiler = SolverProfiler()

    builder = GlobalMatrixBuilder(
        size=2,
        matrix_storage="sparse",
        profiler=profiler,
    )
    builder.add(
        np.array([0, 1]),
        np.array([[2.0, 0.0], [0.0, 4.0]]),
    )
    matrix = builder.finalize()

    solution = solve_direct(
        matrix,
        np.array([2.0, 8.0]),
        profiler=profiler,
    )

    np.testing.assert_allclose(solution, [1.0, 2.0])

    snapshot = profiler.snapshot()
    assert snapshot["counts"]["matrix_scatter_calls"] == 1
    assert snapshot["counts"]["matrix_finalize_calls"] == 1
    assert snapshot["counts"]["linear_solve_calls"] == 1
    assert snapshot["counts"]["sparse_direct_solves"] == 1
    assert snapshot["timings"]["matrix_scatter"] >= 0.0
    assert snapshot["timings"]["matrix_finalize"] >= 0.0
    assert snapshot["timings"]["linear_solve"] >= 0.0
