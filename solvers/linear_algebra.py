"""Linear-system backend shared by GeoPyFEM mechanical solvers.

The nonlinear algorithms should not care whether the global matrix is stored
as a dense NumPy array or a sparse SciPy CSR matrix.  This module provides that
small abstraction while retaining the current direct-solver policy.
"""

import warnings
from time import perf_counter

import numpy as np

try:
    from scipy.sparse import bmat, csr_matrix, issparse
    from scipy.sparse.linalg import MatrixRankWarning, lsmr, spsolve
    SCIPY_AVAILABLE = True
except ImportError:  # Preserve the historical dense-only execution path.
    SCIPY_AVAILABLE = False

    def issparse(_):
        return False


def _require_scipy_for_sparse():
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "Sparse linear algebra requires SciPy. Install scipy or use "
            "<MatrixStorage>dense</MatrixStorage>."
        )


def matrix_storage_name(matrix):
    return "sparse" if issparse(matrix) else "dense"


def solve_direct(matrix, rhs, profiler=None):
    """Solve ``matrix @ x = rhs`` using the appropriate direct backend."""
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    profile_start = perf_counter() if profiler is not None else None
    if profiler is not None:
        profiler.increment("linear_solve_calls")

    if issparse(matrix):
        matrix = matrix.tocsr()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            try:
                solution = spsolve(matrix, rhs)
            except MatrixRankWarning as error:
                raise np.linalg.LinAlgError(
                    f"sparse matrix is rank deficient: {error}"
                ) from error
            except RuntimeError as error:
                raise np.linalg.LinAlgError(
                    f"sparse direct solve failed: {error}"
                ) from error

        solution = np.asarray(solution, dtype=float).reshape(-1)
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                "sparse direct solve returned non-finite values."
            )
        if profiler is not None:
            profiler.add_time("linear_solve", perf_counter() - profile_start)
            profiler.increment("sparse_direct_solves")
        return solution

    try:
        solution = np.linalg.solve(np.asarray(matrix, dtype=float), rhs)
    except np.linalg.LinAlgError:
        raise

    if profiler is not None:
        profiler.add_time("linear_solve", perf_counter() - profile_start)
        profiler.increment("dense_direct_solves")

    return solution


def solve_least_squares(matrix, rhs, profiler=None):
    """Least-squares fallback used only for arc-length predictor directions."""
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    profile_start = perf_counter() if profiler is not None else None
    if profiler is not None:
        profiler.increment("linear_solve_calls")
        profiler.increment("least_squares_solves")

    if issparse(matrix):
        solution = lsmr(matrix.tocsr(), rhs)[0]
        solution = np.asarray(solution, dtype=float).reshape(-1)
    else:
        solution, *_ = np.linalg.lstsq(
            np.asarray(matrix, dtype=float),
            rhs,
            rcond=None,
        )

    if not np.all(np.isfinite(solution)):
        raise np.linalg.LinAlgError(
            "least-squares solve returned non-finite values."
        )

    if profiler is not None:
        profiler.add_time("linear_solve", perf_counter() - profile_start)

    return solution

def extract_submatrix(matrix, row_dofs, col_dofs=None):
    """Extract a dense/sparse submatrix without changing storage type."""
    row_dofs = np.asarray(row_dofs, dtype=int).reshape(-1)
    if col_dofs is None:
        col_dofs = row_dofs
    else:
        col_dofs = np.asarray(col_dofs, dtype=int).reshape(-1)

    if issparse(matrix):
        return matrix.tocsr()[row_dofs, :][:, col_dofs]

    return np.asarray(matrix)[np.ix_(row_dofs, col_dofs)]


def solve_augmented_arc_system(
    K_ff,
    q_ff,
    grad_u,
    grad_factor,
    rhs_equilibrium,
    rhs_constraint,
    profiler=None,
):
    """Solve the augmented arc-length Newton corrector system.

    The system is

        [ K_ff   q ] [du] = [r]
        [ g_u^T gM] [dM]   [c]

    and may be dense or sparse depending on ``K_ff``.
    """
    q_ff = np.asarray(q_ff, dtype=float).reshape(-1)
    grad_u = np.asarray(grad_u, dtype=float).reshape(-1)
    rhs_equilibrium = np.asarray(rhs_equilibrium, dtype=float).reshape(-1)

    n = q_ff.size
    if grad_u.size != n or rhs_equilibrium.size != n:
        raise ValueError("Arc-length augmented-system vector sizes are inconsistent.")

    rhs = np.empty(n + 1, dtype=float)
    rhs[:-1] = rhs_equilibrium
    rhs[-1] = float(rhs_constraint)

    build_start = perf_counter() if profiler is not None else None

    if issparse(K_ff):
        augmented = bmat(
            [
                [K_ff.tocsr(), csr_matrix(q_ff.reshape(-1, 1))],
                [csr_matrix(grad_u.reshape(1, -1)), csr_matrix([[float(grad_factor)]])],
            ],
            format="csr",
        )
    else:
        augmented = np.empty((n + 1, n + 1), dtype=float)
        augmented[:-1, :-1] = np.asarray(K_ff, dtype=float)
        augmented[:-1, -1] = q_ff
        augmented[-1, :-1] = grad_u
        augmented[-1, -1] = float(grad_factor)

    if profiler is not None:
        profiler.add_time(
            "arc_augmented_system_build",
            perf_counter() - build_start,
        )
        profiler.increment("arc_augmented_system_builds")

    return solve_direct(augmented, rhs, profiler=profiler)
