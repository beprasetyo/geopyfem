"""Small architecture regression checks for the solver facade."""

import solver

from solvers.static_linear import solve_static_linear
from solvers.nonlinear import solve_static_nonlinear
from solvers.staged import solve_staged_static
from solvers.time_dependent import solve_time_dependent


def test_solver_facade_keeps_public_entry_points():
    assert solver.solve_static_linear is solve_static_linear
    assert solver.solve_static_nonlinear is solve_static_nonlinear
    assert solver.solve_staged_static is solve_staged_static
    assert solver.solve_time_dependent is solve_time_dependent
    assert callable(solver.solve_problem)


def test_solver_facade_keeps_historical_private_helpers_importable():
    # These aliases are retained only for backwards compatibility while new
    # code should import helpers from the owning solvers module directly.
    assert callable(solver._nonlinear_settings)
    assert callable(solver._update_trial_gauss_states)
    assert callable(solver._solve_nonlinear_load_path)
    assert callable(solver._solve_strength_reduction_path)
    assert callable(solver._solve_k0_initial_stage)
