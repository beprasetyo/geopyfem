"""Lightweight timing profiler for GeoPyFEM solver hot paths.

The profiler is intentionally numerical-policy agnostic.  It only accumulates
wall-clock timings and call counts under string keys.  Solver/assembly modules
may receive ``profiler=None`` and therefore keep their historical behaviour.

Two timing levels are used:

Primary categories
------------------
These are intended to partition most of a nonlinear stage without overlap:

- ``constitutive_update``
- ``internal_force_assembly``
- ``tangent_assembly``
- ``linear_solve``
- ``k0_initialization``
- ``global_stiffness_assembly`` (single-phase linear analysis)
- ``linear_state_update``       (single-phase linear analysis)

Detail categories
-----------------
These are nested inside primary categories and therefore must NOT be added to
primary times when computing percentages:

- ``constitutive_integration``
- ``tangent_element_integration``
- ``matrix_scatter``
- ``matrix_finalize``
- ``sparse_pattern_build`` (one-time/active-domain setup outside Newton stages)
- ``arc_augmented_system_build``
- ``arc_factor_sensitivity``
- ``mc_*`` fine-grained Mohr-Coulomb local-integration timings

The distinction lets the main program report a clean stage-level profile while
retaining enough detail to decide where the next optimization should occur.
"""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


PRIMARY_TIMING_KEYS = (
    "constitutive_update",
    "internal_force_assembly",
    "tangent_assembly",
    "linear_solve",
    "k0_initialization",
    "global_stiffness_assembly",
    "linear_state_update",
)


class SolverProfiler:
    """Accumulate wall-clock timings and call counts for one solver scope."""

    def __init__(self):
        self._timings = defaultdict(float)
        self._counts = defaultdict(int)

    def add_time(self, name, elapsed):
        elapsed = float(elapsed)
        if elapsed < 0.0:
            raise ValueError("Profiler elapsed time cannot be negative.")
        self._timings[str(name)] += elapsed

    def increment(self, name, count=1):
        self._counts[str(name)] += int(count)

    @contextmanager
    def section(self, name, count_name=None):
        start = perf_counter()
        try:
            yield
        finally:
            self.add_time(name, perf_counter() - start)
            if count_name is not None:
                self.increment(count_name)

    def snapshot(self, total_elapsed=None):
        timings = {
            key: float(value)
            for key, value in sorted(self._timings.items())
        }
        counts = {
            key: int(value)
            for key, value in sorted(self._counts.items())
        }

        primary_total = sum(
            timings.get(key, 0.0)
            for key in PRIMARY_TIMING_KEYS
        )

        result = {
            "timings": timings,
            "counts": counts,
            "primary_timing_keys": list(PRIMARY_TIMING_KEYS),
            "primary_profiled_time": float(primary_total),
        }

        if total_elapsed is not None:
            total_elapsed = max(float(total_elapsed), 0.0)
            result["total_elapsed_time"] = total_elapsed
            result["other_time"] = max(total_elapsed - primary_total, 0.0)

        return result


def profile_add_time(profiler, name, elapsed):
    if profiler is not None:
        profiler.add_time(name, elapsed)


def profile_increment(profiler, name, count=1):
    if profiler is not None:
        profiler.increment(name, count)


@contextmanager
def profile_section(profiler, name, count_name=None):
    """No-op context manager when profiling is not requested."""
    if profiler is None:
        yield
        return

    with profiler.section(name, count_name=count_name):
        yield
