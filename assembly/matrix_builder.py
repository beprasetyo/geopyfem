"""Global matrix-assembly utilities.

This module centralizes the dense/sparse scatter operation used by both the
linear stiffness and nonlinear tangent assembly paths.

The element routine still computes a conventional dense element matrix ``Ke``.
Only the GLOBAL storage changes:

- ``dense``  -> NumPy ndarray
- ``sparse`` -> SciPy CSR

Sparse assembly supports two paths:

1. Backward-compatible dynamic COO accumulation followed by COO -> CSR.
2. A reusable :class:`SparseAssemblyPattern` whose CSR row/column structure
   and element-to-CSR scatter maps are built once.  Subsequent Newton tangent
   assemblies then zero only the CSR ``data`` array and accumulate new element
   values directly into their precomputed slots.

No symmetry is assumed because non-associated Mohr-Coulomb can produce a
non-symmetric algorithmic tangent.
"""

from time import perf_counter

import numpy as np


def normalize_matrix_storage(matrix_storage):
    value = str(matrix_storage or "dense").strip().lower()
    aliases = {
        "numpy": "dense",
        "ndarray": "dense",
        "scipy": "sparse",
        "csr": "sparse",
        "coo": "sparse",
    }
    value = aliases.get(value, value)

    if value not in {"dense", "sparse"}:
        raise ValueError(
            "MatrixStorage must be 'dense' or 'sparse'. "
            f'Received "{matrix_storage}".'
        )

    return value


def element_dof_indices(connectivity, ndof_node=2):
    """Return flattened global DOF numbers for one element connectivity."""
    connectivity = np.asarray(connectivity, dtype=int).reshape(-1)

    if ndof_node < 1:
        raise ValueError("ndof_node must be >= 1.")

    dofs = np.empty(connectivity.size * ndof_node, dtype=int)
    for local_dof in range(ndof_node):
        dofs[local_dof::ndof_node] = (
            ndof_node * connectivity + local_dof
        )
    return dofs


class SparseAssemblyPattern:
    """Reusable CSR sparsity pattern and element scatter maps.

    The mesh topology and global DOF numbering are invariant in GeoPyFEM's
    current small-strain formulation.  Therefore the locations in the global
    sparse matrix touched by every element are also invariant, even when
    construction stages activate/deactivate elements.

    The pattern is deliberately built from the *complete* continuum mesh.
    Inactive elements simply contribute no values during a particular stage.
    This means one CSR structure can be shared safely by K0, ordinary
    nonlinear, excavation, and safety stages.
    """

    def __init__(self, size, indptr, indices, element_positions):
        self.size = int(size)
        self.indptr = np.asarray(indptr, dtype=np.int64).copy()
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.element_positions = {
            int(element_id): np.asarray(positions, dtype=np.int64).copy()
            for element_id, positions in element_positions.items()
        }

        if self.indptr.ndim != 1 or self.indptr.size != self.size + 1:
            raise ValueError("CSR indptr has an invalid size.")
        if self.indices.ndim != 1:
            raise ValueError("CSR indices must be one-dimensional.")
        if int(self.indptr[-1]) != int(self.indices.size):
            raise ValueError("CSR indptr/indices are inconsistent.")

        # The structure is immutable; each matrix assembly owns only a fresh
        # floating-point data vector.
        self.indptr.setflags(write=False)
        self.indices.setflags(write=False)
        for positions in self.element_positions.values():
            positions.setflags(write=False)

    @property
    def nnz(self):
        return int(self.indices.size)

    @property
    def number_of_element_maps(self):
        return len(self.element_positions)

    @classmethod
    def from_kinematics_cache(
        cls,
        size,
        kinematics_cache,
        active_element_ids=None,
    ):
        """Build one fixed CSR pattern from cached element DOF maps.

        ``active_element_ids`` may be supplied for staged construction.  The
        resulting structure then contains only the current active domain plus
        a diagonal slot for every global DOF.  It can be reused for every
        Newton iteration while that active domain remains unchanged.
        """
        try:
            from scipy.sparse import coo_matrix
        except ImportError as error:
            raise ImportError(
                "Sparse matrix storage requires SciPy. Install scipy or "
                "use <MatrixStorage>dense</MatrixStorage>."
            ) from error

        size = int(size)
        if size < 1:
            raise ValueError("Global matrix size must be positive.")
        if kinematics_cache is None:
            raise ValueError(
                "SparseAssemblyPattern requires an ElementKinematicsCache."
            )

        row_blocks = []
        col_blocks = []
        element_dofs = {}

        if active_element_ids is None:
            selected_element_ids = kinematics_cache.element_ids()
        else:
            selected_element_ids = tuple(
                sorted({int(element_id) for element_id in active_element_ids})
            )

        for element_id in selected_element_ids:
            kinematics = kinematics_cache[element_id]
            dofs = np.asarray(kinematics.dofs, dtype=np.int64).reshape(-1)
            if np.any(dofs < 0) or np.any(dofs >= size):
                raise IndexError(
                    f"Element {element_id} contains a DOF outside the global matrix."
                )

            element_dofs[int(element_id)] = dofs
            row_blocks.append(np.repeat(dofs, dofs.size))
            col_blocks.append(np.tile(dofs, dofs.size))

        # Guarantee a diagonal slot for every global DOF.  This is useful for
        # zero-displacement row replacement, including temporarily inactive
        # nodes in staged construction.
        diagonal = np.arange(size, dtype=np.int64)
        row_blocks.append(diagonal)
        col_blocks.append(diagonal)

        rows = np.concatenate(row_blocks) if row_blocks else diagonal.copy()
        cols = np.concatenate(col_blocks) if col_blocks else diagonal.copy()
        values = np.ones(rows.size, dtype=np.int8)

        structure = coo_matrix(
            (values, (rows, cols)),
            shape=(size, size),
        ).tocsr()
        structure.sum_duplicates()
        structure.sort_indices()

        indptr = np.asarray(structure.indptr, dtype=np.int64)
        indices = np.asarray(structure.indices, dtype=np.int64)

        element_positions = {}
        for element_id, dofs in element_dofs.items():
            n = dofs.size
            positions = np.empty(n * n, dtype=np.int64)
            cursor = 0

            for row_dof in dofs:
                start = int(indptr[row_dof])
                end = int(indptr[row_dof + 1])
                row_columns = indices[start:end]
                local = np.searchsorted(row_columns, dofs)

                if (
                    np.any(local >= row_columns.size)
                    or np.any(row_columns[local] != dofs)
                ):
                    raise RuntimeError(
                        "Failed to map an element entry into the fixed CSR pattern."
                    )

                positions[cursor:cursor + n] = start + local
                cursor += n

            if np.unique(positions).size != positions.size:
                raise RuntimeError(
                    f"Element {element_id} has duplicate local-to-global CSR slots."
                )

            element_positions[element_id] = positions

        return cls(
            size=size,
            indptr=indptr,
            indices=indices,
            element_positions=element_positions,
        )

    def positions_for_element(self, element_id, dofs=None):
        element_id = int(element_id)
        try:
            positions = self.element_positions[element_id]
        except KeyError as error:
            raise KeyError(
                f"Element {element_id} is not present in the sparse assembly pattern."
            ) from error

        if dofs is not None:
            dofs = np.asarray(dofs, dtype=int).reshape(-1)
            expected_entries = dofs.size * dofs.size
            if positions.size != expected_entries:
                raise ValueError(
                    f"Element {element_id} DOF count does not match its fixed "
                    "CSR scatter map."
                )

        return positions


class GlobalMatrixBuilder:
    """Accumulate element matrices into dense or sparse global storage."""

    def __init__(
        self,
        size,
        matrix_storage="dense",
        profiler=None,
        sparse_pattern=None,
    ):
        self.size = int(size)
        if self.size < 1:
            raise ValueError("Global matrix size must be positive.")

        self.matrix_storage = normalize_matrix_storage(matrix_storage)
        self.profiler = profiler
        self.sparse_pattern = sparse_pattern

        if self.sparse_pattern is not None:
            if self.matrix_storage != "sparse":
                raise ValueError(
                    "A SparseAssemblyPattern can only be used with sparse storage."
                )
            if int(self.sparse_pattern.size) != self.size:
                raise ValueError(
                    "SparseAssemblyPattern size does not match the global matrix."
                )

        if self.matrix_storage == "dense":
            self._dense = np.zeros((self.size, self.size), dtype=float)
            self._rows = None
            self._cols = None
            self._data = None
        elif self.sparse_pattern is not None:
            self._dense = None
            self._rows = None
            self._cols = None
            self._data = np.zeros(self.sparse_pattern.nnz, dtype=float)
        else:
            self._dense = None
            self._rows = []
            self._cols = []
            self._data = []

    @property
    def uses_fixed_sparse_pattern(self):
        return self.matrix_storage == "sparse" and self.sparse_pattern is not None

    def add(self, dofs, element_matrix, element_id=None):
        dofs = np.asarray(dofs, dtype=int).reshape(-1)
        element_matrix = np.asarray(element_matrix, dtype=float)

        expected = (dofs.size, dofs.size)
        if element_matrix.shape != expected:
            raise ValueError(
                "Element matrix shape does not match element DOF count: "
                f"matrix={element_matrix.shape}, expected={expected}."
            )

        if np.any(dofs < 0) or np.any(dofs >= self.size):
            raise IndexError("Element DOF index lies outside the global matrix.")

        profile_start = perf_counter() if self.profiler is not None else None

        if self.matrix_storage == "dense":
            self._dense[np.ix_(dofs, dofs)] += element_matrix
            if self.profiler is not None:
                self.profiler.add_time("matrix_scatter", perf_counter() - profile_start)
                self.profiler.increment("matrix_scatter_calls")
            return

        if self.sparse_pattern is not None:
            if element_id is None:
                raise ValueError(
                    "Fixed sparse-pattern assembly requires the element ID."
                )
            positions = self.sparse_pattern.positions_for_element(
                element_id=element_id,
                dofs=dofs,
            )
            self._data[positions] += element_matrix.reshape(-1)

            if self.profiler is not None:
                self.profiler.add_time("matrix_scatter", perf_counter() - profile_start)
                self.profiler.increment("matrix_scatter_calls")
                self.profiler.increment("fixed_csr_scatter_calls")
            return

        # Backward-compatible dynamic sparse path.  COO -> CSR automatically
        # sums duplicates from neighbouring elements during finalization.
        rows = np.repeat(dofs, dofs.size)
        cols = np.tile(dofs, dofs.size)
        data = element_matrix.reshape(-1)

        self._rows.extend(rows.tolist())
        self._cols.extend(cols.tolist())
        self._data.extend(data.tolist())

        if self.profiler is not None:
            self.profiler.add_time("matrix_scatter", perf_counter() - profile_start)
            self.profiler.increment("matrix_scatter_calls")
            self.profiler.increment("dynamic_coo_scatter_calls")

    def finalize(self):
        profile_start = perf_counter() if self.profiler is not None else None

        if self.matrix_storage == "dense":
            matrix = self._dense
            if self.profiler is not None:
                self.profiler.add_time("matrix_finalize", perf_counter() - profile_start)
                self.profiler.increment("matrix_finalize_calls")
            return matrix

        try:
            from scipy.sparse import coo_matrix, csr_matrix
        except ImportError as error:
            raise ImportError(
                "Sparse matrix storage requires SciPy. Install scipy or "
                "use <MatrixStorage>dense</MatrixStorage>."
            ) from error

        if self.sparse_pattern is not None:
            # Reuse the immutable CSR structure.  Only the data array belongs
            # to this assembled matrix.  Explicit zeros are intentionally
            # retained so the structural pattern stays fixed across stages.
            matrix = csr_matrix(
                (
                    self._data,
                    self.sparse_pattern.indices,
                    self.sparse_pattern.indptr,
                ),
                shape=(self.size, self.size),
                copy=False,
            )
            matrix.has_sorted_indices = True
            matrix.has_canonical_format = True

            if self.profiler is not None:
                self.profiler.increment("fixed_csr_finalize_calls")
        else:
            matrix = coo_matrix(
                (self._data, (self._rows, self._cols)),
                shape=(self.size, self.size),
                dtype=float,
            ).tocsr()
            matrix.sum_duplicates()
            matrix.eliminate_zeros()

            if self.profiler is not None:
                self.profiler.increment("dynamic_coo_finalize_calls")

        if self.profiler is not None:
            self.profiler.add_time("matrix_finalize", perf_counter() - profile_start)
            self.profiler.increment("matrix_finalize_calls")

        return matrix
