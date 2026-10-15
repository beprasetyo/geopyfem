import numpy as np

from initial_stress.k0 import (
    K0OverburdenIndex,
    _build_overburden_column_data,
    _vertical_total_stress_compression,
)


class _Material:
    def __init__(self, density):
        self.density = float(density)


def _synthetic_mesh():
    # Three unit rectangles. Elements 0 and 1 form a vertical soil column;
    # element 2 is horizontally remote and must never be tested for x=0.5.
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    elements = [
        {
            "id": 0,
            "type": "Quad4",
            "material": "lower",
            "connectivity": [0, 1, 2, 3],
        },
        {
            "id": 1,
            "type": "Quad4",
            "material": "upper",
            "connectivity": [3, 2, 4, 5],
        },
        {
            "id": 2,
            "type": "Quad4",
            "material": "remote",
            "connectivity": [6, 7, 8, 9],
        },
    ]
    materials = {
        "lower": _Material(1000.0),
        "upper": _Material(2000.0),
        "remote": _Material(3000.0),
    }
    return nodes, elements, materials


def _build_index(active_element_ids=None):
    nodes, elements, materials = _synthetic_mesh()
    return _build_overburden_column_data(
        nodes=nodes,
        mesh_elements=elements,
        material_models=materials,
        gravity={"enabled": True, "gx": 0.0, "gy": -10.0},
        units={"length": "m", "force": "kN"},
        active_element_ids=active_element_ids,
    )


def test_k0_overburden_index_matches_bruteforce_exactly():
    index = _build_index()
    assert isinstance(index, K0OverburdenIndex)
    brute_force_elements = list(index)

    points = [
        (0.50, 0.25),
        (0.50, 1.25),
        (0.00, 0.50),  # polygon boundary, tolerance-sensitive
        (2.50, 0.25),
        (4.00, 0.00),  # outside all elements
    ]

    for x, y in points:
        indexed = _vertical_total_stress_compression(
            x=x,
            y=y,
            column_elements=index,
        )
        brute = _vertical_total_stress_compression(
            x=x,
            y=y,
            column_elements=brute_force_elements,
        )
        assert indexed == brute


def test_k0_overburden_index_filters_noncontributing_elements():
    index = _build_index()

    # At x=0.5, y=0.25 both stacked elements can contribute; the remote
    # element is rejected by x-range filtering.
    sigma, candidates = _vertical_total_stress_compression(
        x=0.5,
        y=0.25,
        column_elements=index,
        return_candidate_count=True,
    )
    assert candidates == 2
    assert len(index) == 3

    # gamma_lower = 10 kN/m^3, gamma_upper = 20 kN/m^3.
    # Overburden above y=0.25 is 0.75 m lower + 1.0 m upper.
    assert np.isclose(sigma, 10.0 * 0.75 + 20.0 * 1.0)

    # At y=1.25 the lower element is also removed by the precomputed ymax
    # filter, leaving only the upper element.
    sigma, candidates = _vertical_total_stress_compression(
        x=0.5,
        y=1.25,
        column_elements=index,
        return_candidate_count=True,
    )
    assert candidates == 1
    assert np.isclose(sigma, 20.0 * 0.75)


def test_k0_overburden_index_respects_active_domain():
    index = _build_index(active_element_ids={1})
    assert len(index) == 1
    assert index[0]["element_id"] == 1

    sigma, candidates = _vertical_total_stress_compression(
        x=0.5,
        y=0.25,
        column_elements=index,
        return_candidate_count=True,
    )
    assert candidates == 1
    assert np.isclose(sigma, 20.0)
