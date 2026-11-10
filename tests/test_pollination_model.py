"""Equations of the pollination model, against hand calculations and properties."""
import numpy as np

from esa_toolbox.services.pollination import model


def test_forest_core_needs_eight_forest_neighbours():
    clc = np.full((5, 5), 311)
    clc[0, 0] = 211
    core = model.forest_core(clc)
    assert core[2, 2] and core[2, 3] and not core[1, 1]  # (1, 1) touches the cropland cell
    assert not core[0].any() and not core[:, 0].any()  # beyond the block counts as non-forest


def test_decay_at_zero_and_limit():
    clc = np.full((1, 3), 211)
    s = model.expert_model(clc, np.zeros((1, 3)), np.array([[0.0, 20000.0, np.nan]], "float32"),
                           np.full((1, 3), 100.0, "float32"))
    base = model.scores().suitability[211]
    assert np.isclose(s[0, 0], base)
    assert np.isclose(s[0, 1], base * np.exp(-0.00046 * 20000), rtol=1e-6)
    assert s[0, 2] == 0


def test_roads_zero_and_unscored_stays_nan():
    clc = np.array([[211, 9999]])
    s = model.expert_model(clc, np.array([[1, 0]]), np.zeros((1, 2), "float32"), np.full((1, 2), 50.0, "float32"))
    assert s[0, 0] == 0 and np.isnan(s[0, 1])


def test_exact_distance():
    clc = np.full((5, 5), 211)
    clc[0, 0] = 311
    d = model.exact_distance(clc)
    assert d[0, 0] == 0 and np.isclose(d[3, 4], 500.0)


def test_classes_of_potential():
    c = model.potential_classes(np.array([0.05, 0.1, 0.19999, 0.2, 0.3, np.nan]))
    assert c.tolist() == [0, 1, 1, 2, 3, 255]


def test_allocation_keeps_each_km_total():
    rng = np.random.default_rng(0)
    w = rng.random((30, 30))
    w[:10, :10] = 0  # a kilometre with no weight spreads evenly over land
    land = np.ones((30, 30))
    land[20:, 20:] = np.nan
    w[np.isnan(land)] = np.nan
    v = rng.random((3, 3)) * 100
    out = model.allocate(v, w, land)
    from esa_toolbox.rio import block_sum

    s = block_sum(out, 10)
    assert np.allclose(s[:2, :], v[:2, :]) and np.isclose(s[0, 0], v[0, 0])
    assert np.isclose(out[0, 0], v[0, 0] / 100)
    assert np.isnan(s[2, 2])
    assert np.allclose(model.spread_1km(v, model.allocation_share(w, land)), out, equal_nan=True)
