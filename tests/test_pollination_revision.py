"""The revision of 2026: dependency ratios, the Maxent reader, the bioclimatic layers and the composite."""
import numpy as np
import polars as pl
import pytest

from esa_toolbox.core import Context, EsaError, Settings, check_method
from esa_toolbox.data.fetch import bioclim
from esa_toolbox.services.pollination import maxent, model, sdm

LAMBDAS = """a, 0.0, 0.0, 10.0
b, 0.5, 0.0, 4.0
'a, 2.0, 4.0, 10.0
`a, -1.0, 0.0, 5.0
linearPredictorNormalizer, 1.5
densityNormalizer, 20.0
numBackgroundPoints, 100
entropy, 3.0
"""


def test_cloglog_against_hand_calculation():
    m = maxent.parse(LAMBDAS, "toy")
    out = maxent.cloglog(m, {"a": np.array([7.0, 2.0, 99.0, np.nan]), "b": np.array([2.0, 4.0, -3.0, 1.0])})
    # a = 7, b = 2: linear b 0.5 * 2 / 4, hinge 2 * (7 - 4) / 6, reverse hinge 0 above 5
    s0 = 0.25 + 1.0
    # a = 2, b = 4: linear 0.5, hinge 0 below 4, reverse hinge -1 * (5 - 2) / 5
    s1 = 0.5 - 0.6
    # a = 99 is limited to 10 and b = -3 to 0: hinge 2 * 1, the others 0
    s2 = 2.0
    for got, s in zip(out, (s0, s1, s2)):
        assert np.isclose(got, 1 - np.exp(-np.exp(3.0) * np.exp(s - 1.5) / 20.0), rtol=1e-12)
    assert np.isnan(out[3])


def test_other_feature_kinds_stop():
    with pytest.raises(ValueError):
        maxent.parse("a, 0.0, 0.0, 1.0\na^2, 1.0, 0.0, 1.0\n" + LAMBDAS.split("\n", 4)[4], "toy")


def test_bundled_models():
    ms = maxent.read_zip(sdm.LAMBDAS)
    species = {m.name.rsplit("_", 1)[0] for m in ms}
    assert len(ms) == 235 and len(species) == 47
    layers = set(sdm.land_classes()[0]) | {"snd_km", "elmode", *sdm.CLIMATE}
    assert all(set(m.layers) == layers for m in ms)
    assert set(sdm.thresholds()) == species


def test_composite_counts_only_species_present():
    prob = {"x": np.array([0.6, 0.1, 0.05]), "y": np.array([0.2, 0.3, 0.05])}
    mean, n = sdm.composite(prob, {"x": 0.5, "y": 0.1})
    assert np.allclose(mean, [0.4, 0.3, 0.0]) and n.tolist() == [2, 1, 0]


def test_bioclim_definitions():
    months = np.arange(12.0)
    tx = (10 + 10 * np.sin((months - 3) / 12 * 2 * np.pi))[:, None, None]
    tn = tx - 8
    rr = np.full((12, 1, 1), 10.0)
    rr[[11, 0, 1]] = 50.0  # the wettest quarter runs from December to February
    b = bioclim(tx, tn, rr)[:, 0, 0]
    tm = (tx + tn)[:, 0, 0] / 2
    assert np.isclose(b[0], 100 * np.std(tm, ddof=1))
    assert np.isclose(b[1], tx.max())
    assert np.isclose(b[2], tm[[11, 0, 1]].mean())
    assert np.isclose(b[3], 100 * np.std(rr[:, 0, 0], ddof=1) / 20.0)


def test_dependency_ratios_follow_their_components():
    comp = model.crop_components().group_by("capri_code").agg(mean=pl.col("dependency").mean())
    t = model.crops().join(comp, left_on="capri_code", right_on="capri_code")
    assert t.height == 10
    assert (t["dependency"] - t["mean"]).abs().max() < 0.0005 + 1e-9  # rounded to a tenth of a percent
    assert t["dependency_class"].to_list() == [model.dependency_class(r) for r in t["dependency"]]


def test_original_method_keeps_klein():
    t = dict(model.crops("original").select("capri_code", "dependency").iter_rows())
    assert t == {"APPL": 0.65, "OFRU": 0.40, "RAPE": 0.25, "SUNF": 0.25, "SOYA": 0.25, "OOIL": 0.175, "PULS": 0.05,
                 "TEXT": 0.05, "TOMA": 0.05, "CITR": 0.05}


def test_original_method_has_its_own_folders(tmp_path):
    st = Settings.load(tmp_path / "none.toml", outputs=tmp_path / "o", work=tmp_path / "w", method="original")
    ctx = Context("pollination", st)
    assert ctx.out.name == "pollination_original" and ctx.work.name == "pollination_original"
    assert Context("pollination", Settings.load(tmp_path / "none.toml")).out.name == "pollination"
    check_method("pollination", st)
    with pytest.raises(EsaError):
        check_method("wood", st)
