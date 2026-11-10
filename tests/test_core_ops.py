"""Raster operations, against hand calculations."""
import numpy as np
import pytest

from esa_toolbox import rio
from esa_toolbox.core import markdown_table
from esa_toolbox.grid import Grid


def test_block_sum_and_mean_keep_no_data():
    a = np.array([[1, np.nan, 2, 2], [np.nan, np.nan, 2, 2]], dtype="float32")
    a = np.vstack([a, np.full((2, 4), np.nan)])
    s = rio.block_sum(a, 2)
    m = rio.block_mean(a, 2)
    assert s[0, 0] == 1 and s[0, 1] == 8 and np.isnan(s[1, 0])
    assert m[0, 0] == 1 and m[0, 1] == 2 and np.isnan(m[1, 1])


def test_partial_blocks_are_padded():
    a = np.ones((3, 5))
    assert rio.block_sum(a, 2).shape == (2, 3)
    assert rio.block_sum(a, 2)[1, 2] == 1


def test_zonal_sum_ignores_outside_and_nan():
    z = np.array([[1, 1, 2], [0, 2, 2]])
    v = np.array([[1.0, np.nan, 3], [100, 4, 5]])
    assert list(rio.zonal_sum(v, z, 2)) == [1.0, 12.0]


def test_read_pads_outside_the_file(tmp_path):
    g = Grid(0, 0, 400, 400, 100)
    rio.write(tmp_path / "a.tif", np.arange(16, dtype="uint16").reshape(4, 4), g, nodata=999)
    out = rio.read(tmp_path / "a.tif", Grid(-100, 300, 100, 500, 100))
    assert out.tolist() == [[999, 999], [999, 0]]
    f = rio.read(tmp_path / "a.tif", Grid(-100, 300, 100, 500, 100), as_float=True)
    assert np.isnan(f[0, 0]) and f[1, 1] == 0


def test_read_refuses_misaligned_grid(tmp_path):
    g = Grid(0, 0, 400, 400, 100)
    rio.write(tmp_path / "a.tif", np.zeros((4, 4), "uint8"), g, nodata=255)
    with pytest.raises(ValueError):
        rio.read(tmp_path / "a.tif", Grid(50, 0, 250, 200, 100))


def test_tiles_cover_the_grid_once():
    g = Grid(0, 0, 1050, 700, 100)
    cells = sum(t.width * t.height for t in g.tiles(4))
    assert cells == g.width * g.height


def test_markdown_table_formats_numbers():
    import polars as pl

    md = markdown_table(pl.DataFrame({"Year": [2018], "Value": [12345.678]}), 1)
    assert "| 2018 | 12,345.7 |" in md


def test_year_keys_follow_inputs(tmp_path):
    import polars as pl

    from esa_toolbox.core import Context, Settings, work_key

    f = tmp_path / "input.tif"
    f.write_bytes(b"a")
    ctx = Context("wood", Settings.load(path=tmp_path / "none.toml", work=tmp_path / "work", workers=1))
    k = work_key(rio, 2018, f, np.arange(3), pl.DataFrame({"x": [1.0]}))
    out = tmp_path / "map.tif"
    out.write_bytes(b"m")
    assert not ctx.year_done("maps", 2018, k, out)
    ctx.mark_year_done("maps", 2018, k)
    assert ctx.year_done("maps", 2018, k, out)
    assert not ctx.year_done("maps", 2018, k, tmp_path / "missing.tif")
    assert work_key(rio, 2018, f, np.arange(3), pl.DataFrame({"x": [2.0]})) != k
    f.write_bytes(b"bb")
    assert work_key(rio, 2018, f, np.arange(3), pl.DataFrame({"x": [1.0]})) != k


def test_mosaic_of_plain_sources(tmp_path):
    g = Grid(0, 0, 400, 200, 100)
    left, right = Grid(0, 0, 200, 200, 100), Grid(200, 0, 400, 200, 100)
    a = np.array([[1, 2], [3, 65535]], dtype="uint16")
    rio.write(tmp_path / "l.tif", a, left, nodata=65535)
    vrt = rio.build_vrt([tmp_path / "l.tif"], g, tmp_path / "m.vrt", dtype="uint16", nodata=65535)
    rio.to_cog(vrt, tmp_path / "m.tif", predictor=2, threads=1)
    import rasterio

    with rasterio.open(tmp_path / "m.tif") as s:
        assert s.overviews(1) == []
        b = s.read(1)
    assert (b[:, :2] == a).all() and (b[:, 2:] == 65535).all()
    assert right.width == 2
