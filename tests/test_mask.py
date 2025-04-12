# Copyright (c) 2025 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
"""Mask testing module."""

from pathlib import Path

import numpy as np

from maskmaker.mask import Mask


def test_mask():
    """Test CONUS mask on HRRR grid."""
    ref = Path(__file__).parent / 'data' / 'reference_conus_mask.npz'
    shp = Path(__file__).parent / 'data' / 'conus.shp'
    grid = Path(__file__).parent / 'data' / 'hrrr_grid.npz'

    with np.load(grid) as _grid:
        lat = _grid['lat']
        lon = _grid['lon']

    with np.load(ref) as _ref:
        ref_mask = _ref['mask']

    test_obj = Mask(lon, lat, shp)
    test_obj.make()

    np.testing.assert_equal(test_obj.mask, ref_mask)
