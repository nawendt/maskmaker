# Copyright (c) 2025 Nathan Wendt.
# Distributed under the terms of the BSD 3-Clause License.
"""Mask module."""

import fiona
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import shape
from skimage import draw


class Mask:
    """Mask object."""

    def __init__(self, x, y, shapefile, dx=np.inf, centered=True):
        """Create a mask object.

        Parameters
        ----------
        x: array_like
            x coordinates

        y: array_like
            y coordinates

        shapefile: str or `pathlib.Path`
            Path to shapefile

        dx: float
            Maximum searh distance for KDTree. Default is infinite. Knowing this value
            will help performance.

        centered: bool
            Flag for whether the coordinates are at the grid center. Default is True.
        """
        self.x = x
        self.y = y
        self.dx = dx
        self.centered = centered
        self.mask = None
        self.tree = None

        self._features = []
        with fiona.open(shapefile) as _shape:
            for feature in _shape:
                self._features.append(feature)

    def _make_tree(self):
        """Create KDTree."""
        tx = np.array(self.x, copy=True, subok=True)
        ty = np.array(self.y, copy=True, subok=True)
        if not self.centered:
            x = tx.copy()
            y = ty.copy()
            x[:, :-1] = tx[:, :-1] + (tx[:, 1:] - tx[:, :-1]) / 2.
            x[:, -1] += (tx[:, -1] - tx[:, -2]) / 2.
            y[:-1, :] = ty[:-1, :] + (ty[1:, :] - ty[:-1, :]) / 2.
            y[-1, :] += (ty[-1, :] - ty[-2, :]) / 2.
            tx = x
            ty = y
            del x
            del y
        tpoints = np.asarray(list(zip(tx.ravel(), ty.ravel(), strict=True)))

        self.tree = KDTree(tpoints)

    def make(self):
        """Create a mask within given grid based on given shapefile."""
        if self.tree is not None:
            raise ValueError('Tree already exists!')

        self._make_tree()
        self.mask = np.zeros(self.x.shape)

        # Get coordinates from shapefile for each polygon
        for feature in self._features:
            geom = shape(feature['geometry'])

            if geom.geom_type == 'Polygon':
                xx, yy, coords = self._gridify(geom)
                self._update_mask(xx, yy, coords)
            elif geom.geom_type == 'MultiPolygon':
                # Loop over each piece of the polygon and add it to mask
                for part in geom.geoms:
                    xx, yy, coords = self._gridify(part)
                    self._update_mask(xx, yy, coords)
            else:
                raise NotImplementedError(f'Cannot process {geom.__qualname__} geometry.')

        # Make sure double-counted points are handled
        self.mask[self.mask > 1] = 1

    def _gridify(self, polygon):
        """Grid a polygon using a KDTree.

        Parameters
        ----------
        polygon: Polygon or MultiPolygon
            Polygon to grid

        Returns
        -------
        x: array_like
            x coordinates of filled polygon

        y: array_like
            y coordinates of filled polygon

        coords: list of arrays
            coordinate arrays of polygon exterior
        """
        px, py = polygon.exterior.xy
        try:
            points = np.asarray(list(zip(px, py, strict=True)))
        except TypeError:
            points = np.asarray(list(zip([px], [py], strict=True)))
        _, inds = self.tree.query(points, k=1, distance_upper_bound=self.dx)
        # Remove points outside the destination grid
        bad_inds = np.where(inds >= len(self.tree.data))[0]
        inds = np.delete(inds, bad_inds)

        coords = np.unravel_index(inds, self.x.shape)
        x, y = draw.polygon(*coords)

        return x, y, coords

    def _update_mask(self, x, y, coords):
        """Update the mask at point(s) (x, y) in place.

        Parameters
        ----------
        x: array_like
            x coordinates of filled polygon

        y: array_like
            y coordinates of filled polygon

        coords: list of arrays
            coordinate arrays of polygon exterior
        """
        if not x.shape:
            # Handle situation where a polygon cannot be drawn.
            # This will be more likely when the destination grid
            # is coarse enough that small islands get placed in the
            # same grid cell, or similar situation.
            for pt in zip(*coords, strict=True):
                self.mask[pt] += 1
        else:
            self.mask[x, y] += 1
