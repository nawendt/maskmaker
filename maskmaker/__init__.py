"""
Create masks for grids using shapefiles
"""

import fiona
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon, shape
from skimage import draw

__all__ = ['Mask']

__author__ = 'Nathan Wendt'
__date__ = '07/10/2017'
__version__ = '0.1.0'
__maintainer__ = 'Nathan Wendt'
__email__ = 'nawendt@ou.edu'
__status__ = 'Development'


class Mask(object):

    def __init__(self, x, y, shapefile, dx=np.inf, centered=True):
        """
        Create a ``Mask`` instance from which you can grid a shapefile and use it as a
        mask for a field.

        Parameters
        ----------
        x: array_like
            x coordinates
        y: array_like
            y coordinates
        shapefile: str
            Path to shapefile
        dx: float
            Maximum searh distance for KDTree. Default is infinite. Knowing this value
            will help performance.
        centered: bool
            Flag for whether the coordinates are at the grid center. Default is True.
        """

        self.x = x
        self.y = y
        self.shapefile = fiona.open(shapefile)
        self.dx = dx
        self.centered = centered
        self.mask = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.shapefile.closed:
            self.shapefile.close()

    def _make_tree(self):
        """
        Create KDTree.
        """

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
        tpoints = np.asarray(list(zip(tx.ravel(), ty.ravel())))

        self.tree = cKDTree(tpoints)

    def make(self):
        """
        Create a mask within given grid based on given shapefile.
        """

        self._make_tree()
        self.mask = np.zeros(self.x.shape)

        # Get coordinates from shapefile for each polygon
        for shp in self.shapefile:
            poly = shape(shp['geometry'])

            if isinstance(poly, Polygon):
                xx, yy, coords = self._gridify(poly)
                self._update_mask(xx, yy, coords)
            elif isinstance(poly, MultiPolygon):
                # Loop over each piece of the polygon and add it to mask
                for part in poly:
                    xx, yy, coords = self._gridify(part)
                    self._update_mask(xx, yy, coords)
            else:
                raise TypeError('Unknown shape type {}'.format(type(poly).__name__))

        # Make sure double-counted points are handled
        self.mask[self.mask > 1] = 1

    def _gridify(self, polygon):
        """
        Grid a polygon using a KDTree.

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
            points = np.asarray(list(zip(px, py)))
        except TypeError:
            points = np.asarray(list(zip([px], [py])))
        _, inds = self.tree.query(points, k=1, distance_upper_bound=self.dx)
        # Remove points outside the destination grid
        bad_inds = np.where(inds >= len(self.tree.data))[0]
        inds = np.delete(inds, bad_inds)

        coords = np.unravel_index(inds, self.x.shape)
        x, y = draw.polygon(*coords)

        return x, y, coords

    def _update_mask(self, x, y, coords):
        """
        Update the mask at point(s) (x, y) in place.

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
            for pt in zip(*coords):
                self.mask[pt] += 1
        else:
            self.mask[x, y] += 1
