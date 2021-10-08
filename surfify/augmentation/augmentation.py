# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Surface augmentation tools.
"""

# Imports
import numbers
import datetime
import numpy as np
from surfify.utils import neighbors, rotate_data, find_neighbors


class SphericalRandomRotation(object):
    """ Rotation of the icosahedron's vertices.

    See Also
    --------
    rotate_data

    Examples
    --------
    >>> from surfify.utils import icosahedron
    >>> from surfify.datasets import make_classification
    >>> from surfify.augmentation import SphericalRandomRotation
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> X, y = make_classification(ico3_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> processor = SphericalRandomRotation(
            ico3_verts, ico3_tris, angles=(45, 0, 0))
    >>> y_rot = processor(y)
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, texture=y_rot,
                     is_label=False)
    >>> plt.show()
    """
    def __init__(self, vertices, triangles, angles=(5, 0, 0)):
        """ Init class.

        Parameters
        ----------
        vertices: array (N, 3)
            icosahedron's vertices.
        triangles: array (M, 3)
            icosahdron's triangles.
        angles: 3-uplet, default (5, 0, 0)
            the rotation angles intervals in degrees for each axis (Euler
            representation).
        """
        self.vertices = vertices
        self.triangles = triangles
        self.angles = [interval(val) for val in angles]

    def __call__(self, data):
        """ Rotates the provided vertices and projects the input data
        accordingly.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        rot_data: arr (N, )
            rotated input data.
        """
        np.random.seed(datetime.datetime.now().second +
                       datetime.datetime.now().microsecond)
        angles = [np.random.uniform(val[0], val[1]) for val in self.angles]
        data = data.reshape(1, -1, 1)
        return rotate_data(data, self.vertices, self.triangles,
                           angles).squeeze()


class SphericalRandomCut(object):
    """ Random cut of patches on the icosahedron: use Direct Neighbors (DiNe)
    to build patches.

    See Also
    --------
    neighbors

    Examples
    --------
    >>> from surfify.utils import icosahedron
    >>> from surfify.datasets import make_classification
    >>> from surfify.augmentation import SphericalRandomCut
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> X, y = make_classification(ico3_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> processor = SphericalRandomCut(
            ico3_verts, ico3_tris, replacement_value=5)
    >>> y_cut = processor(y)
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, texture=y_cut,
                     is_label=True)
    >>> plt.show()
    """
    def __init__(self, vertices, triangles, neighs=None, n_rings=3,
                 n_patches=1, replacement_value=0):
        """ Init class.

        Parameters
        ----------
        vertices: array (N, 3)
            icosahedron's vertices.
        triangles: array (M, 3)
            icosahdron's triangles.
        neighs: dict, default None
            optionnaly specify the DiNe neighboors of each vertex as build
            with `sufify.utils.neighbors`, ie. a dictionary with vertices row
            index as keys and a dictionary of neighbors vertices row indexes
            organized by rings as values.
        n_rings: int, default 3
            the number of neighboring rings from one node to be considered
            during the ablation.
        n_patches: int, default 1
            the number of patches to be considered.
        replacement_value: float, default 0
            the replacement patch value.
        """
        self.vertices = vertices
        self.triangles = triangles
        if neighs is None:
            self.neighs = neighbors(vertices, triangles, direct_neighbor=True)
        else:
            self.neighs = neighs
        self.n_rings = n_rings
        self.n_patches = n_patches
        self.replacement_value = replacement_value

    def __call__(self, data):
        """ Applies the cut out (ablation) augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        cut_data: arr (N, )
            ablated input data.
        """
        data_cut = data.copy()
        for idx in range(self.n_patches):
            random_node = np.random.randint(0, len(self.vertices))
            patch_indices = find_neighbors(
                random_node, self.n_rings, self.neighs)
            data_cut[patch_indices] = self.replacement_value
        return data_cut


def interval(bound, lower=None):
    """ Create an interval.

    Parameters
    ----------
    bound: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(bound, numbers.Number):
        if bound < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -bound
        return (lower, bound)
    if len(bound) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = bound
    if min_val > max_val:
        raise ValueError("Wrong interval boundaries.")
    return tuple(bound)
