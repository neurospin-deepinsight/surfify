# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Augmentations are directly inspired by natural image transformations used to
train CNNs.
"""

# Imports
import numbers
import datetime
import numpy as np
from surfify.utils import (
    neighbors, rotate_data, find_neighbors, find_rotation_interpol_coefs)
from surfify.utils.io import compute_and_store
from .utils import RandomAugmentation


class SurfCutOut(RandomAugmentation):
    """ Starting from random vertices, the SurfCutOut sets an adaptive connex
    neighborhood to zero.

    See Also
    --------
    surfify.utils.neighbors
    """
    def __init__(self, vertices, triangles, neighs=None, patch_size=3,
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
        patch_size: int, default 3
            the number of neighboring rings from one node to be considered
            during the ablation.
        n_patches: int, default 1
            the number of patches to be considered.
        replacement_value: float, default 0
            the replacement patch value.
        """
        super().__init__()
        self.vertices = vertices
        self.triangles = triangles
        if neighs is None:
            self.neighs = neighbors(vertices, triangles, direct_neighbor=True)
        else:
            self.neighs = neighs
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.replacement_value = replacement_value

    def run(self, data):
        """ Applies the cut out (ablation) augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        _data: arr (N, )
            ablated input data.
        """
        _data = data.copy()
        for idx in range(self.n_patches):
            random_node = np.random.randint(0, len(self.vertices))
            patch_indices = find_neighbors(
                random_node, self.patch_size, self.neighs)
            _data[patch_indices] = self.replacement_value
        return _data


class SurfNoise(RandomAugmentation):
    """ The SurfNoise adds a Gaussian white noise with standard deviation
    sigma.
    """
    def __init__(self, sigma):
        """ Init class.

        Parameters
        ----------
        sigma: float
            the noise standard deviation.
        """
        super().__init__()
        self.sigma = sigma

    def run(self, data):
        """ Applies the noising augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        _data: arr (N, )
            noised input data.
        """
        _data = data.copy()
        _data += np.random.normal(0, self.sigma, len(_data))
        return _data


class SurfRotation(RandomAugmentation):
    """ The SurfRotation rotate the cortical measures.

    See Also
    --------
    surfify.utils.rotate_data
    """
    def __init__(self, vertices, triangles, phi=5, theta=0, psi=0,
                 interpolation="barycentric", cachedir=None):
        """ Init class.

        Parameters
        ----------
        vertices: array (N, 3)
            icosahedron's vertices.
        triangles: array (M, 3)
            icosahdron's triangles.
        phi: float, default 5
            the rotation phi angle in degrees: Euler representation.
        theta: float, default 0
            the rotation theta angle in degrees: Euler representation.
        psi: float, default 0
            the rotation psi angle in degrees: Euler representation.
        interpolation: str, default 'barycentric'
            type of interpolation to use by the rotate_data function, see
            `rotate_data`.
        cachedir: str, default None
            set this folder to use smart caching speedup.
        """
        super().__init__()
        self.vertices = vertices
        self.triangles = triangles
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.interpolation = interpolation
        self.rotate_data_cached = compute_and_store(
            find_rotation_interpol_coefs, cachedir)(rotate_data)

    def run(self, data):
        """ Rotates the provided vertices and projects the input data
        accordingly.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        _data: arr (N, )
            rotated input data.
        """
        return self.rotate_data_cached(
            data[np.newaxis, :, np.newaxis], self.vertices, self.triangles,
            [self.phi, self.theta, self.psi]).squeeze()
