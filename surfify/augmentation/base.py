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
from joblib import Memory
import itertools
import numpy as np
import torch
from surfify.utils import (
    neighbors, rotate_data, find_rotation_interpol_coefs)
from surfify.nn import IcoDiNeConv
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
                 n_patches=1, sigma=0, replacement_value=0, cachedir=None):
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
        sigma: int, default 0
            use different patch size in [patch_size-sigma, patch_size+sigma]
            for each cutout.
        replacement_value: float, default 0
            the replacement patch value.
        cachedir: str, default None
            the optional path to cache the neighbors function output.
        """
        super().__init__()
        memory = Memory(cachedir, verbose=0)
        neighbors_cached = memory.cache(neighbors)
        self.vertices = vertices
        self.triangles = triangles
        max_depth = patch_size
        if isinstance(patch_size, RandomAugmentation.Interval):
            max_depth = patch_size.high
        if neighs is None or type(neighs[0]) is not dict:
            self.neighs = neighbors_cached(
                vertices, triangles, depth=max_depth)
        else:
            self.neighs = neighs
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.sigma = sigma
        self.replacement_value = replacement_value

    def run(self, data):
        """ Applies the cut out (ablation) augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        data: arr (N, )
            ablated input data.
        """
        for _ in range(self.n_patches):
            random_node = np.random.randint(len(self.vertices))
            
            # If sigma is not null, patch size can vary around its current
            # value by +- sigma for each patch, but staying in
            # [0, max_patch_size]
            max_patch_size = self.patch_size
            if "patch_size" in self.intervals.keys():
                max_patch_size = self.intervals["patch_size"].high
            random_size = np.random.randint(
                max(self.patch_size - self.sigma, 0),
                min(self.patch_size + self.sigma + 1, max_patch_size))
            # for each neighbor, so in each direction, we seek neighbors
            # up to a different random order, creating irregular patches
            # in different directions
            patch_indices = []
            for neigh in self.neighs[random_node][1]:
                _size = np.random.randint(random_size)
                for ring in range(1, _size + 1):
                    patch_indices += self.neighs[neigh][ring]
            patch_indices = list(set(patch_indices))
            data[patch_indices] = self.replacement_value
        return data


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
        data: arr (N, )
            noised input data.
        """
        data += np.random.normal(0, self.sigma, len(data))
        return data


class SurfBlur(RandomAugmentation):
    """ An icosahedron texture Gaussian blur implementation. It uses the DiNe
    convolution filter for speed. The receptive field is controlled by sigma,
    the standard deviation of the kernel.

    See Also
    --------
    surfify.utils.neighbors
    surfify.nn.modules.IcoDiNeConv
    """
    def __init__(self, vertices, triangles, sigma, neighs=None, cachedir=None):
        """ Init class.
        Parameters
        ----------
        vertices: array (N, 3)
            icosahedron's vertices.
        triangles: array (M, 3)
            icosahdron's triangles.
        sigma: float
            sigma parameter of the gaussian filter.
        neighs: dict, default None
            optionnaly specify the DiNe neighboors of each vertex as build
            with `sufify.utils.neighbors`, ie. a dictionary with vertices row
            index as keys and a dictionary of neighbors vertices row indexes
            organized by rings as values.
        cachedir: str, default None
            the optional path to cache the neighbors function output.
        """
        super().__init__()
        memory = Memory(cachedir, verbose=0)
        neighbors_cached = memory.cache(neighbors)
        self.vertices = vertices
        self.triangles = triangles
        max_sigma = sigma
        if isinstance(sigma, RandomAugmentation.Interval):
            max_sigma = sigma.high
        depth = max(1, int(2 * max_sigma + 0.5))
        if neighs is None:
            self.neighs = neighbors_cached(vertices, triangles, depth=depth,
                                           direct_neighbor=True)
        else:
            self.neighs = neighs
        self.neighs = np.asarray(list(self.neighs.values()))
        self.positions = np.array([0] + list(itertools.chain(*[
            [ring] * (6 * ring) for ring in range(1, depth + 1)])))
        assert len(self.positions) == len(self.neighs[0])
        self.conv = IcoDiNeConv(1, 1, self.neighs, bias=False)

    def run(self, data):
        """ Applies the augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        data: array (N, )
            blurred output data.
        """
        gaussian_kernel = np.exp(-0.5 * (self.positions / self.sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        with torch.no_grad():
            self.conv.weight.weight = torch.nn.Parameter(
                torch.Tensor(gaussian_kernel), False)
        data = self.conv(torch.from_numpy(data[None, None]).float())
        return data.numpy().squeeze()


class SurfRotation(RandomAugmentation):
    """ The SurfRotation rotate the cortical measures.

    See Also
    --------
    surfify.utils.rotate_data
    """
    def __init__(self, vertices, triangles, phi=5, theta=0, psi=0,
                 interpolation="barycentric", cachedir=None,
                 *args, **kwargs):
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
        super().__init__(*args, **kwargs)
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
        data: arr (N, )
            rotated input data.
        """
        return self.rotate_data_cached(
            data[np.newaxis, :, np.newaxis], self.vertices, self.triangles,
            [self.phi, self.theta, self.psi], self.interpolation).squeeze()