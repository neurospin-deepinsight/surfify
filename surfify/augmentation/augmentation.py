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
import itertools
import torch
import numpy as np
from joblib import Memory
from surfify.utils import (
    neighbors, rotate_data, find_neighbors, find_rotation_interpol_coefs)
from surfify.utils.io import compute_and_store
from surfify.utils import get_logger, debug_msg, order_of_ico_from_vertices
from surfify.nn import IcoDiNeConv


logger = get_logger()


class SphericalRandomRotation(object):
    """ Rotation of the icosahedron's vertices.

    See Also
    --------
    surfify.utils.rotate_data

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
    def __init__(self, vertices, triangles, angles=(5, 0, 0), fixed_angle=True,
                 interpolation="barycentric", cachedir=None):
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
        fixed_angle: bool, default True
            if True changes the angle of the rotation at each call. This option
            slows down the training as the rotation needs to be initialiazed at
            each call
        interpolation: str, default 'barycentric'
            type of interpolation to use by the rotate_data function, see
            `rotate_data`.
        cachedir: str, default None
            set this folder to use smart caching speedup.
        """
        self.vertices = vertices
        self.triangles = triangles
        self.angles = [interval(val) for val in angles]
        self.fixed_angle = fixed_angle
        if fixed_angle:
            self.angles = [
                np.random.uniform(val[0], val[1]) for val in self.angles]
        self.interpolation = interpolation
        self.rotate_data_cached = compute_and_store(
            find_rotation_interpol_coefs, cachedir)(rotate_data)

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
        angles = self.angles
        if not self.fixed_angle:
            angles = [np.random.uniform(val[0], val[1]) for val in self.angles]
        return self.rotate_data_cached(
            data[np.newaxis, :], self.vertices, self.triangles,
            angles).squeeze()


class SphericalRandomCut(object):
    """ Random cut of patches on the icosahedron: use Direct Neighbors (DiNe)
    to build patches.

    See Also
    --------
    surfify.utils.neighbors

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
    def __init__(self, vertices, triangles, neighs=None, patch_size=3,
                 n_patches=1, replacement_value=0, random_size=True,
                 cachedir=None):
        """ Init class.

        Parameters
        ----------
        vertices: array (N, 3)
            icosahedron's vertices.
        triangles: array (M, 3)
            icosahdron's triangles.
        neighs: dict, default None
            optionnaly specify the undirected neighbors of each vertex as built
            with `sufify.utils.neighbors`, ie. a dictionary with vertices row
            index as keys and a dictionary of neighbors vertices row indexes
            organized by rings as values.
        patch_size: int, default 3
            the depth of the selected node's neighborhood to cut out.
        n_patches: int, default 1
            the number of patches to be considered.
        replacement_value: float, default 0
            the replacement patch value.
        cachedir: string or None, default None
            path to the storage directory, where to store heavy computation
            outputs.
        """
        memory = Memory(cachedir, verbose=0)
        neighbors_cached = memory.cache(neighbors)
        self.vertices = vertices
        self.triangles = triangles
        if neighs is None or type(neighs[0]) is not dict:
            self.neighs = neighbors_cached(
                vertices, triangles, depth=patch_size)
        else:
            self.neighs = neighs

        self.patch_size = patch_size
        self.n_patches = n_patches
        self.replacement_value = replacement_value
        self.random_size = random_size

    def __call__(self, data):
        """ Applies the cut out (ablation) augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        cut_data: array (N, )
            ablated input data.
        """
        n_dim = len(data.shape)
        if type(data) is torch.Tensor:
            data_cut = data.clone()
            if n_dim == 1:
                data_cut = data_cut.unsqueeze(0)
        else:
            data_cut = data.copy()
            if n_dim == 1:
                data_cut = data_cut[np.newaxis]
        logger.debug(data_cut.shape)
        for idx in range(self.n_patches):
            for channel_dim in range(data_cut.shape[0]):
                random_node = np.random.randint(len(self.vertices))
                patch_size = self.patch_size
                if self.random_size:
                    patch_indices = []
                    for neigh in self.neighs[random_node][1]:
                        patch_size = np.random.randint(self.patch_size)
                        for ring in range(1, patch_size):
                            patch_indices += self.neighs[neigh][ring]
                    patch_indices = list(set(patch_indices))
                else:
                    patch_indices = [random_node]
                    for ring in range(1, patch_size + 1):
                        patch_indices += self.neighs[random_node][ring]
                
                patch_indices = tuple([channel_dim, patch_indices])
                data_cut[patch_indices] = self.replacement_value
        return data_cut.squeeze()


class SphericalBlur(object):
    """ Gaussian blur implementation for textures on an icosahedron.
    It uses the DiNe convolution filters for its rapidity

    See Also
    --------
    surfify.utils.neighbors

    Examples
    --------
    >>> from surfify.utils import icosahedron
    >>> from surfify.datasets import make_classification
    >>> from surfify.augmentation import SphericalBlur
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> X, y = make_classification(ico3_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> processor = SphericalBlur(
            ico3_verts, ico3_tris, sigma=(0.1, 2))
    >>> y_cut = processor(y)
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, texture=y_cut)
    >>> plt.show()
    """
    def __init__(self, vertices, triangles, neighs=None, sigma=(0.1, 1),
                 fixed_sigma=False, cachedir=None):
        """ Init class.

        Parameters
        ----------
        vertices: array (N, 3)
            icosahedron's vertices.
        triangles: array (M, 3)
            icosahdron's triangles.
        neighs: dict, default None
            optionnaly specify the DiNe neighboors of each vertex as built
            with `sufify.utils.neighbors`, ie. a dictionary with vertices row
            index as keys and a list of properly ordered neighbors.
        sigma: float or tuple of floats, default (0.1, 1)
            sigma or range of the sigma parameter of the gaussian filter.
        fixed_sigma: bool, default False
            whether to randomly sample sigma in the range or use a constant
            value.
        cachedir: string or None, default None
            path to the storage directory, where to store heavy computation
            outputs
        """
        memory = Memory(cachedir, verbose=0)
        neighbors_cached = memory.cache(neighbors)
        self.vertices = vertices
        self.triangles = triangles
        assert type(sigma) in [list, tuple, int, float]
        if fixed_sigma and type(sigma) not in [int, float]:
            raise ValueError(
                "A fixed sigma implied sigma must be a signel number")
        elif type(sigma) in [list, tuple] and len(sigma) > 2:
            raise ValueError(
                "If sigma is an iterable, it must only contain the lower and "
                "upper bound of the interval in which sigma values will be "
                "sampled.")
        elif type(sigma) in [int, float]:
            sigma = (0.01, sigma)
        if any([sig <= 0 for sig in sigma]):
            raise ValueError("Sigma values must be positive.")
        self.sigma = list(sorted(sigma))
        max_sigma = self.sigma[1]
        if fixed_sigma:
            self.sigma = self.sigma[1]
        self.fixed_sigma = fixed_sigma
        depth = max(1, int(2 * max_sigma + 0.5))
        if neighs is None:
            neighs = neighbors_cached(
                vertices, triangles, depth=depth, direct_neighbor=True)

        if not type(neighs) is np.ndarray:
            neighs = np.asarray(list(neighs.values()))
        self.neighs = neighs
        self.positions = np.array([0] + list(itertools.chain(*[
            [ring] * (6 * ring) for ring in range(1, depth+1)])))
        self.conv = IcoDiNeConv(1, 1, self.neighs, bias=False)
        
        assert len(self.positions) == len(neighs[0])
        

    def __call__(self, data):
        """ Applies the augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        data_blur: array (N, )
            blurred output data.
        """
        n_dim = len(data.shape)
        to_numpy = False
        if type(data) is torch.Tensor:
            data_blur = data.clone().float()
        else:
            data_blur = torch.Tensor(data)
            to_numpy = True
        if n_dim == 1:
            data_blur = data_blur.unsqueeze(0)
        for channel_dim in range(data_blur.shape[0]):
            sigma = self.sigma
            if not self.fixed_sigma:
                sigma = np.random.rand() * (sigma[1] - sigma[0]) + sigma[0]
            gaussian_kernel = np.exp(-0.5 * (self.positions / sigma) ** 2)
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            with torch.no_grad():
                self.conv.weight.weight = torch.nn.Parameter(
                    torch.Tensor(gaussian_kernel), False)
            data_blur[channel_dim] = self.conv(
                data_blur[channel_dim][None, None])
        if to_numpy:
            data_blur = data_blur.detach().cpu().numpy()
        return data_blur.squeeze()


class SphericalNoise(object):
    """ Gaussian noise implementation for textures on an icosahedron.

    Examples
    --------
    >>> from surfify.utils import icosahedron
    >>> from surfify.datasets import make_classification
    >>> from surfify.augmentation import SphericalBlur
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> X, y = make_classification(ico3_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> processor = SphericalNoise(sigma=(0.1, 1))
    >>> y_cut = processor(y)
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, texture=y_cut)
    >>> plt.show()
    """
    def __init__(self, sigma=(0.1, 1), fixed_sigma=False):
        """ Init class.

        Parameters
        ----------
        sigma: float or tuple of floats, default (0.1, 1)
            sigma or range of the sigma parameter of the gaussian filter.
        fixed_sigma: bool, default False
            whether to randomly sample sigma in the range or use a constant
            value.
        """
        assert type(sigma) in [list, tuple, int, float]
        if fixed_sigma and type(sigma) not in [int, float]:
            raise ValueError(
                "A fixed sigma implied sigma must be a signel number")
        elif type(sigma) in [list, tuple] and len(sigma) > 2:
            raise ValueError(
                "If sigma is an iterable, it must only contain the lower and "
                "upper bound of the interval in which sigma values will be "
                "sampled.")
        elif type(sigma) in [int, float]:
            sigma = (0.01, sigma)
        if any([sig <= 0 for sig in sigma]):
            raise ValueError("Sigma values must be positive.")
        self.sigma = list(sorted(sigma))
        if fixed_sigma:
            self.sigma = self.sigma[1]
        self.fixed_sigma = fixed_sigma
        

    def __call__(self, data):
        """ Applies the augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        data_noise: array (N, )
            noisy output data.
        """
        n_dim = len(data.shape)
        to_numpy = False
        if type(data) is torch.Tensor:
            data_noise = data.clone().float()
        else:
            data_noise = torch.Tensor(data)
            to_numpy = True
        if n_dim == 1:
            data_noise = data_noise.unsqueeze(0)
        for channel_dim in range(data_noise.shape[0]):
            sigma = self.sigma
            if not self.fixed_sigma:
                sigma = np.random.rand() * (sigma[1] - sigma[0]) + sigma[0]
            noise = np.random.normal(0, sigma, data_noise[channel_dim].shape)
            data_noise[channel_dim] += noise
        if to_numpy:
            data_noise = data_noise.detach().cpu().numpy()
        return data_noise.squeeze()



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
