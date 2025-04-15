# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Coordinate system tools.
"""

# Imports
import math
import warnings
import itertools
import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.spatial import transform


def cart2sph(x, y, z):
    """ Cartesian to spherical coordinate transform.

    See Also
    --------
    sph2cart, text2grid, grid2text

    Parameters
    ----------
    x: float or array_like.
        x-component of Cartesian coordinates
    y: float or array_like.
        y-component of Cartesian coordinates
    z: float or array_like.
        z-component of Cartesian coordinates

    Returns
    -------
    alpha: float or `numpy.ndarray`
        Azimuth angle in radiants. The value of the angle is in the range
        [-pi pi].
    beta: float or `numpy.ndarray`
        Elevation angle in radiants. The value of the angle is in the range
        [-pi/2, pi/2].
    r: float or `numpy.ndarray`
        Radius.
    """
    alpha = np.arctan2(y, x)
    beta = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return alpha, beta, r


def sph2cart(alpha, beta, r):
    """ Spherical to cartesian coordinate transform.

    See Also
    --------
    cart2sph, text2grid, grid2text

    Parameters
    ----------
    alpha: float or array_like
        Azimuth angle in radiants.
    beta: float or array_like
        Elevation angle in radiants.
    r: float or array_like
        Radius.

    Returns
    -------
    x: float or `numpy.ndarray`
        x-component of Cartesian coordinates
    y: float or `numpy.ndarray`
        y-component of Cartesian coordinates
    z: float or `numpy.ndarray`
        z-component of Cartesian coordinates
    """
    x = r * np.cos(alpha) * np.cos(beta)
    y = r * np.sin(alpha) * np.cos(beta)
    z = r * np.sin(beta)
    return x, y, z


def text2grid(vertices, texture, resx=192, resy=192):
    """ Convert a texture onto a spherical surface into an image by evenly
    resampling the spherical surface with respect to sin(e) and a, where e
    and a are elevation and azimuth, respectively. Nearest-neighbor
    interpolation is used to convert data from the 3-D surface to the
    2-D grid.

    See Also
    --------
    grid2text

    Examples
    --------
    >>> from surfify.utils import icosahedron, text2grid
    >>> from surfify.datasets import make_classification
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> X, y = make_classification(ico2_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> y_grid = text2grid(ico2_verts, y)
    >>> plt.imshow(y_grid)
    >>> plt.show()

    Parameters
    ----------
    vertices: array (N, 3)
        x, y, z coordinates of an icosahedron.
    texture: array (N, )
        the input icosahedron texture.
    resx: int, default 192
        the generated image number of samples in the x direction.
    resy: int, default 192
        the generated image number of samples in the y direction.

    Returns
    -------
    proj: array (resx, resy)
        the projecteed texture.
    """
    azimuth, elevation, radius = cart2sph(*vertices.T)
    points = np.stack((azimuth, np.sin(elevation))).T
    grid_x, grid_y = np.mgrid[-np.pi:np.pi:resx * 1j, -1:1:resy * 1j]
    return griddata(points, texture, (grid_x, grid_y), method="nearest")


def grid2text(vertices, proj):
    """ Convert a grided-texture into a spherical surface. Nearest-neighbor
    interpolation is used to convert data from the 2-D grid to the
    3-D surface.

    See Also
    --------
    text2grid

    Examples
    --------
    >>> from surfify.utils import icosahedron, grid2text
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> y_grid = np.zeros((192, 192), dtype=int)
    >>> y_grid[:, :96] = 1
    >>> y = grid2text(ico2_verts, y_grid)
    >>> plot_trisurf(ico2_verts, triangles=ico2_tris, texture=y,
                     is_label=True)
    >>> plt.show()

    Parameters
    ----------
    vertices: array (N, 3)
        x, y, z coordinates of an icosahedron.
    proj: array (resx, resy)
        the grided-texture.

    Returns
    -------
    texture: array (N, )
        the input icosahedron texture.
    """
    azimuth, elevation, radius = cart2sph(*vertices.T)
    points = np.stack((azimuth, np.sin(elevation))).T
    resx, resy = proj.shape
    grid_x, grid_y = np.mgrid[-np.pi:np.pi:resx * 1j, -1:1:resy * 1j]
    grid_points = np.stack((grid_x.flatten(), grid_y.flatten())).T
    proj_values = proj.flatten()
    interp = NearestNDInterpolator(grid_points, proj_values)
    return interp(points)


def ico2ico(vertices, ref_vertices):
    """ Find a mapping between two icosahedrons: a simple rotation is
    estimated by identifying 4 vertices with same coordinates up to their signs
    and then finding the best rotation using permutations.

    See Also
    --------
    text2ico

    Examples
    --------
    >>> from surfify.utils import icosahedron, ico2ico
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> ico2_std_verts, ico2_std_tris = icosahedron(order=2, standard_ico=True)
    >>> rotation = ico2ico(ico2_verts, ico2_std_verts)
    >>> fig, ax = plt.subplots(1, 1, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    >>> plot_trisurf(ico2_std_verts, triangles=ico2_std_tris, colorbar=False,
                     fig=fig, ax=ax, alpha=0.3, edgecolors="blue")
    >>> plot_trisurf(rotation.apply(ico2_verts), triangles=ico2_tris,
                     colorbar=False, fig=fig, ax=ax, alpha=0.3,
                     edgecolors="green")
    >>> plt.show()

    Parameters
    ----------
    vertices: array (N, 3)
        the vertices to project.
    ref_vertices: array (N, 3)
        the reference/target vertices.

    Returns
    -------
    rotation: scipy.spatial.tranform.Rotation
        the rotation that transforms the vertices to the reference.
    """
    if len(vertices) != len(ref_vertices):
        raise ValueError("Input vertices must be of the same order.")

    vertices_of_interest = []
    for _vertices in (vertices, ref_vertices):
        for idx in range(len(_vertices)):
            coords_of_interest = _vertices[idx]
            idx_of_interest = (
                np.abs(_vertices) == np.abs(coords_of_interest)).all(axis=1)
            if idx_of_interest.sum() == 4:
                vertices_of_interest.append(_vertices[idx_of_interest])
                break

    permutations = itertools.permutations(range(4))
    n_permutations = math.factorial(4)
    rmse = 1000
    it = 0
    best_rmse = rmse
    best_rotation = None
    while rmse > 0 and it < n_permutations:
        it += 1
        order = np.array(next(permutations))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            rotation, rmse = transform.Rotation.align_vectors(
                vertices_of_interest[1], vertices_of_interest[0][order])
        if rmse < best_rmse:
            best_rmse = rmse
            best_rotation = rotation

    if it == n_permutations and best_rmse > 0:
        warnings.warn(
            "A proper mapping between the two icosahedrons could not be "
            "found. The closest rotation has a rmse of {0}.".format(rmse))

    return best_rotation


def text2ico(texture, vertices, ref_vertices, atol=1e-4):
    """ Projects a texture associated to an icosahedron onto an other one.

    See Also
    --------
    ico2ico

    Examples
    --------
    >>> from surfify.utils import icosahedron, text2ico
    >>> from surfify.datasets import make_classification
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> ico2_std_verts, ico2_std_tris = icosahedron(order=2, standard_ico=True)
    >>> X, y = make_classification(ico2_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> y_std = text2ico(y, ico2_verts, ico2_std_verts)
    >>> plot_trisurf(ico2_std_verts, triangles=ico2_std_tris, texture=y_std,
                     is_label=True)
    >>> plt.show()

    Parameters
    ----------
    texture: array (N, K)
        the input texture to project.
    vertices: array (N, 3)
        the vertices corresponding to the input texture.
    ref_vertices: array (N, 3)
        the reference/target vertices.
    atol: float, default 1e-4
        tolerance when matching the vertices.

    Returns
    -------
    texture: array (N, K)
        the texture projected on the reference icosahedron.
    """
    rotation = ico2ico(vertices, ref_vertices)
    new_vertices = rotation.apply(vertices)
    new_order = find_corresponding_order(
        new_vertices, ref_vertices, atol=atol, axis=0)
    return texture[new_order]


def find_corresponding_order(array, ref_array, atol=1e-4, axis=0):
    """ Find unique match between two arrays: assume that arrays are the
    same up to a permutation.

    Parameters
    ----------
    array: array (N, *)
        the array to find the corresponding order for.
    ref_array: array (N, *)
        the reference array on which the order is base.
    atol: float, default 1e-4
        tolerance when matching the values.
    axis: int, default 0
        axis along which to permute ordering.

    Returns
    -------
    new_order: array (N, )
        the indices to match the input array with the reference array.
    """
    array = np.asarray(array)
    ref_array = np.asarray(ref_array)
    if not np.array_equal(array.shape, ref_array.shape):
        raise ValueError("The arrays must be permuted versions of each other "
                         "and must have the same shape.")
    new_order = []
    other_dims = list(range(array.ndim))
    other_dims.remove(axis)
    other_dims = tuple(other_dims)
    for idx in range(len(array)):
        match = np.isclose(array, np.take(ref_array, idx, axis=axis),
                           atol=atol).all(other_dims)
        idx = np.where(match)[0]
        if len(idx) != 1:
            raise ValueError(
                "The arrays must be permuted versions of each other and an "
                "element in the reference array was not found or found "
                "multiple times.")
        new_order.append(idx[0])
    return new_order
