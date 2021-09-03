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
import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator


def cart2sph(x, y, z):
    """ Cartesian to spherical coordinate transform.

    .. math::

        \alpha = \arctan \left( \frac{y}{x} \right) \\
        \beta = \arctan \left( \frac{z}{\sqrt{x^2 + y^2}} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}

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

    .. math::

        x = r \cos \alpha \cos \beta \\
        y = r \sin \alpha \cos \beta \\
        z = r \sin \beta

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

    Parameters
    ----------
    vertices array (N, 3)
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

    Parameters
    ----------
    vertices array (N, 3)
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
