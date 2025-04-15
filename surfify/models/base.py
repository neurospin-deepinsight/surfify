# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A base class for spherical networks that generate the requested icosahedrons
and related informations.
"""

# Imports
from collections import namedtuple
import numpy as np
from joblib import Memory
import torch.nn as nn
from ..utils import (
    icosahedron, neighbors, downsample, interpolate,
    neighbors_rec, get_logger)
from ..nn import IcoDiNeConv, IcoRePaConv, IcoPool

# Global parameters
logger = get_logger()


class SphericalBase(nn.Module):
    """ Spherical network base information.

    Use either RePa - Rectangular Patch convolution method or DiNe - Direct
    Neighbor convolution method.

    Examples
    --------
    >>> from surfify.models import SphericalBase
    >>> ico_info = SphericalBase.build_ico_info(input_order=3, n_layers=2)
    >>> print(ico_info.keys())
    """
    Ico = namedtuple("Ico", ["order", "vertices", "triangles",
                             "neighbor_indices", "down_indices", "up_indices",
                             "conv_neighbor_indices"])

    def __init__(self, input_order, n_layers, conv_mode="DiNe",
                 dine_size=1, repa_size=5, repa_zoom=5,
                 dynamic_repa_zoom=False, standard_ico=False, cachedir=None):
        """ Init class.

        Parameters
        ----------
        input_order: int
            the input icosahedron order.
        n_layers: int
            the number of layers in the network.
        conv_mode: str, default 'DiNe'
            use either 'RePa' - Rectangular Patch convolution method or 'DiNe'
            - 1 ring Direct Neighbor convolution method.
        dine_size: int, default 1
            the size of the spherical convolution filter, ie. the number of
            neighbor rings to be considered.
        repa_size: int, default 5
            the size of the rectangular grid in the tangent space.
        repa_zoom: int, default 5
            control the rectangular grid spacing in the tangent space by
            applying a multiplicative factor of `1 / repa_zoom`.
        dynamic_repa_zoom: bool, default False
            dynamically adapt the RePa zoom by applying a multiplicative factor
            of `log(order + 1) + 1`.
        standard_ico: bool, default False
            optionally uses a standard icosahedron tessalation. FreeSurfer
            tesselation is used by default.
        cachedir: str, default None
            set this folder to use smart caching speedup.
        """
        super().__init__()
        self.input_order = input_order
        self.n_layers = n_layers
        self.conv_mode = conv_mode
        self.dine_size = dine_size
        self.repa_size = repa_size
        self.repa_zoom = repa_zoom
        self.dynamic_repa_zoom = dynamic_repa_zoom
        self.standard_ico = standard_ico
        self.cachedir = cachedir
        if conv_mode == "RePa":
            self.sconv = IcoRePaConv
        else:
            self.sconv = IcoDiNeConv
        self.ico = self.build_ico_info(
            input_order, n_layers, conv_mode, dine_size, repa_size, repa_zoom,
            dynamic_repa_zoom, standard_ico, cachedir)

    def _safe_forward(self, block, x, act=None, skip_last_act=False):
        """ Perform a safe forward pass on a specific input block.
        """
        n_mods = len(list(block.children()))
        for cnt, mod in enumerate(block.children()):
            if isinstance(mod, IcoPool):
                x = mod(x)[0]
            else:
                x = mod(x)
                if skip_last_act and cnt == (n_mods - 1):
                    continue
                if act is not None:
                    x = act(x)
        return x

    @classmethod
    def build_ico_info(cls, input_order, n_layers, conv_mode="DiNe",
                       dine_size=1, repa_size=5, repa_zoom=5,
                       dynamic_repa_zoom=False, standard_ico=False,
                       cachedir=None):
        """ Build an dictionnary containing icosehedron informations at
        each order of interest with the related upsampling and downsampling
        informations. This methods is useful to speed up processings
        by caching icosahedron onformations.

        Parameters
        ----------
        input_order: int
            the input icosahedron order.
        n_layers: int
            the number of layers in the network.
        conv_mode: str, default 'DiNe'
            use either 'RePa' - Rectangular Patch convolution method or 'DiNe'
            - 1 ring Direct Neighbor convolution method.
        dine_size: int, default 1
            the size of the spherical convolution filter, ie. the number of
            neighbor rings to be considered.
        repa_size: int, default 5
            the size of the rectangular grid in the tangent space.
        repa_zoom: int, default 5
            control the rectangular grid spacing in the tangent space by
            applying a multiplicative factor of `1 / repa_zoom`.
        dynamic_repa_zoom: bool, default False
            dynamically adapt the RePa zoom by applying a multiplicative factor
            of `log(order + 1) + 1`.
        standard_ico: bool, default False
            optionally uses a standard icosahedron tessalation. FreeSurfer
            tesselation is used by default.
        cachedir: str, default None
            set this folder to use smart caching speedup.

        Returns
        -------
        ico: dict of Ico
            the icosahedron informations at different orders.
        """
        ico = {}
        memory = Memory(cachedir, verbose=0)
        icosahedron_cached = memory.cache(icosahedron)
        neighbors_cached = memory.cache(neighbors)
        neighbors_rec_cached = memory.cache(neighbors_rec)
        for order in range(input_order - n_layers,
                           input_order + 1):
            vertices, triangles = icosahedron_cached(
                order=order, standard_ico=standard_ico)
            logger.debug("- ico {0}: verts {1} - tris {2}".format(
                order, vertices.shape, triangles.shape))
            neighs = neighbors_cached(
                vertices, triangles, depth=1, direct_neighbor=True)
            neighs = np.asarray(list(neighs.values()))
            logger.debug("- neighbors {0}: {1}".format(order, neighs.shape))
            if conv_mode == "DiNe":
                if dine_size == 1:
                    conv_neighs = neighs
                else:
                    conv_neighs = neighbors_cached(
                        vertices, triangles, depth=dine_size,
                        direct_neighbor=True)
                    conv_neighs = np.asarray(list(conv_neighs.values()))
                logger.debug("- conv neighbors {0}: {1}".format(
                    order, conv_neighs.shape))
            elif conv_mode == "RePa":
                if dynamic_repa_zoom:
                    current_zoom = repa_zoom * (np.log(order + 1) + 1)
                else:
                    current_zoom = repa_zoom
                conv_neighs, conv_weights, _ = neighbors_rec_cached(
                    vertices, triangles, size=repa_size,
                    zoom=current_zoom)
                logger.debug("- conv neighbors {0} - {1}: {2} - {3}".format(
                    order, current_zoom, conv_neighs.shape,
                    conv_weights.shape))
                conv_neighs = (conv_neighs, conv_weights)
            else:
                raise ValueError("Unexptected convolution mode.")
            ico[order] = cls.Ico(
                order=order, vertices=vertices, triangles=triangles,
                neighbor_indices=neighs, down_indices=None, up_indices=None,
                conv_neighbor_indices=conv_neighs)
        downsample_cached = memory.cache(downsample)
        for order in range(
                input_order, input_order - n_layers, -1):
            down_indices = downsample_cached(
                ico[order].vertices, ico[order - 1].vertices)
            logger.debug("- down {0}: {1}".format(order, down_indices.shape))
            ico[order] = ico[order]._replace(
                down_indices=down_indices)
        interpolate_cached = memory.cache(interpolate)
        for order in range(input_order - n_layers, input_order):
            up_indices = interpolate_cached(
                ico[order].vertices, ico[order + 1].vertices,
                ico[order + 1].triangles)
            up_indices = np.asarray(list(up_indices.values()))
            logger.debug("- up {0}: {1}".format(order, up_indices.shape))
            ico[order] = ico[order]._replace(
                up_indices=up_indices)
        return ico
