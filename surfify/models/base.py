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
    icosahedron, neighbors, number_of_ico_vertices, downsample, interpolate,
    neighbors_rec, get_logger)
from ..nn import IcoDiNeConv, IcoRePaConv

# Global parameters
logger = get_logger()


class SphericalBase(nn.Module):
    """ Spherica Network Building Information.

    Use either RePa - Rectangular Patch convolution method or DiNe - Direct
    Neighbor convolution method.
    """
    Ico = namedtuple("Ico", ["order", "vertices", "triangles",
                             "neighbor_indices", "down_indices", "up_indices",
                             "conv_neighbor_indices"])

    def __init__(self, input_order, n_layers, conv_mode="DiNe",
                 dine_size=1, repa_size=5, repa_zoom=5, standard_ico=True,
                 cachedir=None):
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
            a multiplicative factor applied to the rectangular grid in the
            tangent space.
        standard_ico: bool, default True
            optionaly use FreeSurfer tesselation.
        cachedir: str, default None
            set this folder to use smart caching speedup.
        """
        super(SphericalBase, self).__init__()
        self.input_order = input_order
        self.n_layers = n_layers
        self.conv_mode = conv_mode
        self.dine_size = dine_size
        self.repa_size = repa_size
        self.repa_zoom = repa_zoom
        self.standard_ico = standard_ico
        self.cachedir = cachedir
        self.memory = Memory(cachedir, verbose=0)
        if conv_mode == "RePa":
            self.sconv = IcoRePaConv
        else:
            self.sconv = IcoDiNeConv
        self.build_ico()

    def build_ico(self):
        """ Build an ico dictionnary containing icosehedron informations at
        each order of interest with the related upsampling and downsampling
        informations.
        """
        self.ico = {}
        icosahedron_cached = self.memory.cache(icosahedron)
        neighbors_cached = self.memory.cache(neighbors)
        neighbors_rec_cached = self.memory.cache(neighbors_rec)
        for order in range(self.input_order - self.n_layers,
                           self.input_order + 1):
            vertices, triangles = icosahedron_cached(
                order=order, standard_ico=self.standard_ico,
                path=self.cachedir)
            logger.debug("- ico {0}: verts {1} - tris {2}".format(
                order, vertices.shape, triangles.shape))
            neighs = neighbors_cached(
                vertices, triangles, depth=1, direct_neighbor=True)
            neighs = np.asarray(list(neighs.values()))
            logger.debug("- neighbors {0}: {1}".format(order, neighs.shape))
            if self.conv_mode == "DiNe":
                if self.dine_size == 1:
                    conv_neighs = neighs
                else:
                    conv_neighs = neighbors_cached(
                        vertices, triangles, depth=self.dine_size,
                        direct_neighbor=True)
                    conv_neighs = np.asarray(list(conv_neighs.values()))
                logger.debug("- conv neighbors {0}: {1}".format(
                    order, conv_neighs.shape))
            elif self.conv_mode == "RePa":
                conv_neighs, conv_weights, _ = neighbors_rec_cached(
                    vertices, triangles, size=self.repa_size,
                    zoom=self.repa_zoom)
                logger.debug("- conv neighbors {0}: {1} - {2}".format(
                    order, conv_neighs.shape, conv_weights.shape))
                conv_neighs = (conv_neighs, conv_weights)
            else:
                raise ValueError("Unexptected convolution mode.")
            self.ico[order] = self.Ico(
                order=order, vertices=vertices, triangles=triangles,
                neighbor_indices=neighs, down_indices=None, up_indices=None,
                conv_neighbor_indices=conv_neighs)
        downsample_cached = self.memory.cache(downsample)
        for order in range(
                self.input_order, self.input_order - self.n_layers, -1):
            print(self.standard_ico)
            down_indices = downsample_cached(
                self.ico[order].vertices, self.ico[order - 1].vertices)
            logger.debug("- down {0}: {1}".format(order, down_indices.shape))
            self.ico[order] = self.ico[order]._replace(
                down_indices=down_indices)
        interpolate_cached = self.memory.cache(interpolate)
        for order in range(self.input_order - self.n_layers, self.input_order):
            up_indices = interpolate_cached(
                self.ico[order].vertices, self.ico[order + 1].vertices,
                self.ico[order + 1].triangles)
            up_indices = np.asarray(list(up_indices.values()))
            logger.debug("- up {0}: {1}".format(order, up_indices.shape))
            self.ico[order] = self.ico[order]._replace(
                up_indices=up_indices)
