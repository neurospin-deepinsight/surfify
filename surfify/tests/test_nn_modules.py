# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import unittest
import numpy as np
import torch
from surfify import utils
from surfify import nn


class TestNNModules(unittest.TestCase):
    """ Test spherical modules.
    """
    def setUp(self):
        """ Setup test.
        """
        self.ico1_vertices, ico1_triangles = utils.icosahedron(order=1)
        self.ico2_vertices, ico2_triangles = utils.icosahedron(order=2)
        self.ico3_vertices, ico3_triangles = utils.icosahedron(order=3)
        self.up_indices = utils.interpolate(
            self.ico2_vertices, self.ico3_vertices, ico3_triangles)
        self.up_indices = np.asarray(list(self.up_indices.values()))
        self.down_indices = utils.downsample(
            self.ico3_vertices, self.ico2_vertices)
        self.neighbor_indices = utils.neighbors(
            self.ico3_vertices, ico3_triangles, depth=1, direct_neighbor=True)
        self.neighbor_indices = np.asarray(
            list(self.neighbor_indices.values()))
        self.neighbor_rec = utils.neighbors_rec(
            self.ico3_vertices, ico3_triangles, size=5, zoom=5)[:2]
        self.down_neigh_indices = utils.neighbors(
            self.ico3_vertices, ico3_triangles, depth=1, direct_neighbor=True)
        self.down_neigh_indices = np.asarray(
            list(self.down_neigh_indices.values()))
        self.ico2_tensor = torch.zeros((10, 8, len(self.ico2_vertices)))
        self.ico3_tensor = torch.zeros((10, 4, len(self.ico3_vertices)))
        self.grid2_tensor = torch.zeros((10, 8, 96, 96))
        self.grid3_tensor = torch.zeros((10, 4, 192, 192))

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_generic_up_conv(self):
        """ Test IcoGenericUpConv module.
        """
        module = nn.IcoGenericUpConv(
            in_feats=8, out_feats=4, up_neigh_indices=self.neighbor_indices,
            down_indices=self.down_indices)
        x = module(self.ico2_tensor)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_up_conv(self):
        """ Test IcoUpConv module.
        """
        module = nn.IcoUpConv(
            in_feats=8, out_feats=4, up_neigh_indices=self.neighbor_indices,
            down_indices=self.down_indices)
        x = module(self.ico2_tensor)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_max_index_up_sample(self):
        """ Test IcoMaxIndexUpSample module.
        """
        module = nn.IcoPool(
            down_neigh_indices=self.down_neigh_indices,
            down_indices=self.down_indices, pooling_type="max")
        _, max_pool_indices = module(self.ico3_tensor)
        module = nn.IcoMaxIndexUpSample(
            in_feats=8, out_feats=4, up_neigh_indices=self.neighbor_indices,
            down_indices=self.down_indices)
        x = module(self.ico2_tensor, max_pool_indices)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_up_sample(self):
        """ Test IcoUpConv module.
        """
        module = nn.IcoUpSample(
            in_feats=8, out_feats=4, up_neigh_indices=self.up_indices)
        x = module(self.ico2_tensor)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_fix_index_up_sample(self):
        """ Test IcoFixIndexUpSample module.
        """
        module = nn.IcoFixIndexUpSample(
            in_feats=8, out_feats=4, up_neigh_indices=self.up_indices)
        x = module(self.ico2_tensor)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_pool(self):
        """ Test IcoPool module.
        """
        for pooling_type in ("mean", "max"):
            module = nn.IcoPool(
                down_neigh_indices=self.down_neigh_indices,
                down_indices=self.down_indices, pooling_type=pooling_type)
            x, max_pool_indices = module(self.ico3_tensor)
            self.assertTrue(x.shape[1] == self.ico3_tensor.shape[1])
            self.assertTrue(x.shape[2] == len(self.ico2_vertices))

    def test_dine_conv(self):
        """ Test IcoDiNeConv module.
        """
        module = nn.IcoDiNeConv(
            in_feats=4, out_feats=4, neigh_indices=self.neighbor_indices)
        x = module(self.ico3_tensor)
        self.assertTrue(x.shape[1] == self.ico3_tensor.shape[1])
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_repa_conv(self):
        """ Test IcoRePaConv module.
        """
        module = nn.IcoRePaConv(
            in_feats=4, out_feats=4, neighs=self.neighbor_rec)
        x = module(self.ico3_tensor)
        self.assertTrue(x.shape[1] == self.ico3_tensor.shape[1])
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_spma_conv(self):
        """ Test IcoSpMaConv module.
        """
        module = nn.IcoSpMaConv(
            in_feats=4, out_feats=4, kernel_size=3, stride=1, pad=1)
        x = module(self.grid3_tensor)
        self.assertTrue(x.shape[1] == 4)
        self.assertTrue(x.shape[2] == self.grid3_tensor.shape[2])
        self.assertTrue(x.shape[3] == self.grid3_tensor.shape[3])

    def test_spma_up_conv(self):
        """ Test IcoSpMaConvTranspose module.
        """
        module = nn.IcoSpMaConvTranspose(
            in_feats=8, out_feats=4, kernel_size=4, stride=2, zero_pad=3,
            pad=1)
        x = module(self.grid2_tensor)
        self.assertTrue(x.shape[1] == 4)
        self.assertTrue(x.shape[2] == self.grid3_tensor.shape[2])
        self.assertTrue(x.shape[3] == self.grid3_tensor.shape[3])


if __name__ == "__main__":

    utils.setup_logging(level="debug")
    unittest.main()
