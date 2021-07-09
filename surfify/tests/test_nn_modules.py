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
from surfify.utils import (
    interpolate, downsample, icosahedron, neighbors, neighbors_rec)
from surfify.nn import (
    IcoUpConv, IcoMaxIndexUpSample, IcoFixIndexUpSample, IcoUpSample, IcoPool,
    IcoDiNeConv, IcoRePaConv, IcoGenericUpConv)


class TestNNModules(unittest.TestCase):
    """ Test spherical modules.
    """
    def setUp(self):
        """ Setup test.
        """
        self.ico1_vertices, ico1_triangles = icosahedron(order=1)
        self.ico2_vertices, ico2_triangles = icosahedron(order=2)
        self.ico3_vertices, ico3_triangles = icosahedron(order=3)
        self.up_indices = interpolate(
            self.ico2_vertices, self.ico3_vertices, ico3_triangles)
        self.up_indices = np.asarray(list(self.up_indices.values()))
        self.down_indices = downsample(self.ico3_vertices, self.ico2_vertices)
        self.neighbor_indices = neighbors(
            self.ico3_vertices, ico3_triangles, depth=1, direct_neighbor=True)
        self.neighbor_indices = np.asarray(
            list(self.neighbor_indices.values()))
        self.neighbor_rec = neighbors_rec(
            self.ico3_vertices, ico3_triangles, size=5, zoom=5)[:2]
        self.down_neigh_indices = neighbors(
            self.ico3_vertices, ico3_triangles, depth=1, direct_neighbor=True)
        self.down_neigh_indices = np.asarray(
            list(self.down_neigh_indices.values()))
        self.ico2_tensor = torch.zeros((10, 8, len(self.ico2_vertices)))
        self.ico3_tensor = torch.zeros((10, 4, len(self.ico3_vertices)))

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_generic_up_conv(self):
        """ Test IcoGenericUpConv module.
        """
        module = IcoGenericUpConv(
            in_feats=8, out_feats=4, up_neigh_indices=self.neighbor_indices,
            down_indices=self.down_indices)
        x = module(self.ico2_tensor)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_up_conv(self):
        """ Test IcoUpConv module.
        """
        module = IcoUpConv(
            in_feats=8, out_feats=4, up_neigh_indices=self.neighbor_indices,
            down_indices=self.down_indices)
        x = module(self.ico2_tensor)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_max_index_up_sample(self):
        """ Test IcoMaxIndexUpSample module.
        """
        module = IcoPool(
            down_neigh_indices=self.down_neigh_indices,
            down_indices=self.down_indices, pooling_type="max")
        _, max_pool_indices = module(self.ico3_tensor)
        module = IcoMaxIndexUpSample(
            in_feats=8, out_feats=4, up_neigh_indices=self.neighbor_indices,
            down_indices=self.down_indices)
        x = module(self.ico2_tensor, max_pool_indices)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_up_sample(self):
        """ Test IcoUpConv module.
        """
        module = IcoUpSample(
            in_feats=8, out_feats=4, up_neigh_indices=self.up_indices)
        x = module(self.ico2_tensor)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_fix_index_up_sample(self):
        """ Test IcoFixIndexUpSample module.
        """
        module = IcoFixIndexUpSample(
            in_feats=8, out_feats=4, up_neigh_indices=self.up_indices)
        x = module(self.ico2_tensor)
        self.assertTrue(x.shape[1] == self.ico2_tensor.shape[1] / 2)
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_pool(self):
        """ Test IcoPool module.
        """
        for pooling_type in ("mean", "max"):
            module = IcoPool(
                down_neigh_indices=self.down_neigh_indices,
                down_indices=self.down_indices, pooling_type=pooling_type)
            x, max_pool_indices = module(self.ico3_tensor)
            self.assertTrue(x.shape[1] == self.ico3_tensor.shape[1])
            self.assertTrue(x.shape[2] == len(self.ico2_vertices))

    def test_dine_conv(self):
        """ Test IcoDiNeConv module.
        """
        for n_ring in (1, 2):
            module = IcoDiNeConv(
                in_feats=4, out_feats=4, neigh_indices=self.neighbor_indices,
                n_ring=n_ring)
            x = module(self.ico3_tensor)
            self.assertTrue(x.shape[1] == self.ico3_tensor.shape[1])
            self.assertTrue(x.shape[2] == len(self.ico3_vertices))

    def test_repa_conv(self):
        """ Test IcoRePaConv module.
        """
        module = IcoRePaConv(
            in_feats=4, out_feats=4, neighs=self.neighbor_rec)
        x = module(self.ico3_tensor)
        self.assertTrue(x.shape[1] == self.ico3_tensor.shape[1])
        self.assertTrue(x.shape[2] == len(self.ico3_vertices))


if __name__ == "__main__":

    from surfify.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
