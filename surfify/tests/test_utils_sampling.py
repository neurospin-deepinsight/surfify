# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import os
import numpy as np
import unittest
from surfify.utils import (
    downsample, downsample_data, downsample_ico,
    interpolate, interpolate_data,
    neighbors, neighbors_rec,
    icosahedron, number_of_ico_vertices, order_of_ico_from_vertices,
    setup_logging, find_neighbors, order_triangles, rotate_data,
    patch_tri)


class TestUtilsSampling(unittest.TestCase):
    """ Test spherical sampling.
    """
    def setUp(self):
        """ Setup test.
        """
        self.cachedir = os.environ["HOME"]
        pass

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_patch_tri(self):
        """ Test patch_tri function.
        """
        res = {
            3: {"num_patches": 1280, "num_vertices": 45},
            4: {"num_patches": 320, "num_vertices": 153},
            5: {"num_patches": 80, "num_vertices": 561},
            6: {"num_patches": 20, "num_vertices": 2145}
        }
        for size in range(3, 7):
            patches = patch_tri(order=6, size=size, direct_neighbor=True,
                                n_jobs=-1)
            self.assertTrue(res[size]["num_patches"] == patches.shape[0])
            self.assertTrue(res[size]["num_vertices"] == patches.shape[1])

    def test_icosahedron(self):
        """ Test icosahedron function.
        """
        for order in range(4):
            vertices, _ = icosahedron(order)
            self.assertTrue(len(vertices) == number_of_ico_vertices(order))
            self.assertTrue(
                order == order_of_ico_from_vertices(len(vertices)))

    def test_neighbors(self):
        """ Test neighbors function.
        """
        vertices, triangles = icosahedron(order=1)
        neighs = neighbors(vertices, triangles, depth=1, direct_neighbor=True)
        self.assertTrue(len(neighs) == len(vertices))
        self.assertTrue(all([len(elem) == 7 for elem in neighs.values()]))

    def test_neighbors_rec(self):
        """ Test neighbors_rec function.
        """
        vertices, triangles = icosahedron(order=1)
        neighs, weights, grid_in_sphere = neighbors_rec(
            vertices, triangles, size=5, zoom=5)
        self.assertTrue(grid_in_sphere.shape == (len(vertices), 25, 3))

    def test_find_neighbors(self):
        """ Test find_neighbors function.
        """
        vertices, triangles = icosahedron(order=1)
        neighs = neighbors(vertices, triangles, depth=1, direct_neighbor=True)
        node_neighs = find_neighbors(0, order=1, neighbors=neighs)
        self.assertTrue(all([node in node_neighs for node in neighs[0]]))

    def test_interpolate(self):
        """ Test interpolate function.
        """
        vertices, triangles = icosahedron(order=1)
        target_vertices, _ = icosahedron(order=0)
        interp = interpolate(target_vertices, vertices, triangles)
        self.assertTrue(len(interp) == len(vertices))

    def test_interpolate_data(self):
        """ Test interpolate_data function.
        """
        n_ico1_vertices = number_of_ico_vertices(order=1)
        n_ico3_vertices = number_of_ico_vertices(order=3)
        data = np.ones((n_ico1_vertices, ), dtype=int)
        data = data.reshape(1, -1, 1)
        interp_data = interpolate_data(data, by=2).squeeze()
        self.assertTrue(len(interp_data) == n_ico3_vertices)

    def test_downsample(self):
        """ Test downsample function.
        """
        vertices, triangles = icosahedron(order=1)
        target_vertices, _ = icosahedron(order=0)
        down_indexes = downsample(vertices, target_vertices)
        self.assertTrue(len(down_indexes) == len(target_vertices))
        self.assertTrue(all(down_indexes == range(len(target_vertices))))

    def test_downsample_data(self):
        """ Test downsample_data function.
        """
        n_ico1_vertices = number_of_ico_vertices(order=1)
        n_ico3_vertices = number_of_ico_vertices(order=3)
        data = np.ones((n_ico3_vertices, ), dtype=int)
        data = data.reshape(1, -1, 1)
        down_data = downsample_data(data, by=2).squeeze()
        self.assertTrue(len(down_data) == n_ico1_vertices)

    def test_downsample_ico(self):
        """ Test downsample function.
        """
        vertices, triangles = icosahedron(order=4)
        target_vertices, _ = icosahedron(order=1)
        new_vertices, _ = downsample_ico(
            vertices, triangles, by=3)
        self.assertTrue(np.array_equal(target_vertices, new_vertices))

    def test_rotate_data(self):
        """ Test rotate_data function.
        """
        vertices, triangles = icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.ones((n_vertices, ), dtype=int)
        data = data.reshape(1, -1, 1)
        rot_data = rotate_data(data, vertices, triangles, angles=(360, 0, 0))
        rot_data_euclid = rotate_data(data, vertices, triangles,
                                      angles=(360, 0, 0),
                                      interpolation="euclidian")
        self.assertTrue(np.allclose(data, rot_data))
        self.assertTrue(np.allclose(data, rot_data_euclid))

    def test_order_triangles(self):
        """ Test order_triangles function.
        """
        vertices, triangles = icosahedron(order=0)
        clockwise_tris = order_triangles(
            vertices, triangles, clockwise_from_center=True)
        counter_clockwise_tris = order_triangles(
            vertices, triangles, clockwise_from_center=False)
        self.assertTrue(np.allclose(clockwise_tris,
                                    counter_clockwise_tris[:, (0, 2, 1)]))


if __name__ == "__main__":

    setup_logging(level="debug")
    unittest.main()
