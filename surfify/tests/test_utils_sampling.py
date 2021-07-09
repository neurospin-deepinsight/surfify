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
from surfify.utils import (
    interpolate, neighbors, downsample, neighbors_rec, icosahedron,
    number_of_ico_vertices)


class TestUtilsSampling(unittest.TestCase):
    """ Test spherical sampling.
    """
    def setUp(self):
        """ Setup test.
        """
        pass

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_icosahedron(self):
        """ Test icosahedron function.
        """
        for order in range(4):
            vertices, triangles = icosahedron(order)
            self.assertTrue(len(vertices) == number_of_ico_vertices(order))

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

    def test_interpolate(self):
        """ Test interpolate function.
        """
        vertices, triangles = icosahedron(order=1)
        target_vertices, _ = icosahedron(order=0)
        interp = interpolate(target_vertices, vertices, triangles)
        self.assertTrue(len(interp) == len(vertices))

    def test_downsample(self):
        """ Test downsample function.
        """
        vertices, triangles = icosahedron(order=1)
        target_vertices, _ = icosahedron(order=0)
        down_indexes = downsample(vertices, target_vertices)
        self.assertTrue(len(down_indexes) == len(target_vertices))
        self.assertTrue(all(down_indexes == range(len(target_vertices))))


if __name__ == "__main__":

    from surfify.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
