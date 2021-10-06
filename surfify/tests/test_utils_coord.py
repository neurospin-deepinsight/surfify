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
from surfify import datasets


class TestUtilsCoord(unittest.TestCase):
    """ Test logging.
    """
    def setUp(self):
        """ Setup test.
        """
        self.tensor = torch.zeros((10, 10))
        self.ico_order = 3
        self.n_classes = 3
        self.ico_vertices, _ = utils.icosahedron(
            order=self.ico_order)
        _, self.labels = datasets.make_classification(
            self.ico_vertices, n_samples=40, n_classes=self.n_classes, scale=1,
            seed=42)
        self.ico_vertices_standard, _ = utils.icosahedron(
            order=self.ico_order, standard_ico=True)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_grid_projection(self):
        """ Test text2grid & grid2text functions.
        """
        proj_texture = utils.text2grid(self.ico_vertices, self.labels)
        texture = utils.grid2text(self.ico_vertices, proj_texture)
        self.assertTrue(np.allclose(self.labels, texture))

    def test_find_corresponding_order(self):
        """ Test the matching between 2 icoshedron of the same order
        """
        a = list(range(10))
        b = list(range(9, -1, -1))
        new_order = utils.coord.find_corresponding_order(b, a)

        self.assertTrue(np.array_equal(new_order, b))

    def test_ico2ico(self):
        """ Test ico2ico function.
        """
        rotation = utils.ico2ico(self.ico_vertices, self.ico_vertices_standard)
        rotation_inv = utils.ico2ico(
            self.ico_vertices_standard, self.ico_vertices)
        rotated_vertices = rotation.apply(self.ico_vertices)
        rotated_and_back_vertices = rotation_inv.apply(rotated_vertices)
        new_order = utils.coord.find_corresponding_order(
            rotated_vertices, self.ico_vertices_standard)
        self.assertTrue(np.allclose(
            self.ico_vertices_standard,
            rotated_vertices[new_order], atol=1e-4))
        self.assertTrue(np.allclose(
            self.ico_vertices,
            rotated_and_back_vertices))

    def test_text2ico(self):
        """ Test text2ico function.
        """
        texture = self.labels
        new_texture = utils.text2ico(
            texture, self.ico_vertices, self.ico_vertices_standard)
        new_and_back_texture = utils.text2ico(
            new_texture, self.ico_vertices_standard, self.ico_vertices)
        self.assertTrue(np.allclose(texture, new_and_back_texture))


if __name__ == "__main__":

    utils.setup_logging(level="debug")
    unittest.main()
