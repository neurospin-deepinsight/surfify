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
        self.ico_vertices, ico_triangles = utils.icosahedron(
            order=self.ico_order)
        _, self.labels = datasets.make_classification(
            self.ico_order, n_samples=40, n_classes=self.n_classes, scale=1,
            seed=42)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_grid_projection(self):
        """ Test the spherical texture to 2-d grid conversion and vis versa.
        """
        proj_texture = utils.text2grid(self.ico_vertices, self.labels)
        texture = utils.grid2text(self.ico_vertices, proj_texture)
        self.assertTrue(np.allclose(self.labels, texture))


if __name__ == "__main__":

    from surfify.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
