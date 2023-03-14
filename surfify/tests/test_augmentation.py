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
import surfify.utils as utils
import surfify.augmentation as augment


class TestAugmentation(unittest.TestCase):
    """ Test spherical augmentation.
    """
    def setUp(self):
        """ Setup test.
        """
        pass

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_random_rotation(self):
        """ Test SphericalRandomRotation.
        """
        vertices, triangles = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.ones((n_vertices, ), dtype=int)
        processor = augment.SurfRotation(
            vertices, triangles, phi=10, theta=0, psi=0)
        data_rot = processor(data)
        self.assertEqual(len(data), len(data_rot))

    def test_random_cut(self):
        """ Test SphericalRandomCut.
        """
        vertices, triangles = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.ones((n_vertices, ), dtype=int)
        processor = augment.SurfCutOut(
            vertices, triangles, neighs=None, patch_size=3,
            n_patches=1, replacement_value=5)
        data_cut = processor(data)
        self.assertEqual(len(data), len(data_cut))


if __name__ == "__main__":

    utils.setup_logging(level="debug")
    unittest.main()
