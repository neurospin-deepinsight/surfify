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

    def test_surf_rotation(self):
        """ Test SphericalRandomRotation.
        """
        vertices, triangles = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.ones((n_vertices, ), dtype=int)
        data[0] = 0
        processor = augment.SurfRotation(
            vertices, triangles, phi=10, theta=0, psi=0)
        data_rot = processor(data)
        self.assertEqual(len(data), len(data_rot))
        self.assertTrue((data == data_rot).sum() < n_vertices)

    def test_surf_cutout(self):
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
        self.assertTrue((data == data_cut).sum() < n_vertices)

    def test_surf_noise(self):
        """ Test SphericalRandomRotation.
        """
        vertices, _ = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.ones((n_vertices, ), dtype=int)
        processor = augment.SurfNoise(sigma=3)
        data_noise = processor(data)
        self.assertEqual(len(data), len(data_noise))
        self.assertTrue((data == data_noise).sum() < n_vertices)

    def test_surf_blur(self):
        """ Test SphericalRandomCut.
        """
        vertices, triangles = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.ones((n_vertices, ), dtype=int)
        processor = augment.SurfBlur(
            vertices, triangles, sigma=2)
        data_blur = processor(data)
        self.assertEqual(len(data), len(data_blur))
        self.assertTrue((data == data_blur).sum() < n_vertices)


if __name__ == "__main__":

    utils.setup_logging(level="debug")
    unittest.main()
