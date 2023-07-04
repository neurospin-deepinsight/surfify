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
        """ Test SurfRotation.
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
        """ Test SurfCutOut.
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
        """ Test SurfNoise.
        """
        vertices, _ = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.random.uniform(0, 1, n_vertices)
        processor = augment.SurfNoise(sigma=3)
        data_noise = processor(data)
        self.assertEqual(len(data), len(data_noise))
        self.assertTrue((data == data_noise).sum() < n_vertices)

    def test_surf_blur(self):
        """ Test SurfBlur.
        """
        vertices, triangles = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.random.uniform(0, 1, n_vertices)
        processor = augment.SurfBlur(
            vertices, triangles, sigma=2)
        data_blur = processor(data)
        self.assertEqual(len(data), len(data_blur))
        self.assertTrue((data == data_blur).sum() < n_vertices)
    
    def test_hemi_mixup(self):
        """ Test SurfBlur.
        """
        vertices, _ = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.random.uniform(0, 1, n_vertices)
        controlateral_data = np.random.uniform(0, 1, n_vertices)
        processor = augment.HemiMixUp(0.3, n_vertices)
        _data = processor(data, controlateral_data)
        self.assertEqual(len(data), len(_data))
        self.assertTrue((data == _data).sum() < n_vertices)

    def test_group_mixup(self):
        """ Test SurfBlur.
        """
        vertices, _ = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        # data = np.random.uniform(0, 1, n_vertices)
        all_data = np.random.uniform(0, 1, (100, n_vertices))
        neigh_idx = augment.GroupMixUp.groupby(all_data)
        processor = augment.GroupMixUp(0.3, n_vertices)
        _data = processor(all_data[0], all_data[neigh_idx[0]])
        self.assertEqual(len(all_data[0]), len(_data))
        self.assertTrue((all_data[0] == _data).sum() < n_vertices)

    def test_hemi_mixup(self):
        """ Test SurfBlur.
        """
        vertices, _ = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        data = np.random.uniform(0, 1, n_vertices)
        controlateral_data = np.random.uniform(0, 1, n_vertices)
        processor = augment.HemiMixUp(0.3, n_vertices)
        data_mixup = processor(data, controlateral_data)
        self.assertEqual(len(data), len(data_mixup))
        self.assertTrue((data == data_mixup).sum() < n_vertices)

    def test_group_mixup(self):
        """ Test SurfBlur.
        """
        vertices, _ = utils.icosahedron(order=3)
        n_vertices = len(vertices)
        # data = np.random.uniform(0, 1, n_vertices)
        all_data = np.random.uniform(0, 1, (100, n_vertices))
        neigh_idx = augment.GroupMixUp.groupby(all_data)
        processor = augment.GroupMixUp(0.3, n_vertices)
        data_mixup = processor(all_data[0], all_data[neigh_idx[0]])
        self.assertEqual(len(all_data[0]), len(data_mixup))
        self.assertTrue((all_data[0] == data_mixup).sum() < n_vertices)


if __name__ == "__main__":

    utils.setup_logging(level="debug")
    unittest.main()
