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
from surfify import models
from surfify import utils
from surfify import datasets


class TestModelsVAE(unittest.TestCase):
    """ Test the SphericalVAE.
    """
    def setUp(self):
        """ Setup test.
        """
        self.order = 3
        self.n_classes = 2
        self.depth = 2
        ico_vertices, ico_triangles = utils.icosahedron(order=self.order)
        self.X, _ = datasets.make_classification(
            self.order, n_samples=40, n_classes=self.n_classes, scale=1,
            seed=42)
        self.X = torch.from_numpy(self.X)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test SphericalVAE forward.
        """
        model = models.SphericalVAE(
            input_channels=self.n_classes, input_order=self.order,
            latent_dim=32,  conv_mode="DiNe", dine_size=1,
            conv_flts=[32, 64], use_freesurfer=False)
        out = model(self.X, self.X)
        model = models.SphericalVAE(
            input_channels=self.n_classes, input_order=self.order,
            latent_dim=32, conv_mode="RePa", repa_size=5, repa_zoom=5,
            conv_flts=[32, 64], use_freesurfer=False)
        out = model(self.X, self.X)


class TestModelsGVAE(unittest.TestCase):
    """ Test the SphericalGVAE.
    """
    def setUp(self):
        """ Setup test.
        """
        self.order = 3
        self.n_classes = 2
        self.depth = 2
        ico_vertices, ico_triangles = utils.icosahedron(order=self.order)
        X, _ = datasets.make_classification(
            self.order, n_samples=40, n_classes=self.n_classes, scale=1,
            seed=42)
        self.X = []
        for sample_idx in range(X.shape[0]):
            _X = []
            for ch_idx in range(X.shape[1]):
                _X.append(utils.text2grid(ico_vertices, X[sample_idx, ch_idx]))
            self.X.append(_X)
        self.X = np.asarray(self.X)
        self.X = torch.from_numpy(self.X)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test SphericalGVAE forward.
        """
        model = models.SphericalGVAE(
            input_channels=self.n_classes, input_dim=194, latent_dim=32,
            conv_flts=[64, 128, 128])
        out = model(self.X, self.X)


if __name__ == "__main__":

    from surfify.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
