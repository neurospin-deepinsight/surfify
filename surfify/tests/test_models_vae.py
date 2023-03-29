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
        ico_vertices, _ = utils.icosahedron(
            order=self.order, standard_ico=True)
        self.X, _ = datasets.make_classification(
            ico_vertices, n_samples=40, n_classes=self.n_classes, scale=1,
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
            latent_dim=32, conv_mode="DiNe", dine_size=1,
            conv_flts=[32, 64], standard_ico=True)
        out = model(self.X, self.X)
        model = models.SphericalVAE(
            input_channels=self.n_classes, input_order=self.order,
            latent_dim=32, conv_mode="RePa", repa_size=5, repa_zoom=5,
            conv_flts=[32, 64], standard_ico=True)
        out = model(self.X, self.X)


class TestModelsGVAE(unittest.TestCase):
    """ Test the SphericalVAE.
    """
    def setUp(self):
        """ Setup test.
        """
        self.order = 3
        self.n_classes = 2
        self.depth = 2
        ico_vertices, _ = utils.icosahedron(
            order=self.order, standard_ico=True)
        X, _ = datasets.make_classification(
            ico_vertices, n_samples=40, n_classes=self.n_classes, scale=1,
            seed=42)
        self.X = []
        self.input_dim = 192
        for sample_idx in range(X.shape[0]):
            _X = []
            for ch_idx in range(X.shape[1]):
                _X.append(utils.text2grid(
                    ico_vertices, X[sample_idx, ch_idx], resx=self.input_dim,
                    resy=self.input_dim))
            self.X.append(_X)
        self.X = np.asarray(self.X)
        self.X = torch.from_numpy(self.X)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test SphericalVAE forward.
        """
        model = models.SphericalVAE(
            input_channels=self.n_classes, input_dim=self.input_dim,
            latent_dim=32, conv_flts=[64, 128, 128], conv_mode="SpMa")
        out = model(self.X, self.X)


if __name__ == "__main__":

    utils.setup_logging(level="debug")
    unittest.main()
