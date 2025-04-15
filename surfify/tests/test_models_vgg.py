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


class TestModelsVGG(unittest.TestCase):
    """ Test the SphericalVGG.
    """
    def setUp(self):
        """ Setup test.
        """
        self.order = 3
        self.n_classes = 2
        self.n_samples_per_class = 40
        self.depth = 2
        self.cfg = [64, "M", 128, "M"]
        ico_vertices, _ = utils.icosahedron(
            order=self.order, standard_ico=True)
        X1, _ = datasets.make_classification(
            ico_vertices, n_samples=self.n_samples_per_class,
            n_classes=self.n_classes, scale=1, seed=42)
        X2, _ = datasets.make_classification(
            ico_vertices, n_samples=self.n_samples_per_class,
            n_classes=self.n_classes, scale=1, seed=82)
        self.X = np.concatenate((X1, X2), axis=0)
        self.y = np.asarray(
            [0] * self.n_samples_per_class + [1] * self.n_samples_per_class)
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test SphericalVGG forward.
        """
        model = models.SphericalVGG(
            input_channels=self.n_classes, cfg=self.cfg,
            n_classes=self.n_classes, input_order=self.order, conv_mode="DiNe",
            dine_size=1, hidden_dim=128, init_weights=True,
            standard_ico=True)
        out = model(self.X, self.X)
        model = models.SphericalVGG(
            input_channels=self.n_classes, cfg=self.cfg,
            n_classes=self.n_classes, input_order=self.order, conv_mode="RePa",
            repa_size=5, repa_zoom=5, hidden_dim=128, init_weights=True,
            standard_ico=True)
        out = model(self.X, self.X)


class TestModelsGVGG(unittest.TestCase):
    """ Test the SphericalGVGG.
    """
    def setUp(self):
        """ Setup test.
        """
        self.order = 3
        self.n_classes = 2
        self.n_samples_per_class = 40
        self.depth = 2
        ico_vertices, _ = utils.icosahedron(
            order=self.order, standard_ico=True)
        X1, _ = datasets.make_classification(
            ico_vertices, n_samples=self.n_samples_per_class,
            n_classes=self.n_classes, scale=1, seed=42)
        X2, _ = datasets.make_classification(
            ico_vertices, n_samples=self.n_samples_per_class,
            n_classes=self.n_classes, scale=1, seed=82)
        X = np.concatenate((X1, X2), axis=0)
        self.y = np.asarray(
            [0] * self.n_samples_per_class + [1] * self.n_samples_per_class)
        self.X = []
        for sample_idx in range(X.shape[0]):
            _X = []
            for ch_idx in range(X.shape[1]):
                _X.append(utils.text2grid(ico_vertices, X[sample_idx, ch_idx]))
            self.X.append(_X)
        self.X = np.asarray(self.X).astype(np.float32)
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test SphericalGVGG forward.
        """
        model = models.SphericalGVGG11(
            input_channels=self.n_classes, n_classes=self.n_classes,
            input_dim=194, hidden_dim=128, init_weights=True)
        out = model(self.X, self.X)


if __name__ == "__main__":

    utils.setup_logging(level="debug")
    unittest.main()
