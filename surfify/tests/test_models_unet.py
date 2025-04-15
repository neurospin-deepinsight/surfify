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


class TestModelsUNet(unittest.TestCase):
    """ Test the SphericalUNet.
    """
    def setUp(self):
        """ Setup test.
        """
        self.order = 3
        self.vertices, _ = utils.icosahedron(
            order=self.order, standard_ico=True)
        self.n_classes = 2
        self.depth = 2
        self.start_filts = 8
        self.conv_modes = ["DiNe", "RePa"]
        self.rings = [1, 2]
        self.up_modes = ["interp", "transpose", "maxpad", "zeropad"]
        self.X, self.y = datasets.make_classification(
            self.vertices, n_samples=40, n_classes=self.n_classes, scale=1,
            seed=42)
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test SphericalUNet forward.
        """
        for conv_mode in self.conv_modes:
            model = models.SphericalUNet(
                in_order=self.order, in_channels=self.n_classes,
                out_channels=self.n_classes, depth=self.depth,
                start_filts=self.start_filts, conv_mode=conv_mode,
                up_mode="interp", standard_ico=True)
            out = model(self.X)
        for n_rings in self.rings:
            model = models.SphericalUNet(
                in_order=self.order, in_channels=self.n_classes,
                out_channels=self.n_classes, depth=self.depth,
                start_filts=self.start_filts, conv_mode="DiNe",
                dine_size=n_rings, up_mode="interp", standard_ico=True)
            out = model(self.X)
        for up_mode in self.up_modes:
            model = models.SphericalUNet(
                in_order=self.order, in_channels=self.n_classes,
                out_channels=self.n_classes, depth=self.depth,
                start_filts=self.start_filts, conv_mode="DiNe",
                dine_size=1, up_mode=up_mode, standard_ico=True)
            out = model(self.X)


class TestModelsGUNet(unittest.TestCase):
    """ Test the SphericalGUNet.
    """
    def setUp(self):
        """ Setup test.
        """
        self.order = 3
        self.n_classes = 2
        self.depth = 2
        self.start_filts = 8
        ico_vertices, _ = utils.icosahedron(
            order=self.order, standard_ico=True)
        X, y = datasets.make_classification(
            ico_vertices, n_samples=40, n_classes=self.n_classes, scale=1,
            seed=42)
        self.X = []
        for sample_idx in range(X.shape[0]):
            _X = []
            for ch_idx in range(X.shape[1]):
                _X.append(utils.text2grid(ico_vertices, X[sample_idx, ch_idx]))
            self.X.append(_X)
        self.X = np.asarray(self.X).astype(np.float32)
        self.y = utils.text2grid(ico_vertices, y)
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_forward(self):
        """ Test SphericalGUNet forward.
        """
        model = models.SphericalGUNet(
            in_channels=self.n_classes, out_channels=self.n_classes,
            input_dim=192, depth=self.depth, start_filts=self.start_filts)
        out = model(self.X)


if __name__ == "__main__":

    utils.setup_logging(level="debug")
    unittest.main()
