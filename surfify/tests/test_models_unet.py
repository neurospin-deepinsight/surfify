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
import torch
from surfify.models import SphericalUNet
from surfify.datasets import make_classification


class TestModelsUNet(unittest.TestCase):
    """ Test the SphericalUNet.
    """
    def setUp(self):
        """ Setup test.
        """
        self.order = 3
        self.n_classes = 2
        self.depth = 2
        self.start_filts = 8
        self.conv_modes = ["1ring", "2ring", "repa"]
        self.up_modes = ["interp", "transpose", "maxpad", "zeropad"]
        self.X, self.y = make_classification(
            self.order, n_samples=40, n_classes=self.n_classes, scale=1,
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
            model = SphericalUNet(
                in_order=self.order, in_channels=self.n_classes,
                out_channels=self.n_classes, depth=self.depth,
                start_filts=self.start_filts, conv_mode=conv_mode,
                up_mode="interp")
            out = model(self.X)
        for up_mode in self.up_modes:
            model = SphericalUNet(
                in_order=self.order, in_channels=self.n_classes,
                out_channels=self.n_classes, depth=self.depth,
                start_filts=self.start_filts, conv_mode="1ring",
                up_mode=up_mode)
            out = model(self.X)


if __name__ == "__main__":

    from surfify.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
