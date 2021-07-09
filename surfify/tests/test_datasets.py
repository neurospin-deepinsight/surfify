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
from torch.utils.data import DataLoader
from surfify.datasets import ClassificationDataset


class TestDatasets(unittest.TestCase):
    """ Test the datasets.
    """
    def setUp(self):
        """ Setup test.
        """
        self.ico_order = 2
        self.n_classes = 3
        self.batch_size = 5

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_classification_dataset(self):
        """ Test ClassificationDataset dataset.
        """
        dataset = ClassificationDataset(
            self.ico_order, n_samples=40, n_classes=self.n_classes, scale=1,
            seed=42)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        X, y = next(iter(loader))
        self.assertTrue(X.shape == (self.batch_size, self.n_classes, 162))
        self.assertTrue(y.shape == (self.batch_size, 162))


if __name__ == "__main__":

    from surfify.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
