# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Generate samples of synthetic data sets.
"""

# Imports
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ..utils import icosahedron


def make_classification(order, n_samples=40, n_classes=2, scale=1, seed=None):
    """ Generate a random n-class classification problem.

    Parameters
    ----------
    order: int
        the icosahedron order.
    n_samples: int, default 40
        the number of gene rate samples.
    n_classes: int, default 2
        the number of classes.
    scale: int or list of int, default 1
        the scale of each Gaussian.
    seed: int, default None
        seed to control random number generation.

    Returns
    -------
    X: ndarray (n_samples, n_classes, n_vertices)
        the generated samples.
    y: ndarray (n_vertices, )
        the assocaited labels.
    """
    # Check inputs
    if not isinstance(scale, list):
        scale = [scale] * n_classes
    if n_classes != len(scale):
        raise ValueError(
            "'scale' should be an integer or a list of integer with "
            "'n_features' values.")

    # Load surface
    ico_vertices, ico_triangles = icosahedron(order=order)
    n_vertices = len(ico_vertices)

    # Generate labels
    np.random.seed(seed)
    features = []
    for loc in ico_vertices[np.random.randint(0, n_vertices, n_classes)]:
        dist = np.linalg.norm(ico_vertices - loc, axis=1)
        features.append(norm.pdf(dist, loc=0, scale=1))
    features = np.asarray(features)
    y = np.argmax(features, axis=0).astype(int)

    # Generate samples
    X = np.zeros((n_samples, n_classes, n_vertices), dtype=np.float32)
    np.random.seed(seed)
    locs = np.random.rand(n_classes) * 2
    for klass in range(n_classes):
        indices = np.argwhere(y == klass).squeeze()
        for loc, scl in zip(locs, scale):
            np.random.seed(seed)
            X[:, klass, indices] = np.random.normal(loc=loc, scale=scl,
                                                    size=len(indices))

    return X, y


class ClassificationDataset(Dataset):
    """ Generate a random n-class classification dataset.
    """
    def __init__(self, order, n_samples=40, n_classes=2, scale=1, seed=None):
        """ Init ClassificationDataset.

        Parameters
        ----------
        order: int
            the icosahedron order.
        n_samples: int, default 40
            the number of gene rate samples.
        n_classes: int, default 2
            the number of classes.
        scale: int or list of int, default 1
            the scale of each Gaussian.
        seed: int, default None
            seed to control random number generation.
        """
        super(ClassificationDataset).__init__()
        self.X, y = make_classification(
            order, n_samples, n_classes, scale, seed)
        y = np.expand_dims(y, axis=0)
        self.y = np.repeat(y, len(self.X), axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
