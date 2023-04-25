# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Original mixup augmentations.
"""

# Imports
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from .utils import RandomAugmentation, listify


class MixUpAugmentation(RandomAugmentation):
    """ Aplly an augmentation with random parameters defined in intervals.
    """
    def __init__(self, prob, n_vertices):
        """ Init class.

        Parameters
        ----------
        prob: float
            the probability of curuption.
        n_vertices: int (N, )
            the size of the cortical measures.
        """
        super().__init__()
        self.prob = prob
        self.n_vertices = n_vertices
        self.rand_mask()

    def _randomize(self):
        """ Update the random parameters.
        """
        super()._randomize()
        if self.writable:
            self.rand_mask()

    def rand_mask(self):
        """ Generate a binary corruption mask.
        """
        self.mask = np.random.binomial(n=1, p=self.prob, size=self.n_vertices)


class HemiMixUp(MixUpAugmentation):
    """ Randomly permutes a subject’s measurements at specific vertices
    across hemispheres, assuming a vertex-to-vertex correspondence between
    hemispheres.
    """
    def __init__(self, prob, n_vertices):
        """ Init class.

        Parameters
        ----------
        prob: float
            the probability of curuption.
        n_vertices: int (N, )
            the size of the cortical measures.
        """
        super().__init__(prob, n_vertices)

    def run(self, data, controlateral_data):
        """ Applies the hemispheric permutations.

        Parameters
        ----------
        data: array (N, )
            input data/texture.
        controlateral_data: array (N, )
            input controlateral data/texture.

        Returns
        -------
        data: arr (N, )
            permuted input data.
        """
        data[self.mask == 1] = controlateral_data[self.mask == 1]
        return data


class GroupMixUp(MixUpAugmentation):
    """ Randomly bootstraps measures at specific vertices across a group of
    K subjects, assuming a vertex-to-vertex correspondence between
    hemispheres.
    """
    def __init__(self, prob, n_vertices):
        """ Init class.

        Parameters
        ----------
        prob: float
            the probability of curuption.
        n_vertices: int (N, )
            the size of the cortical measures.
        """
        super().__init__(prob, n_vertices)

    def run(self, data, group_data, n_samples=1):
        """ Applies the group bootstaping.

        Parameters
        ----------
        data: array (N, )
            input data/texture.
        group_data: array (k, N)
            input group data/textures.
        n_samples: int, default 1
            the number of bootstraping to be performed.

        Returns
        -------
        data: arr (N, ) or (M, N)
            bootsraped input data.
        """
        _b_data = []
        group_size = len(group_data)
        for idx in range(n_samples):
            _data = data.copy()
            _selector = np.random.choice(group_size, replace=True,
                                         size=self.n_vertices)
            _b_sample = group_data[_selector, range(self.n_vertices)]
            _data[self.mask == 1] = _b_sample[self.mask == 1]
            _b_data.append(_data)
        _b_data = np.array(_b_data)
        return np.squeeze(_b_data)

    @classmethod
    def groupby(cls, data, by=("texture", ), n_neighbors=30, n_components=20,
                meta=None, weights=None):
        """ Regroup subjects based on a combination of metrics.

        Parameters
        ----------
        data: array (M, N)
            input data/textures.
        by: list of str, default ('texture', )
            used to determine the metrics.
        n_neighbors: int, default 30
            the number of neighbors.
        n_components: int, default 20
            the number of PCA components, used to reduce the input data size.
        meta: pandas.DataFrame, default None
            the external data.
        weights: array, default None
            the weight applied to each distance matrix formed from the metric
            defined in the by parameter.

        Returns
        -------
        neigh_ind: array (M, n_neighbors)
            indices of the nearest subjects in the population.
        """
        dists = []
        weights = weights or [1, ] * len(by)
        assert len(weights) == len(by)
        for dtype in by:
            if dtype == "texture":
                pca = PCA(n_components=n_components)
                reduced_data = pca.fit_transform(data)
                dists.append(squareform(pdist(reduced_data, "euclidean")))
            else:
                dtypes = listify(dtype)
                meta_data = meta[dtypes].values
                dists.append(squareform(pdist(meta_data, "euclidean")))
        dist = np.sum(np.array(dists).transpose(1, 2, 0) * np.array(weights),
                      axis=-1)
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors + 1, metric="precomputed").fit(dist)
        _, neigh_ind = nbrs.kneighbors(dist)
        return neigh_ind[1:]
