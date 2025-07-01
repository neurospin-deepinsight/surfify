##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Methods for scaling, centering, normalization, and more.
"""

import nibabel
import numpy as np

from sklearn.preprocessing import StandardScaler as _StandardScaler


class StandardScaler(_StandardScaler):
    """ Standardize features by removing the mean and scaling to unit variance.

    Based on `sklearn.preprocessing.StandardScaler`.
    """
    def __init__(self, mode="sub", mask=None, copy=True, with_mean=True,
                 with_std=True):
        """ Init scler.

        Parameters
        ----------
        mode: str, default='sub'
            the scaling mode, either 'sub' to operate at the subject level or
            'group' to operate at the group level.
        mask: array or str, default=None
            mask the input data before scaling, i.e. normalize only vertices
            that are masked.
        copy: bool, default=True
            If False, try to avoid a copy and do inplace scaling instead.
            This is not guaranteed to always work inplace; e.g. if the data is
            not a NumPy array or scipy.sparse CSR matrix, a copy may still be
            returned.
        with_mean: bool, default=True
            If True, center the data before scaling.
            This does not work (and will raise an exception) when attempted on
            sparse matrices, because centering them entails building a dense
            matrix which in common use cases is likely to be too large to fit
            in memory.
        with_std: bool, default=True
            If True, scale the data to unit variance (or equivalently,
            unit standard deviation).
        """
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.mask = self._load(mask) if mask is not None else None
        self.mode = mode

    def get_metadata_routing(self, *args, **kwargs):
        """ Get metadata routing of this object.

        .. _metadata_routing:
        """
        super().get_metadata_routing(*args, **kwargs)

    @classmethod
    def _load_mask(cls, mask):
        if isinstance(mask, str):
            if mask.endswith(".gii"):
                mask = np.array(nibabel.load(mask).agg_data())
            elif mask.endswith(".npy"):
                mask = np.load(mask)
            else:
                raise ValueError("Unexpeccted mask file extension!")
        if not isinstance(mask, np.ndarray):
            raise ValueError("The input mask is not a numpy array!")
        return mask

    @classmethod
    def _sanitize(cls, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def fit(self, X, y=None):
        """ Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_subjects, n_vertices)
            the data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y: None
            ignored.

        Returns
        -------
        self: object
            fitted scaler.
        """
        X = self._sanitize(X)
        if self.mode == "sub":
            return super().fit(X.T, y=y, sample_weight=self.mask)
        else:
            return super().fit(X, y=y, sample_weight=None)

    def transform(self, X, copy=None):
        """ Perform standardization by centering and scaling.

        Parameters
        ----------
        X: {array-like, sparse matrix of shape (n_subjects, n_vertices)
            the data used to scale along the features axis.
        copy: bool, default=None
            copy the input X or not.

        Returns
        -------
        X_tr: {ndarray, sparse matrix} of shape (n_subjects, n_vertices)
            transformed array.
        """
        X = self._sanitize(X)
        if self.mode == "sub":
            return super().transform(X.T, copy=copy).T.squeeze()
        else:
            return super().transform(X, copy=copy).squeeze()

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_subjects, n_vertices)
            the data used to scale along the features axis.
        copy: bool, default=None
            copy the input X or not.

        Returns
        -------
        X_tr: {ndarray, sparse matrix} of shape (n_subjects, n_vertices)
            transformed array.
        """
        X = self._sanitize(X)
        if self.mode == "sub":
            return super().inverse_transform(X.T, copy=copy).T.squeeze()
        else:
            return super().inverse_transform(X, copy=copy).squeeze()

    def __call__(self, X):
        """ Standardize input data.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_vertices, )
            input data/texture.

        Returns
        -------
        X_tr: {ndarray, sparse matrix} of shape (n_vertices, )
            transformed data.
        """
        assert self.mode == "sub", (
            "On the fly normalization works only at the subject level!")
        return self.fit_transform(X)

    def __repr__(self):
        return f"{self.__class__.__name__}<mode={self.mode}>"
