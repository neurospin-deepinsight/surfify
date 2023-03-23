# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A module with common augmentation utility functions.
"""

# Import
import abc
import numbers
import torch
import numpy as np
from collections import namedtuple


class RandomAugmentation(object):
    """ Aplly an augmentation with random parameters defined in intervals.
    """
    Interval = namedtuple("Interval", ["low", "high", "dtype"])

    def __init__(self):
        """ Init class.
        """
        self.intervals = {}
        self.writable = True

    def _randomize(self, name=None):
        """ Update the random parameters.
        """
        if self.writable:
            for param, bound in self.intervals.items():
                if name is not None and param == name:
                    setattr(self, param, self._rand(bound))
                elif name is None:
                    setattr(self, param, self._rand(bound))

    def _rand(self, bound):
        """ Generate a new random value.
        """
        if bound.dtype == int:
            return np.random.randint(bound.low, bound.high + 1)
        elif bound.dtype == float:
            return np.random.uniform(bound.low, bound.high)
        else:
            raise ValueError(f"'{bound.dtype}' dtype not supported.")

    def __setattr__(self, name, value):
        """ Store intervals.
        """
        if isinstance(value, RandomAugmentation.Interval):
            self.intervals[name] = value
            value = self._rand(value)
        super().__setattr__(name, value)

    def __call__(self, data, *args, **kwargs):
        """ Applies the augmentation to the data.

        Parameters
        ----------
        data: array (N, )
            input data/texture.

        Returns
        -------
        _data: arr (N, )
            augmented input data.
        """
        self._randomize()
        return self.run(data, *args, **kwargs)

    @abc.abstractmethod
    def run(self, data):
        return


def interval(bound, dtype=float):
    """ Create an interval.

    Parameters
    ----------
    bound: 2-uplet or number
        the object used to build the interval.
    dtype: object, default float
        data type: float, int, ...

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(bound, numbers.Number):
        if bound < 0:
            raise ValueError("Specified interval value must be positive.")
        bound = (-bound, bound)
    if len(bound) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = bound
    if min_val > max_val:
        raise ValueError("Wrong interval boundaries.")
    return RandomAugmentation.Interval(min_val, max_val, dtype)


class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self):
        """ Init class.
        """
        self.transforms = []

    def register(self, transform, probability=1):
        """ Register a new transformation.

        Parameters
        ----------
        transform: RandomAugmentation instance
            a transformation.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = self.Transform(transform=transform, probability=probability)
        self.transforms.append(trf)

    def __call__(self, data, *args, **kwargs):
        """ Apply the registered transformations.

        Parameters
        ----------
        data: array (N, ) or (n_channels, N)
            the input data.

        Returns
        -------
        _data: array (N, ) or (n_channels, N)
            the transformed input data.
        """
        ndim = data.ndim
        assert ndim in (1, 2)
        _data = data.copy()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                if ndim == 1:
                    _data = trf.transform(data, *args, **kwargs)
                else:
                    _c_data = []
                    for _data in data:
                        _c_data.append(trf.transform(_data, *args, **kwargs))
                        trf.transform.writable = False
                    trf.transform.writable = True
                    _data = np.array(_c_data)
        return _data


def listify(data):
    """ Ensure that the input is a list or tuple.

    Parameters
    ----------
    data: list or array
        the input data.

    Returns
    -------
    out: list
        the liftify input data.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        return data
    else:
        return [data]


def copy_with_channel_dim(data, to_tensor=False):
    """ Create a copy of a 1 or 2 dimensional array with a channel dimension.

    Parameters
    ----------
    data: array or torch.Tensor
        the input data
    to_tensor: bool, default False
        optionnaly casts input data to a tensor if its not

    Returns
    -------
    _data: array or torch.Tensor
        copy of the input array with same type, except if to_tensor if True
    back_to_numpy: bool
        True if the array was casted to a tensor else False
    """
    n_dim = len(data.shape)
    if n_dim > 2:
        raise ValueError("Input array must be 1 or 2 dimensional.")
    back_to_numpy = False
    if type(data) is torch.Tensor:
        _data = data.clone()
        if n_dim == 1:
            _data = _data.unsqueeze(0)
    elif to_tensor:
        _data = torch.Tensor(data)
        back_to_numpy = True
        if n_dim == 1:
            _data = _data.unsqueeze(0)
    else:
        _data = data.copy()
        if n_dim == 1:
            _data = _data[np.newaxis]

    return _data, back_to_numpy
