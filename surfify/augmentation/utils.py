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
import numpy as np
from collections import namedtuple


Interval = namedtuple("Interval", ["low", "high", "dtype"])


class RandomAugmentation(object):
    """ Aplly an augmentation with random parameters defined in intervals.
    """
    def __init__(self):
        """ Init class.
        """
        self.intervals = {}
        self.writable = True

    def _randomize(self):
        """ Update the random parameters.
        """
        if self.writable:
            for param, bound in self.intervals.items():
                setattr(self, param, self._rand(bound))

    def _rand(self, bound):
        """ Generate a new random value.
        """
        if bound.dtype == int:
            return np.random.randint(bound.low, bound.high)
        elif bound.dtype == float:
            return np.random.uniform(bound.low, bound.high)
        else:
            raise ValueError(f"'{bound.dtype}' dtype not supported.")

    def __setattr__(self, name, value):
        """ Store intervals.
        """
        if isinstance(value, Interval):
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
    return Interval(min_val, max_val, dtype)


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
    arr: list or array
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
