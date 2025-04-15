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


class RandomAugmentation:
    """ Apply an augmentation with random parameters defined in intervals.
    """
    Interval = namedtuple("Interval", ["low", "high", "dtype"])

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
        inplace: bool, default False
            wether to copy or not the input data (pass as a kwargs).

        Returns
        -------
        data: arr (N, )
            augmented input data.
        """
        self._randomize()

        if kwargs.get("inplace", True):
            data = data.copy()
        return self.run(data, *args, **kwargs)

    @abc.abstractmethod
    def run(self, data, *args, **kwargs):
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


class BaseTransformer:
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", [
        "transform", "probability", "randomize_per_channel"])

    def __init__(self):
        """ Init class.
        """
        self.transforms = []

    def register(self, transform, probability=1, randomize_per_channel=True):
        """ Register a new transformation.

        Parameters
        ----------
        transform: RandomAugmentation instance
            a transformation.
        probability: float, default 1
            the transform is applied with the specified probability.
        randomize_per_channel: bool, default True
            a parameter to control if the randomization of tranformation
            parameters must be applied channel-wise.
        """
        trf = self.Transform(transform=transform, probability=probability,
                             randomize_per_channel=randomize_per_channel)
        self.transforms.append(trf)

    @abc.abstractmethod
    def __call__(self, data, *args, **kwargs):
        return


class Transformer(BaseTransformer):
    """ Class that can be used to register a sequence of transformations and
    apply them to some data.
    """
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
        return apply_chained_transforms(data, self.transforms, *args, **kwargs)


def apply_chained_transforms(data, transforms, *args, **kwargs):
    """ Function to apply a series of transforms to some data.

    Parameters
    ----------
    data: array (N, ) or (n_channels, N)
        the input data.
    transforms: list of BaseTransformer.Transform
        list of transforms to apply.

    Returns
    -------
    _data: array (N, ) or (n_channels, N)
        the transformed input data.
    """
    ndim = data.ndim
    assert ndim in (1, 2)
    _data = data.copy()
    if ndim == 1:
        _data = _data[np.newaxis]
    all_c_data = []
    for _c_data in _data:
        for trf in transforms:
            if np.random.rand() < trf.probability:
                _c_data = trf.transform(_c_data, *args, **kwargs)
            if not trf.randomize_per_channel:
                trf.transform.writable = False
        all_c_data.append(_c_data)
    for trf in transforms:
        trf.transform.writable = True
    _data = np.array(all_c_data)
    return _data.squeeze()


def multichannel_augmentation(augmentation, randomize_per_channel=True):
    """ Decorator to transform an augmentation to a multichannel one.

    Parameters
    ----------
    augmentation: RandomAugmentation class
        the augmentation class.
    randomize_per_channel: bool, default True
        optionnaly randomizes the augmentation parameter for each channel.

    Returns
    -------
    MultiChannelAugmentation: child class of augmentation
        augmentation applicable to multi channel data.
    """
    class MultiChannelAugmentation(augmentation):

        def __call__(self, data, *args, **kwargs):
            """ Function to apply a series of transforms to some data.

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
            if ndim == 1:
                _data = _data[np.newaxis]
            all_c_data = []
            for _c_data in _data:
                _c_data = super().__call__(_c_data, *args, **kwargs)
                if not randomize_per_channel:
                    self.writable = False
                all_c_data.append(_c_data)
            self.writable = True
            _data = np.array(all_c_data)
            return _data.squeeze()

    return MultiChannelAugmentation


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
    if isinstance(data, (list, tuple)):
        return data
    else:
        return [data]
