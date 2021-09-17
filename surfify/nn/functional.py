# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Suface utilities.
"""

# Imports
import torch.nn.functional as F


def circular_pad(x, pad):
    """ Circular pad of a tensor.

    Since a spherical patterns are circularly continuous with respect to the
    azimuth, we need to apply circular padding to the boundaries of azimuth for
    the flattened 2-D map but applied zero padding to the boundaries of
    evaluation.

    Parameters
    ----------
    x: Tensor (samples, channels, azimuth, elevation)
        input tensor.
    pad: int or tuple (pad_azimuth, pad_elevation)
        the size of the padding.
    """
    if not isinstance(pad, list) and not isinstance(pad, tuple):
        pad = [pad, pad]
    x = F.pad(x, (pad[1], pad[1], 0, 0), "constant", 0)
    x = F.pad(x, (0, 0, pad[0], pad[0]), "circular")
    return x
