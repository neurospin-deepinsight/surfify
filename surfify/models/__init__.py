# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common architectures.
"""

from .base import SphericalBase
from .unet import SphericalUNet, SphericalGUNet
from .vae import (
    SphericalVAE, HemiFusionEncoder, HemiFusionDecoder,
    SphericalHemiFusionEncoder, SphericalHemiFusionDecoder)
from .vgg import (
    SphericalVGG, SphericalGVGG,
    SphericalVGG11, SphericalVGG13, SphericalVGG16, SphericalVGG19,
    SphericalVGG11BN, SphericalVGG13BN, SphericalVGG16BN, SphericalVGG19BN,
    SphericalGVGG11, SphericalGVGG13, SphericalGVGG16, SphericalGVGG19,
    SphericalGVGG11BN, SphericalGVGG13BN, SphericalGVGG16BN, SphericalGVGG19BN)
