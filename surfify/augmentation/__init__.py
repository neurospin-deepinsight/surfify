# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Surface augmentations.
"""

# Imports
<<<<<<< HEAD
from .base import SurfCutOut, SurfNoise, SurfRotation, SurfBlur
from .mixup import HemiMixUp, GroupMixUp
from .utils import interval, Transformer
=======
from .augmentation import (SphericalRandomCut, SphericalRandomRotation,
                           SphericalBlur, SphericalNoise)
>>>>>>> b088a8b (lots of stuff : spherical and grided vae, new augmentations, plot new augmentations, new function to compute the right number of ring for surf cutout)
