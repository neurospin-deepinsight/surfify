##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Methods to decalre transforms.
"""


from torchvision.transforms import Compose, RandomApply


class RandomCompose:
    """ Composes several transforms together.
    """
    def __init__(self, **transforms):
        """ Init.

        Parameters
        ----------
        transforms: dict-like object
            a set of transformations define using the 'transformation' and
            'probability' keys. A suffix '_{order}' where 'order' stands for
            application order is mandatory.
        """
        transform_split = sorted([
            int(name.split("_")[1])
            for name in transforms if name.startswith("transform_")
        ])
        probability_split = sorted([
            int(name.split("_")[1]) for name in transforms
            if name.startswith("probability_")])
        assert len(transform_split) == len(probability_split), (
            "A transform and probability must be declared for each transform!")
        assert len(transform_split) > 0, "No transformation declared!"
        assert transform_split == probability_split, "Order mismatch!"
        self.transforms = [RandomApply([
            transforms[f"transform_{idx}"]],
            p=transforms[f"probability_{idx}"]) for idx in transform_split]
        self.compose = Compose(self.transforms)

    def __call__(self, data):
        return self.compose(data)
