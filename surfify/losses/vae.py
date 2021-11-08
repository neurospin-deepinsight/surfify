# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Definition of the Cortical Spherical Variational Auto-Encoder (SVAE) loss.
"""

# Imports
import torch
from torch.nn import functional as func
from torch.distributions import Normal, kl_divergence


def img_log_likelihood(recon, xs):
    """ Computes the log likelihood of the input image given the reconstructed
    image for each subject

    Parameters
    ----------
    recon: Tensor (N, C, H, W)
        reconstructed images
    xs: Tensor (N, C, H, W)
        original images

    Returns
    -------
    log_likelihoods: Tensor (N)
        log likelihood for each image
    """
    return -Normal(recon, torch.ones_like(recon)).log_prob(xs).sum(
        dim=(1, 2, 3))


class SphericalVAELoss(object):
    """ Spherical VAE Loss.
    """
    def __init__(self, beta=9, left_mask=None, right_mask=None, use_mse=True):
        """ Init class.

        Parameters
        ----------
        beta: float, default 9
            weight of the kl divergence.
        left_mask: Tensor (azimuth, elevation), default None
            left cortical binary mask.
        right_mask: Tensor (azimuth, elevation), default None
            right cortical binary mask.
        use_mse: bool, default True
            optionally uses the log likelihood.
        """
        super(SphericalVAELoss, self).__init__()
        self.beta = beta
        self.left_mask = left_mask
        self.right_mask = right_mask
        self.layer_outputs = None
        self.use_mse = use_mse

    def __call__(self, left_recon_x, right_recon_x, left_x, right_x):
        """ Compute loss.
        """
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        q = self.layer_outputs["q"]
        z = self.layer_outputs["z"]
        if self.left_mask is None:
            device = left_x.device
            self.left_mask = torch.ones(
                (left_x.shape[-2], left_x.shape[-1]), dtype=int).to(device)
            self.right_mask = torch.ones(
                (right_x.shape[-2], right_x.shape[-1]), dtype=int).to(device)

        # Reconstruction loss terms
        if self.use_mse:
            left_recon_loss = func.mse_loss(
                left_recon_x * self.left_mask.detach(),
                left_x * self.left_mask.detach(), reduction="mean")
            right_recon_loss = func.mse_loss(
                right_recon_x * self.right_mask.detach(),
                right_x * self.right_mask.detach(), reduction="mean")
        else:
            left_recon_loss = img_log_likelihood(
                left_recon_x * self.left_mask.detach(),
                left_x * self.left_mask.detach()).mean()
            right_recon_loss = img_log_likelihood(
                right_recon_x * self.right_mask.detach(),
                right_x * self.right_mask.detach()).mean()

        # Latent loss between approximate posterior and prior for z
        kl_div = kl_divergence(q, Normal(0, 1)).mean(dim=0).sum()

        # Need to maximise the ELBO with respect to these weights
        loss = left_recon_loss + right_recon_loss + self.beta * kl_div

        return loss, {"left_recon_loss": left_recon_loss,
                      "right_recon_loss": right_recon_loss,
                      "kl_div": kl_div, "beta": self.beta}
