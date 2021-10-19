# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Cortical Spherical Variational Auto-Encoder (GMVAE) models.

[1] Representation Learning of Resting State fMRI with Variational
Autoencoder: https://github.com/libilab/rsfMRI-VAE
"""

# Imports
import torch
import torch.nn as nn
from torch.distributions import Normal
from ..utils import get_logger, debug_msg
from ..nn import IcoUpConv, IcoPool, IcoSpMaConv, IcoSpMaConvTranspose
from .base import SphericalBase


# Global parameters
logger = get_logger()


class SphericalVAE(SphericalBase):
    """ Spherical VAE architecture.

    Use either RePa - Rectangular Patch convolution method or DiNe - Direct
    Neighbor convolution method.

    Notes
    -----
    Debuging messages can be displayed by changing the log level using
    ``setup_logging(level='debug')``.

    See Also
    --------
    SphericalGVAE

    References
    ----------
    Representation Learning of Resting State fMRI with Variational
    Autoencoder, NeuroImage 2021.
    """
    def __init__(self, input_channels=1, input_order=5, latent_dim=64,
                 conv_flts=[32, 32, 64, 64], conv_mode="DiNe", dine_size=1,
                 repa_size=5, repa_zoom=5, dynamic_repa_zoom=False,
                 standard_ico=False, cachedir=None):
        """ Init class.

        Parameters
        ----------
        input_channels: int, default 1
            the number of input channels.
        input_order: int, default 5
            the input icosahedron order.
        latent_dim: int, default 64
            the size of the stochastic latent state of the SVAE.
        conv_flts: list of int
            the size of convolutional filters.
        conv_mode: str, default 'DiNe'
            use either 'RePa' - Rectangular Patch convolution method or 'DiNe'
            - 1 ring Direct Neighbor convolution method.
        dine_size: int, default 1
            the size of the spherical convolution filter, ie. the number of
            neighbor rings to be considered.
        repa_size: int, default 5
            the size of the rectangular grid in the tangent space.
        repa_zoom: int, default 5
            control the rectangular grid spacing in the tangent space by
            applying a multiplicative factor of `1 / repa_zoom`.
        dynamic_repa_zoom: bool, default False
            dynamically adapt the RePa zoom by applying a multiplicative factor
            of `log(order + 1) + 1`.
        standard_ico: bool, default False
            optionaly use surfify tesselation.
        cachedir: str, default None
            set this folder to use smart caching speedup.
        """
        logger.debug("SphericalVAE init...")
        super(SphericalVAE, self).__init__(
            input_order=input_order, n_layers=len(conv_flts),
            conv_mode=conv_mode, dine_size=dine_size, repa_size=repa_size,
            repa_zoom=repa_zoom, dynamic_repa_zoom=dynamic_repa_zoom,
            standard_ico=standard_ico, cachedir=cachedir)
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.conv_flts = conv_flts
        self.top_flatten_dim = len(
            self.ico[self.input_order - self.n_layers + 1].vertices)
        self.top_final = self.conv_flts[-1] * self.top_flatten_dim

        # define the encoder
        self.enc_left_conv = self.sconv(
            input_channels, int(self.conv_flts[0] / 2),
            self.ico[self.input_order].conv_neighbor_indices)
        self.enc_right_conv = self.sconv(
            input_channels, int(self.conv_flts[0] / 2),
            self.ico[self.input_order].conv_neighbor_indices)
        self.enc_w_conv = nn.Sequential()
        for idx in range(1, self.n_layers):
            order = self.input_order - idx
            pooling = IcoPool(
                down_neigh_indices=self.ico[order + 1].neighbor_indices,
                down_indices=self.ico[order + 1].down_indices,
                pooling_type="mean")
            self.enc_w_conv.add_module("pooling_{0}".format(idx), pooling)
            conv = self.sconv(
                self.conv_flts[idx - 1], self.conv_flts[idx],
                self.ico[order].conv_neighbor_indices)
            self.enc_w_conv.add_module("down_{0}".format(idx), conv)
        self.enc_w_dense = nn.Linear(self.top_final, self.latent_dim * 2)

        # define the decoder
        self.dec_w_dense = nn.Linear(self.latent_dim, self.top_final)
        self.dec_w_conv = nn.Sequential()
        cnt = 1
        for idx in range(self.n_layers - 1, 0, -1):
            tconv = IcoUpConv(
                in_feats=self.conv_flts[idx],
                out_feats=self.conv_flts[idx - 1],
                up_neigh_indices=self.ico[order + 1].neighbor_indices,
                down_indices=self.ico[order + 1].down_indices)
            self.dec_w_conv.add_module("up_{0}".format(cnt), tconv)
            order += 1
            cnt += 1
        self.dec_left_conv = IcoUpConv(
            in_feats=int(self.conv_flts[0] / 2),
            out_feats=self.input_channels,
            up_neigh_indices=self.ico[order].neighbor_indices,
            down_indices=self.ico[order].down_indices)
        self.dec_right_conv = IcoUpConv(
            in_feats=int(self.conv_flts[0] / 2),
            out_feats=self.input_channels,
            up_neigh_indices=self.ico[order].neighbor_indices,
            down_indices=self.ico[order].down_indices)

        self.relu = nn.ReLU(inplace=True)

    def encode(self, left_x, right_x):
        """ The encoder.

        Parameters
        ----------
        left_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input left cortical texture.
        right_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input right cortical texture.

        Returns
        -------
        q(z | x): Normal (batch_size, <latent_dim>)
            a Normal distribution.
        """
        x = torch.cat(
            (self.enc_left_conv(left_x), self.enc_right_conv(right_x)), dim=1)
        x = self.relu(x)
        for layer_idx in range((self.n_layers - 1) * 2):
            if isinstance(self.enc_w_conv[layer_idx], IcoPool):
                x = self.enc_w_conv[layer_idx](x)[0]
            else:
                x = self.relu(self.enc_w_conv[layer_idx](x))
        x = x.reshape(-1, self.top_final)
        x = self.enc_w_dense(x)
        z_mu, z_logvar = torch.chunk(x, chunks=2, dim=1)
        return Normal(loc=z_mu, scale=z_logvar.exp().pow(0.5))

    def decode(self, z):
        """ The decoder.

        Parameters
        ----------
        z: Tensor (samples, <latent_dim>)
            the stochastic latent state z.

        Returns
        -------
        left_recon_x: Tensor (samples, <input_channels>, azimuth, elevation)
            reconstructed left cortical texture.
        right_recon_x: Tensor (samples, <input_channels>, azimuth, elevation)
            reconstructed right cortical texture.
        """
        x = self.relu(self.dec_w_dense(z))
        x = x.view(-1, self.conv_flts[-1], self.top_flatten_dim)
        for layer_idx in range(self.n_layers - 1):
            x = self.relu(self.dec_w_conv[layer_idx](x))
        left_recon_x, right_recon_x = torch.chunk(x, chunks=2, dim=1)
        left_recon_x = self.dec_left_conv(left_recon_x)
        right_recon_x = self.dec_right_conv(right_recon_x)
        return left_recon_x, right_recon_x

    def reparameterize(self, q):
        """ Implement the reparametrization trick.
        """
        if self.training:
            z = q.rsample()
        else:
            z = q.loc
        return z

    def forward(self, left_x, right_x):
        """ The forward method.

        Parameters
        ----------
        left_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input left cortical texture.
        right_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input right cortical texture.

        Returns
        -------
        left_recon_x: Tensor (samples, <input_channels>, azimuth, elevation)
            reconstructed left cortical texture.
        right_recon_x: Tensor (samples, <input_channels>, azimuth, elevation)
            reconstructed right cortical texture.
        """
        logger.debug("SphericalVAE forward pass")
        logger.debug(debug_msg("left cortical", left_x))
        logger.debug(debug_msg("right cortical", right_x))
        q = self.encode(left_x, right_x)
        logger.debug(debug_msg("posterior loc", q.loc))
        logger.debug(debug_msg("posterior scale", q.scale))
        z = self.reparameterize(q)
        logger.debug(debug_msg("z", z))
        left_recon_x, right_recon_x = self.decode(z)
        logger.debug(debug_msg("left recon cortical", left_recon_x))
        logger.debug(debug_msg("right recon cortical", right_recon_x))
        return left_recon_x, right_recon_x, {"q": q, "z": z}


class SphericalGVAE(nn.Module):
    """ Spherical Grided VAE architecture.

    Use SpMa - Spherical Mapping convolution method.

    Notes
    -----
    Debuging messages can be displayed by changing the log level using
    ``setup_logging(level='debug')``.

    See Also
    --------
    SphericalVAE

    References
    ----------
    Representation Learning of Resting State fMRI with Variational
    Autoencoder, NeuroImage 2021.
    """
    def __init__(self, input_channels=1, input_dim=192, latent_dim=64,
                 conv_flts=[64, 128, 128, 256, 256]):
        """ Init class.

        Parameters
        ----------
        input_channels: int, default 1
            the number of input channels.
        input_dim: int, default 192
            the size of the converted 3-D surface to the 2-D grid.
        latent_dim: int, default 64
            the size of the stochastic latent state of the SVAE.
        conv_flts: list of int
            the size of convolutional filters.
        """
        logger.debug("SphericalGVAE init...")
        super(SphericalGVAE, self).__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv_flts = conv_flts
        self.n_layers = len(self.conv_flts)
        self.top_flatten_dim = int(self.input_dim / (2 ** self.n_layers))
        self.top_final = self.conv_flts[-1] * self.top_flatten_dim ** 2

        # define the encoder
        self.enc_left_conv = IcoSpMaConv(
            in_feats=self.input_channels,
            out_feats=int(self.conv_flts[0] / 2),
            kernel_size=8, stride=2, pad=3)
        self.enc_right_conv = IcoSpMaConv(
            in_feats=self.input_channels,
            out_feats=int(self.conv_flts[0] / 2),
            kernel_size=8, stride=2, pad=3)
        self.enc_w_conv = nn.ModuleList([
            IcoSpMaConv(self.conv_flts[i - 1], self.conv_flts[i],
                        kernel_size=4, stride=2, pad=1)
            for i in range(1, self.n_layers)])
        self.enc_w_dense = nn.Linear(self.top_final, self.latent_dim * 2)

        # define the decoder
        self.dec_w_dense = nn.Linear(self.latent_dim, self.top_final)
        self.dec_w_conv = nn.ModuleList([
            IcoSpMaConvTranspose(
                in_feats=self.conv_flts[i],
                out_feats=self.conv_flts[i - 1],
                kernel_size=4, stride=2, pad=1, zero_pad=3)
            for i in range(self.n_layers - 1, 0, -1)])
        self.dec_left_conv = IcoSpMaConvTranspose(
            in_feats=int(self.conv_flts[0] / 2),
            out_feats=self.input_channels,
            kernel_size=8, stride=2, pad=3, zero_pad=9)
        self.dec_right_conv = IcoSpMaConvTranspose(
            in_feats=int(self.conv_flts[0] / 2),
            out_feats=self.input_channels,
            kernel_size=8, stride=2, pad=3, zero_pad=9)

        self.relu = nn.ReLU(inplace=True)

    def encode(self, left_x, right_x):
        """ The encoder.

        Parameters
        ----------
        left_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input left cortical texture.
        right_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input right cortical texture.

        Returns
        -------
        q(z | x): Normal (batch_size, <latent_dim>)
            a Normal distribution.
        """
        x = torch.cat(
            (self.enc_left_conv(left_x), self.enc_right_conv(right_x)), dim=1)
        x = self.relu(x)
        for layer_idx in range(self.n_layers - 1):
            x = self.relu(self.enc_w_conv[layer_idx](x))
        x = x.view(-1, self.top_final)
        x = self.enc_w_dense(x)
        z_mu, z_logvar = torch.chunk(x, chunks=2, dim=1)
        return Normal(loc=z_mu, scale=z_logvar.exp().pow(0.5))

    def decode(self, z):
        """ The decoder.

        Parameters
        ----------
        z: Tensor (samples, <latent_dim>)
            the stochastic latent state z.

        Returns
        -------
        left_recon_x: Tensor (samples, <input_channels>, azimuth, elevation)
            reconstructed left cortical texture.
        right_recon_x: Tensor (samples, <input_channels>, azimuth, elevation)
            reconstructed right cortical texture.
        """
        x = self.relu(self.dec_w_dense(z))
        x = x.view(-1, self.conv_flts[-1], self.top_flatten_dim,
                   self.top_flatten_dim)
        for layer_idx in range(self.n_layers - 1):
            x = self.relu(self.dec_w_conv[layer_idx](x))
        left_recon_x, right_recon_x = torch.chunk(x, chunks=2, dim=1)
        left_recon_x = self.dec_left_conv(left_recon_x)
        right_recon_x = self.dec_right_conv(right_recon_x)
        return left_recon_x, right_recon_x

    def reparameterize(self, q):
        """ Implement the reparametrization trick.
        """
        if self.training:
            z = q.rsample()
        else:
            z = q.loc
        return z

    def forward(self, left_x, right_x):
        """ The forward method.

        Parameters
        ----------
        left_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input left cortical texture.
        right_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input right cortical texture.

        Returns
        -------
        left_recon_x: Tensor (samples, <input_channels>, azimuth, elevation)
            reconstructed left cortical texture.
        right_recon_x: Tensor (samples, <input_channels>, azimuth, elevation)
            reconstructed right cortical texture.
        """
        logger.debug("SphericalGVAE forward pass")
        logger.debug(debug_msg("left cortical", left_x))
        logger.debug(debug_msg("right cortical", right_x))
        q = self.encode(left_x, right_x)
        logger.debug(debug_msg("posterior loc", q.loc))
        logger.debug(debug_msg("posterior scale", q.scale))
        z = self.reparameterize(q)
        logger.debug(debug_msg("z", z))
        left_recon_x, right_recon_x = self.decode(z)
        logger.debug(debug_msg("left recon cortical", left_recon_x))
        logger.debug(debug_msg("right recon cortical", right_recon_x))
        return left_recon_x, right_recon_x, {"q": q, "z": z}
