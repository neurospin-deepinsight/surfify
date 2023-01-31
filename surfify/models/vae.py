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

    Examples
    --------
    >>> import torch
    >>> from surfify.utils import icosahedron
    >>> from surfify.models import SphericalVAE
    >>> verts, tris = icosahedron(order=6)
    >>> x = torch.zeros((1, 2, len(verts)))
    >>> model = SphericalVAE(
    >>>     input_channels=2, input_order=6, latent_dim=64,
    >>>     conv_flts=[32, 32, 64, 64], conv_mode="DiNe", dine_size=1,
    >>>     fusion_level=2, standard_ico=False")
    >>> print(model)
    >>> out = model(x, x)
    >>> print(out[0].shape, out[1].shape)
    """
    def __init__(self, input_channels=1, input_order=5, latent_dim=64,
                 conv_flts=[32, 32, 64, 64], conv_mode="DiNe", dine_size=1,
                 repa_size=5, repa_zoom=5, dynamic_repa_zoom=False,
                 fusion_level=1, standard_ico=False, cachedir=None):
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
        fusion_level: int, default 1
            at which max pooling level left and right hemisphere data
            are concatenated.
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
        if fusion_level > self.n_layers or fusion_level <= 0:
            raise ValueError("Impossible to use input fusion level with "
                             "'{0}' layers.".format(self.n_layers))
        self.fusion_level = fusion_level

        # define the encoder
        self.enc_left_conv = nn.Sequential()
        self.enc_right_conv = nn.Sequential()
        self.enc_w_conv = nn.Sequential()
        multi_path = True
        input_channels = self.input_channels
        for idx in range(self.n_layers):
            order = self.input_order - idx
            if idx == self.fusion_level:
                multi_path = False
                input_channels *= 2
            if idx != 0:
                pooling = IcoPool(
                    down_neigh_indices=self.ico[order + 1].neighbor_indices,
                    down_indices=self.ico[order + 1].down_indices,
                    pooling_type="mean")
            if idx != 0 and multi_path:
                self.enc_left_conv.add_module("pooling_{0}".format(idx),
                                              pooling)
                self.enc_right_conv.add_module("pooling_{0}".format(idx),
                                               pooling)
            elif idx != 0:
                self.enc_w_conv.add_module("pooling_{0}".format(idx), pooling)
            if multi_path:
                output_channels = int(self.conv_flts[idx] / 2)
                lconv = self.sconv(
                    input_channels, output_channels,
                    self.ico[order].conv_neighbor_indices)
                self.enc_left_conv.add_module("l_enc_{0}".format(idx), lconv)
                rconv = self.sconv(
                    input_channels, output_channels,
                    self.ico[order].conv_neighbor_indices)
                self.enc_right_conv.add_module("r_enc_{0}".format(idx), rconv)
                input_channels = output_channels
            else:
                conv = self.sconv(
                    input_channels, self.conv_flts[idx],
                    self.ico[order].conv_neighbor_indices)
                self.enc_w_conv.add_module("enc_{0}".format(idx), conv)
                input_channels = self.conv_flts[idx]
        self.enc_w_dense = nn.Linear(self.top_final, self.latent_dim * 2)

        # define the decoder
        self.dec_w_dense = nn.Linear(self.latent_dim, self.top_final)
        self.dec_w_conv = nn.Sequential()
        self.dec_left_conv = nn.Sequential()
        self.dec_right_conv = nn.Sequential()
        input_channels = self.conv_flts[self.n_layers - 1]
        if self.fusion_level == self.n_layers:
            multi_path = True
            input_channels = int(input_channels / 2)
        else:
            multi_path = False
        for idx in range(self.n_layers - 1, -1, -1):
            if multi_path:
                if idx == 0:
                    output_channels = self.input_channels
                else:
                    output_channels = int(self.conv_flts[idx - 1] / 2)
                lconv = IcoUpConv(
                    in_feats=input_channels, out_feats=output_channels,
                    up_neigh_indices=self.ico[order].neighbor_indices,
                    down_indices=self.ico[order].down_indices)
                self.dec_left_conv.add_module("l_dec_{0}".format(idx), lconv)
                rconv = IcoUpConv(
                    in_feats=input_channels, out_feats=output_channels,
                    up_neigh_indices=self.ico[order].neighbor_indices,
                    down_indices=self.ico[order].down_indices)
                self.dec_right_conv.add_module("r_dec_{0}".format(idx), rconv)
                input_channels = output_channels
            else:
                conv = IcoUpConv(
                    in_feats=input_channels, out_feats=self.conv_flts[idx - 1],
                    up_neigh_indices=self.ico[order + 1].neighbor_indices,
                    down_indices=self.ico[order + 1].down_indices)
                self.dec_w_conv.add_module("dec_{0}".format(idx), conv)
                input_channels = self.conv_flts[idx - 1]
            if idx == self.fusion_level:
                multi_path = True
                input_channels = int(input_channels / 2)
            order += 1

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
        left_x = self._safe_forward(self.enc_left_conv, left_x,
                                    act=self.relu, skip_last_act=True)
        right_x = self._safe_forward(self.enc_right_conv, right_x,
                                     act=self.relu, skip_last_act=True)
        x = torch.cat((left_x, right_x), dim=1)
        x = self.relu(x)
        x = self._safe_forward(self.enc_w_conv, x, act=self.relu)
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
        x = self._safe_forward(self.dec_w_conv, x, act=self.relu)
        left_recon_x, right_recon_x = torch.chunk(x, chunks=2, dim=1)
        left_recon_x = self._safe_forward(self.dec_left_conv, left_recon_x,
                                          act=self.relu, skip_last_act=True)
        right_recon_x = self._safe_forward(self.dec_right_conv, right_recon_x,
                                           act=self.relu, skip_last_act=True)
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

    Examples
    --------
    >>> import torch
    >>> from surfify.models import SphericalGVAE
    >>> x = torch.zeros((1, 2, 192, 192))
    >>> model = SphericalGVAE(
    >>>     input_channels=2, input_dim=192, latent_dim=64,
    >>>     conv_flts=[64, 128, 128, 256, 256], fusion_level=2)
    >>> print(model)
    >>> out = model(x, x)
    >>> print(out[0].shape, out[1].shape)
    """
    def __init__(self, input_channels=1, input_dim=192, latent_dim=64,
                 conv_flts=[64, 128, 128, 256, 256], fusion_level=1):
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
        fusion_level: int, default 1
            at which max pooling level left and right hemisphere data
            are concatenated.
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
        if fusion_level > self.n_layers or fusion_level <= 0:
            raise ValueError("Impossible to use input fusion level with "
                             "'{0}' layers.".format(self.n_layers))
        self.fusion_level = fusion_level

        # define the encoder
        self.enc_left_conv = nn.Sequential()
        self.enc_right_conv = nn.Sequential()
        self.enc_w_conv = nn.Sequential()
        multi_path = True
        input_channels = self.input_channels
        for idx in range(self.n_layers):
            if idx == self.fusion_level:
                multi_path = False
                input_channels *= 2
            if multi_path:
                output_channels = int(self.conv_flts[idx] / 2)
                if idx == 0:
                    kernel_size = 8
                    pad = 3
                else:
                    kernel_size = 4
                    pad = 1
                lconv = IcoSpMaConv(
                    in_feats=input_channels, out_feats=output_channels,
                    kernel_size=kernel_size, stride=2, pad=pad)
                self.enc_left_conv.add_module("l_enc_{0}".format(idx), lconv)
                rconv = IcoSpMaConv(
                    in_feats=input_channels, out_feats=output_channels,
                    kernel_size=kernel_size, stride=2, pad=pad)
                self.enc_right_conv.add_module("r_enc_{0}".format(idx), rconv)
                input_channels = output_channels
            else:
                conv = IcoSpMaConv(
                    input_channels, self.conv_flts[idx], kernel_size=4,
                    stride=2, pad=1)
                self.enc_w_conv.add_module("enc_{0}".format(idx), conv)
                input_channels = self.conv_flts[idx]
        self.enc_w_dense = nn.Linear(self.top_final, self.latent_dim * 2)

        # define the decoder
        self.dec_w_dense = nn.Linear(self.latent_dim, self.top_final)
        self.dec_w_conv = nn.Sequential()
        self.dec_left_conv = nn.Sequential()
        self.dec_right_conv = nn.Sequential()
        input_channels = self.conv_flts[self.n_layers - 1]
        if self.fusion_level == self.n_layers:
            multi_path = True
            input_channels = int(input_channels / 2)
        else:
            multi_path = False
        for idx in range(self.n_layers - 1, -1, -1):
            if multi_path:
                if idx == 0:
                    kernel_size = 8
                    pad = 3
                    zero_pad = 9
                    output_channels = self.input_channels
                else:
                    kernel_size = 4
                    pad = 1
                    zero_pad = 3
                    output_channels = int(self.conv_flts[idx - 1] / 2)
                lconv = IcoSpMaConvTranspose(
                    in_feats=input_channels, out_feats=output_channels,
                    kernel_size=kernel_size, stride=2, pad=pad,
                    zero_pad=zero_pad)
                self.dec_left_conv.add_module("l_dec_{0}".format(idx), lconv)
                rconv = IcoSpMaConvTranspose(
                    in_feats=input_channels, out_feats=output_channels,
                    kernel_size=kernel_size, stride=2, pad=pad,
                    zero_pad=zero_pad)
                self.dec_right_conv.add_module("r_dec_{0}".format(idx), rconv)
                input_channels = output_channels
            else:
                conv = IcoSpMaConvTranspose(
                    in_feats=input_channels, out_feats=self.conv_flts[idx - 1],
                    kernel_size=4, stride=2, pad=1, zero_pad=3)
                self.dec_w_conv.add_module("dec_{0}".format(idx), conv)
                input_channels = self.conv_flts[idx - 1]
            if idx == self.fusion_level:
                multi_path = True
                input_channels = int(input_channels / 2)
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
        left_x = self.enc_left_conv(left_x)
        right_x = self.enc_right_conv(right_x)
        x = torch.cat((left_x, right_x), dim=1)
        x = self.relu(x)
        for mod in self.enc_w_conv.children():
            x = self.relu(mod(x))
        x = x.view(-1, self.top_final)
        x = self.enc_w_dense(x)
        z_mu, z_logvar = torch.chunk(x, chunks=2, dim=1)
        return Normal(loc=z_mu, scale=torch.exp(0.5 * z_logvar))

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
        for mod in self.dec_w_conv.children():
            x = self.relu(mod(x))
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
