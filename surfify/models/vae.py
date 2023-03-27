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
                 conv_flts=[32, 32, 64, 64], fusion_level=1,
                 activation="LeakyReLU", batch_norm=False, conv_mode="DiNe",
                 dine_size=1, repa_size=5, repa_zoom=5,
                 dynamic_repa_zoom=False, standard_ico=False, cachedir=None,
                 encoder=None, decoder=None):
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
        super().__init__(
            input_order=input_order, n_layers=len(conv_flts),
            conv_mode=conv_mode, dine_size=dine_size, repa_size=repa_size,
            repa_zoom=repa_zoom, dynamic_repa_zoom=dynamic_repa_zoom,
            standard_ico=standard_ico, cachedir=cachedir)
        self.encoder = encoder or SphericalHemiFusionEncoder(
            input_channels, input_order, latent_dim * 2, conv_flts,
            fusion_level, activation, batch_norm, conv_mode, dine_size,
            repa_size, repa_zoom, dynamic_repa_zoom, standard_ico, cachedir)
        self.decoder = decoder or SphericalHemiFusionDecoder(
            input_channels, input_order, latent_dim, conv_flts, fusion_level,
            activation, batch_norm, conv_mode, dine_size, repa_size, repa_zoom,
            dynamic_repa_zoom, standard_ico, cachedir)

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
        x = self.encoder((left_x, right_x))
        z_mu, z_logvar = torch.chunk(x, chunks=2, dim=1)
        return Normal(loc=z_mu, scale=(z_logvar * 0.5).exp())

    def decode(self, z):
        """ The decoder.

        Parameters
        ----------
        z: Tensor (samples, <latent_dim>)
            the stochastic latent state z.

        Returns
        -------
        left_recon_x: Tensor (samples, <input_channels>, n_vertices)
            reconstructed left cortical texture.
        right_recon_x: Tensor (samples, <input_channels>, n_vertices)
            reconstructed right cortical texture.
        """
        left_recon_x, right_recon_x = self.decoder(z)
        return left_recon_x, right_recon_x

    def reparameterize(self, q, sample=True):
        """ Implement the reparametrization trick.
        """
        if sample:
            return q.rsample()
        return q.loc

    def forward(self, left_x, right_x, sample=True):
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
        z = self.reparameterize(q, sample)
        logger.debug(debug_msg("z", z))
        left_recon_x, right_recon_x = self.decode(z)
        logger.debug(debug_msg("left recon cortical", left_recon_x))
        logger.debug(debug_msg("right recon cortical", right_recon_x))
        return left_recon_x, right_recon_x, {"q": q, "z": z}


class SphericalHemiFusionEncoder(SphericalBase):
    def __init__(self, input_channels, input_order, latent_dim,
                 conv_flts=[64, 128, 128, 256, 256], fusion_level=1,
                 activation="LeakyReLU", batch_norm=False,
                 conv_mode="DiNe", dine_size=1, repa_size=5, repa_zoom=5,
                 dynamic_repa_zoom=False, standard_ico=False, cachedir=None):
        """ Init class.

        Parameters
        ----------
        input_channels: int, default 1
            the number of input channels.
        input_dim: int, default 192
            the size of the converted 3-D surface to the 2-D grid.
        latent_dim: int, default 64
            the size of the latent space it encodes to.
        conv_flts: list of int
            the size of convolutional filters.
        fusion_level: int, default 1
            at which max pooling level left and right hemisphere data
            are concatenated.
        activation: str, default 'LeakyReLU'
            activation function's class name in pytorch's nn module to use
            after each convolution
        batch_norm: bool, default False
            optionally uses batch normalization after each convolution
        """
        logger.debug("SphericalHemiFusionEncoder init...")
        super().__init__(
            input_order=input_order, n_layers=len(conv_flts),
            conv_mode=conv_mode, dine_size=dine_size, repa_size=repa_size,
            repa_zoom=repa_zoom, dynamic_repa_zoom=dynamic_repa_zoom,
            standard_ico=standard_ico, cachedir=cachedir)
        self.input_channels = input_channels
        self.conv_flts = conv_flts
        self.activation = getattr(nn, activation)(inplace=True)
        self.n_vertices_down = len(
            self.ico[self.input_order - self.n_layers].vertices)
        logger.debug("  number of vertices small ico : {}".format(
            self.n_vertices_down))
        self.flatten_dim = conv_flts[-1] * self.n_vertices_down
        logger.debug("  dimension for linear {}".format(self.flatten_dim))
        if fusion_level > self.n_layers or fusion_level <= 0:
            raise ValueError("Impossible to use input fusion level with "
                             "'{0}' layers.".format(self.n_layers))
        self.fusion_level = fusion_level
        self.latent_dim = latent_dim
        self.left_conv = nn.Sequential()
        self.right_conv = nn.Sequential()
        self.w_conv = nn.Sequential()
        input_channels = self.input_channels
        for idx in range(self.n_layers):
            order = self.input_order - idx
            output_channels = self.conv_flts[idx]
            pooling = IcoPool(
                down_neigh_indices=self.ico[order].neighbor_indices,
                down_indices=self.ico[order].down_indices,
                pooling_type="mean")
            if idx < fusion_level:
                output_channels = int(output_channels / 2)
                lconv = self.sconv(
                    input_channels, output_channels,
                    self.ico[order].conv_neighbor_indices)
                self.left_conv.add_module("l_conv_{0}".format(idx), lconv)
                if batch_norm:
                    self.left_conv.add_module(
                        "l_bn_{0}".format(idx),
                        nn.BatchNorm1d(output_channels))
                self.left_conv.add_module("pooling_{0}".format(idx), pooling)
                rconv = self.sconv(
                    input_channels, output_channels,
                    self.ico[order].conv_neighbor_indices)
                self.right_conv.add_module("r_conv_{0}".format(idx), rconv)
                if batch_norm:
                    self.right_conv.add_module(
                        "r_bn_{0}".format(idx),
                        nn.BatchNorm1d(output_channels))
                self.right_conv.add_module("pooling_{0}".format(idx), pooling)
                input_channels = output_channels
            else:
                input_channels = self.conv_flts[idx - 1]
                conv = self.sconv(
                    input_channels, output_channels,
                    self.ico[order].conv_neighbor_indices)
                self.w_conv.add_module("conv_{0}".format(idx), conv)
                if batch_norm:
                    self.w_conv.add_module(
                        "bn_{0}".format(idx),
                        nn.BatchNorm1d(self.conv_flts[idx]))
                self.w_conv.add_module("pooling_{0}".format(idx), pooling)
        self.w_dense = nn.Linear(self.flatten_dim, self.latent_dim)

    def forward(self, x):
        """ The encoding.

        Parameters
        ----------
        left_x: Tensor (batch_size, <input_channels>, n_vertices)
            input left cortical textures.
        right_x: Tensor (batch_size, <input_channels>, n_vertices)
            input right cortical textures.

        Returns
        -------
        x: Tensor (batch_size, <latent_dim>)
            the latent representations.
        """
        left_x, right_x = x
        logger.debug("SphericalGVAE forward pass")
        logger.debug(debug_msg("  left cortical", left_x))
        logger.debug(debug_msg("  right cortical", right_x))
        left_x = self._safe_forward(
            self.left_conv, left_x, act=self.activation, skip_last_act=True)
        right_x = self._safe_forward(
            self.right_conv, right_x, act=self.activation, skip_last_act=True)
        logger.debug(debug_msg("  left enc", left_x))
        logger.debug(debug_msg("  right enc", right_x))
        x = torch.cat((left_x, right_x), dim=1)
        x = self.activation(x)
        logger.debug(debug_msg("  merged enc", x))
        x = self._safe_forward(self.w_conv, x, act=self.activation)
        logger.debug(debug_msg("  final conv enc", x))
        x = x.view(-1, self.flatten_dim)
        logger.debug(debug_msg("  flattened", x))
        x = self.w_dense(x)
        return x


class SphericalHemiFusionDecoder(SphericalBase):
    def __init__(self, input_channels, input_order, latent_dim,
                 conv_flts=[64, 128, 128, 256, 256], fusion_level=1,
                 activation="LeakyReLU", batch_norm=False,
                 conv_mode="DiNe", dine_size=1, repa_size=5, repa_zoom=5,
                 dynamic_repa_zoom=False, standard_ico=False, cachedir=None):
        """ Init class.

        Parameters
        ----------
        input_channels: int, default 1
            the number of input channels.
        input_dim: int, default 192
            the size of the converted 3-D surface to the 2-D grid.
        latent_dim: int, default 64
            the size of the latent space it encodes to.
        conv_flts: list of int
            the size of convolutional filters.
        fusion_level: int, default 1
            at which max pooling level left and right hemisphere data
            are concatenated.
        activation: str, default 'LeakyReLU'
            activation function's class name in pytorch's nn module to use
            after each convolution
        batch_norm: bool, default False
            optionally uses batch normalization after each convolution
        """
        logger.debug("SphericalHemiFusionDecoder init...")
        super().__init__(
            input_order=input_order, n_layers=len(conv_flts),
            conv_mode=conv_mode, dine_size=dine_size, repa_size=repa_size,
            repa_zoom=repa_zoom, dynamic_repa_zoom=dynamic_repa_zoom,
            standard_ico=standard_ico, cachedir=cachedir)
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.conv_flts = conv_flts

        self.activation = getattr(nn, activation)(inplace=True)
        self.n_vertices_down = len(
            self.ico[self.input_order - self.n_layers].vertices)
        logger.debug("  number of vertices small ico : {}".format(
            self.n_vertices_down))
        self.flatten_dim = conv_flts[-1] * self.n_vertices_down
        logger.debug("  dimension for linear {}".format(self.flatten_dim))
        if fusion_level > self.n_layers or fusion_level <= 0:
            raise ValueError("Impossible to use input fusion level with "
                             "'{0}' layers.".format(self.n_layers))
        self.fusion_level = fusion_level
        self.latent_dim = latent_dim

        self.w_dense = nn.Linear(self.latent_dim, self.flatten_dim)
        self.w_conv = nn.Sequential()
        self.left_conv = nn.Sequential()
        self.right_conv = nn.Sequential()
        input_channels = self.conv_flts[self.n_layers - 1]
        for idx in range(self.n_layers - 1, -1, -1):
            order = self.input_order - idx - 1
            input_channels = self.conv_flts[idx]
            output_channels = self.input_channels * 2
            if idx != 0:
                output_channels = self.conv_flts[idx - 1]
            if idx < fusion_level:
                output_channels = int(output_channels / 2)
                input_channels = int(input_channels / 2)
                logger.debug("input channels : {}".format(input_channels))
                logger.debug("output channels : {}".format(output_channels))
                l_pooling = IcoUpConv(
                    in_feats=input_channels, out_feats=output_channels,
                    up_neigh_indices=self.ico[order + 1].neighbor_indices,
                    down_indices=self.ico[order + 1].down_indices)
                lconv = self.sconv(
                    output_channels, output_channels,
                    self.ico[order + 1].conv_neighbor_indices)
                self.left_conv.add_module(
                    "l_pooling_{0}".format(idx), l_pooling)
                self.left_conv.add_module("l_conv_{0}".format(idx), lconv)
                if batch_norm:
                    self.left_conv.add_module(
                        "l_bn_{0}".format(idx),
                        nn.BatchNorm1d(output_channels))
                r_pooling = IcoUpConv(
                    in_feats=input_channels, out_feats=output_channels,
                    up_neigh_indices=self.ico[order + 1].neighbor_indices,
                    down_indices=self.ico[order + 1].down_indices)
                rconv = self.sconv(
                    output_channels, output_channels,
                    self.ico[order + 1].conv_neighbor_indices)
                self.right_conv.add_module(
                    "r_pooling_{0}".format(idx), r_pooling)
                self.right_conv.add_module("r_conv_{0}".format(idx), rconv)
                if batch_norm:
                    self.right_conv.add_module(
                        "r_bn_{0}".format(idx),
                        nn.BatchNorm1d(output_channels))
            else:
                logger.debug("input channels : {}".format(input_channels))
                logger.debug("output channels : {}".format(output_channels))
                logger.debug("order : {}".format(order))
                pooling = IcoUpConv(
                    in_feats=input_channels, out_feats=output_channels,
                    up_neigh_indices=self.ico[order + 1].neighbor_indices,
                    down_indices=self.ico[order + 1].down_indices)
                conv = self.sconv(
                    output_channels, output_channels,
                    self.ico[order + 1].conv_neighbor_indices)
                self.w_conv.add_module("pooling_{0}".format(idx), pooling)
                self.w_conv.add_module("conv_{0}".format(idx), conv)
                if batch_norm:
                    self.w_conv.add_module(
                        "bn_{0}".format(idx),
                        nn.BatchNorm1d(self.conv_flts[idx]))

    def forward(self, x):
        """ The decoding.

        Parameters
        ----------
        left_x: Tensor (batch_size, <input_channels>, n_vertices)
            input left cortical textures.
        right_x: Tensor (batch_size, <input_channels>, n_vertices)
            input right cortical textures.

        Returns
        -------
        x: Tensor (batch_size, <latent_dim>)
            the latent representations.
        """
        logger.debug("SphericalHemiFusionDecoder forward pass")
        logger.debug(debug_msg("latent", x))
        x = self.activation(self.w_dense(x))
        x = x.view(-1, self.conv_flts[-1], self.n_vertices_down)
        logger.debug(debug_msg("input to conv", x))
        x = self._safe_forward(self.w_conv, x, act=self.activation)
        logger.debug(debug_msg("before hemi sep", x))
        left_x, right_x = torch.chunk(x, chunks=2, dim=1)
        logger.debug(debug_msg("after hemi sep right", right_x))
        logger.debug(debug_msg("after hemi sep left", left_x))
        left_x = self._safe_forward(self.left_conv, left_x,
                                    act=self.activation, skip_last_act=True)
        right_x = self._safe_forward(self.right_conv, right_x,
                                     act=self.activation, skip_last_act=True)
        return left_x, right_x


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
                 conv_flts=[64, 128, 128, 256, 256], fusion_level=1,
                 activation="LeakyReLU", batch_norm=False,
                 encoder=None, decoder=None):
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
        activation: str, default 'LeakyReLU'
            activation function's class name in pytorch's nn module to use
            after each convolution
        batch_norm: bool, default False
            optionally uses batch normalization after each convolution
        encoder: nn.Module or None, default None
            encoder model to use. If None instantiate a HemiFusionEncoder
            with the provided parameters.
        decoder: nn.Module or None, default None
            decoder model to use. If None instantiate a HemiFusionDecoder
            with the provided parameters.
        """
        logger.debug("SphericalGVAE init...")
        super().__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv_flts = conv_flts
        self.n_layers = len(self.conv_flts)
        if fusion_level > self.n_layers or fusion_level <= 0:
            raise ValueError("Impossible to use input fusion level with "
                             "'{0}' layers.".format(self.n_layers))
        self.fusion_level = fusion_level
        self.encoder = encoder or HemiFusionEncoder(
            input_channels, input_dim, latent_dim * 2, conv_flts,
            fusion_level, activation, batch_norm)
        self.decoder = decoder or HemiFusionDecoder(
            input_channels, self.encoder.output_dim, latent_dim, conv_flts,
            fusion_level, activation, batch_norm)

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
        x = self.encoder((left_x, right_x))
        z_mu, z_logvar = torch.chunk(x, chunks=2, dim=1)
        return Normal(loc=z_mu, scale=(z_logvar * 0.5).exp())

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
        return self.decoder(z)

    def reparameterize(self, q, sample=True):
        """ Implement the reparametrization trick.
        """
        if sample:
            return q.rsample()
        return q.loc

    def forward(self, left_x, right_x, sample=True):
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
        z = self.reparameterize(q, sample)
        logger.debug(debug_msg("z", z))
        left_recon_x, right_recon_x = self.decode(z)
        logger.debug(debug_msg("left recon cortical", left_recon_x))
        logger.debug(debug_msg("right recon cortical", right_recon_x))
        return left_recon_x, right_recon_x, {"q": q, "z": z}


def compute_output_dim(input_dim, convnet):
    """ Compute the output dimension of a convolutional network
    that takes as input a square input (H = W)

    Parameters
    ----------
    input_dim: int
        input height and weight
    convnet: iterable[nn.Module]
        iterable containing the various layers. For now, the function
        can only work with nn.Conv2d and IcoSpMaConv layers

    Returns
    -------
    output_dim: int
        output dimension
    """
    output_dim = input_dim
    for layer in convnet:
        if type(layer) is nn.Conv2d:
            output_dim = int(
                (output_dim + 2 * layer.padding[0] - layer.dilation[0] *
                 (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1)
        elif type(layer) is IcoSpMaConv:
            output_dim = compute_output_dim(
                output_dim + 2 * layer.pad, [layer.conv])
    return output_dim


class HemiFusionEncoder(nn.Module):
    def __init__(self, input_channels, input_dim, latent_dim,
                 conv_flts=[64, 128, 128, 256, 256], fusion_level=1,
                 activation="LeakyReLU", batch_norm=False):
        """ Init class.

        Parameters
        ----------
        input_channels: int, default 1
            the number of input channels.
        input_dim: int, default 192
            the size of the converted 3-D surface to the 2-D grid.
        latent_dim: int, default 64
            the size of the latent space it encodes to.
        conv_flts: list of int
            the size of convolutional filters.
        fusion_level: int, default 1
            at which max pooling level left and right hemisphere data
            are concatenated.
        activation: str, default 'LeakyReLU'
            activation function's class name in pytorch's nn module to use
            after each convolution
        batch_norm: bool, default False
            optionally uses batch normalization after each convolution
        """
        logger.debug("HemiFusionEncoder init...")
        super().__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.conv_flts = conv_flts
        self.n_layers = len(self.conv_flts)
        if fusion_level > self.n_layers or fusion_level <= 0:
            raise ValueError("Impossible to use input fusion level with "
                             "'{0}' layers.".format(self.n_layers))
        self.fusion_level = fusion_level
        self.left_conv = nn.Sequential()
        self.right_conv = nn.Sequential()
        self.w_conv = nn.Sequential()
        input_channels = self.input_channels
        for idx in range(self.n_layers):
            if idx == 0:
                kernel_size = 8
                pad = 3
            else:
                kernel_size = 4
                pad = 1
            output_channels = self.conv_flts[idx]
            if idx < fusion_level:
                output_channels /= 2
                lconv = IcoSpMaConv(
                    in_feats=input_channels, out_feats=int(output_channels),
                    kernel_size=kernel_size, stride=2, pad=pad)
                self.left_conv.add_module("l_conv_{0}".format(idx), lconv)
                if batch_norm:
                    self.left_conv.add_module(
                        "l_bn_{0}".format(idx),
                        nn.BatchNorm2d(int(output_channels)))
                self.left_conv.add_module(
                    "l_act_{0}".format(idx),
                    getattr(nn, activation)(inplace=True))
                rconv = IcoSpMaConv(
                    in_feats=input_channels, out_feats=int(output_channels),
                    kernel_size=kernel_size, stride=2, pad=pad)
                self.right_conv.add_module("r_conv_{0}".format(idx), rconv)
                if batch_norm:
                    self.right_conv.add_module(
                        "r_bn_{0}".format(idx),
                        nn.BatchNorm2d(int(output_channels)))
                self.right_conv.add_module(
                    "r_act_{0}".format(idx),
                    getattr(nn, activation)(inplace=True))
                input_channels = int(output_channels)
            else:
                input_channels = self.conv_flts[idx - 1]
                conv = IcoSpMaConv(
                    in_feats=input_channels, out_feats=self.conv_flts[idx],
                    kernel_size=kernel_size, stride=2, pad=pad)
                self.w_conv.add_module("conv_{0}".format(idx), conv)
                if batch_norm:
                    self.w_conv.add_module(
                        "bn_{0}".format(idx),
                        nn.BatchNorm2d(self.conv_flts[idx]))
                self.w_conv.add_module("act_{0}".format(idx),
                                       getattr(nn, activation)(inplace=True))

        self.output_dim = compute_output_dim(input_dim,
                                             [*self.left_conv, *self.w_conv])
        self.flatten_dim = self.output_dim ** 2 * self.conv_flts[-1]
        self.w_dense = nn.Linear(self.flatten_dim, self.latent_dim)

    def forward(self, x):
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
        left_x, right_x = x
        left_x = self.left_conv(left_x)
        right_x = self.right_conv(right_x)
        x = torch.cat((left_x, right_x), dim=1)
        x = self.w_conv(x)
        x = x.view(-1, self.flatten_dim)
        z = self.w_dense(x)
        return z


class HemiFusionDecoder(nn.Module):
    def __init__(self, output_channels, input_dim, latent_dim,
                 conv_flts=[64, 128, 128, 256, 256], fusion_level=1,
                 activation="LeakyReLU", batch_norm=False):
        """ Init class.

        Parameters
        ----------
        output_channels: int, default 1
            the number of output channels.
        input_dim: int,
            the size of the squared input to the convnet, after the dense
            layer transforming the input from the latent space.
        latent_dim: int, default 64
            the size of the latent space it decodes from.
        conv_flts: list of int
            the size of convolutional filters, given in reverse order: the
            first filter in the list will be the last one in the network.
        fusion_level: int, default 1
            at which max pooling level left and right hemisphere data
            are concatenated.
        activation: str, default 'LeakyReLU'
            activation function's class name in pytorch's nn module to use
            after each convolution
        batch_norm: bool, default False
            optionally uses batch normalization after each convolution
        """
        logger.debug("HemiFusionDecoder init...")
        super().__init__()

        self.input_dim = input_dim
        self.conv_flts = conv_flts.copy()
        self.n_layers = len(conv_flts)
        self.conv_flts.insert(0, output_channels*2)
        flatten_dim = input_dim ** 2 * conv_flts[-1]
        self.w_dense = nn.Linear(latent_dim, flatten_dim)
        self.w_conv = nn.Sequential()
        self.left_conv = nn.Sequential()
        self.right_conv = nn.Sequential()
        self.fusion_level = fusion_level
        for idx in range(self.n_layers, 0, -1):
            if idx == 1:
                kernel_size = 8
                pad = 3
                zero_pad = 9
            else:
                kernel_size = 4
                pad = 1
                zero_pad = 3
            input_channels = self.conv_flts[idx]
            output_channels = self.conv_flts[idx - 1]
            if idx < fusion_level + 1:
                input_channels = int(input_channels / 2)
                output_channels = int(output_channels / 2)
                lconv = IcoSpMaConvTranspose(
                    in_feats=input_channels, out_feats=output_channels,
                    kernel_size=kernel_size, stride=2, pad=pad,
                    zero_pad=zero_pad)
                if batch_norm:
                    self.left_conv.add_module(
                        "l_bn_{0}".format(idx),
                        nn.BatchNorm2d(input_channels))
                self.left_conv.add_module("l_act_{0}".format(idx),
                                          getattr(nn, activation)())
                self.left_conv.add_module("l_conv_{0}".format(idx), lconv)
                rconv = IcoSpMaConvTranspose(
                    in_feats=input_channels, out_feats=output_channels,
                    kernel_size=kernel_size, stride=2, pad=pad,
                    zero_pad=zero_pad)
                if batch_norm:
                    self.right_conv.add_module(
                        "r_bn_{0}".format(idx),
                        nn.BatchNorm2d(input_channels))
                self.right_conv.add_module("r_act_{0}".format(idx),
                                           getattr(nn, activation)())
                self.right_conv.add_module("r_conv_{0}".format(idx), rconv)
            else:
                conv = IcoSpMaConvTranspose(
                    input_channels, output_channels, kernel_size=kernel_size,
                    stride=2, pad=pad, zero_pad=zero_pad)
                if batch_norm and idx != self.n_layers:
                    self.w_conv.add_module(
                        "bn_{0}".format(idx),
                        nn.BatchNorm2d(output_channels))
                self.w_conv.add_module("act_{0}".format(idx),
                                       getattr(nn, activation)(inplace=True))
                self.w_conv.add_module("conv_{0}".format(idx), conv)

    def forward(self, z):
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
        x = self.w_dense(z)
        x = x.view(-1, self.conv_flts[-1], self.input_dim,
                   self.input_dim)
        # if self.fusion_level < self.n_layers:
        x = self.w_conv(x)
        left_recon_x, right_recon_x = torch.chunk(x, chunks=2, dim=1)
        left_recon_x = self.left_conv(left_recon_x)
        right_recon_x = self.right_conv(right_recon_x)
        return left_recon_x, right_recon_x
