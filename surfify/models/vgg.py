# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Spherical implementation of the torch vision VGG.
"""

# Imports
import torch
import torch.nn as nn
from ..utils import get_logger, debug_msg
from ..nn import IcoUpConv, IcoPool, IcoSpMaConv
from .base import SphericalBase


# Global parameters
logger = get_logger()


class SphericalVGG(SphericalBase):
    """ Spherical VGG architecture.

    Notes
    -----
    Debuging messages can be displayed by changing the log level using
    ``setup_logging(level='debug')``.

    See Also
    --------
    SphericalGVGG
    """
    def __init__(self, input_channels, cfg, n_classes, input_order=5,
                 conv_mode="DiNe", dine_size=1, repa_size=5, repa_zoom=5,
                 hidden_dim=4096, batch_norm=False, init_weights=True,
                 standard_ico=False, cachedir=None):
        """ Init class.

        Parameters
        ----------
        input_channels: int
            the number of input channels.
        cfg: list
            the definition of layers where 'M' stands for max pooling.
        num_classes: int
            the number of class in the classification problem.
        input_order: int, default 5
            the input icosahedron order.
        conv_mode: str, default 'DiNe'
            use either 'RePa' - Rectangular Patch convolution method or 'DiNe'
            - 1 ring Direct Neighbor convolution method.
        dine_size: int, default 1
            the size of the spherical convolution filter, ie. the number of
            neighbor rings to be considered.
        repa_size: int, default 5
            the size of the rectangular grid in the tangent space.
        repa_zoom: int, default 5
            a multiplicative factor applied to the rectangular grid in the
            tangent space.
        hidden_dim: int, default 4096
            the 2-layer classification MLP number of hidden dims.
        batch_norm: bool, default False
            wether or not to use batch normalization after a convolution
            layer.
        init_weights: bool, default True
            initialize network weights.
        standard_ico: bool, default False
            optionaly use surfify tesselation.
        cachedir: str, default None
            set this folder to use smart caching speedup.
        """
        logger.debug("SphericalVGG init...")
        cfg = cfg[:-1]
        super(SphericalVGG, self).__init__(
            input_order=input_order, n_layers=cfg.count("M"),
            conv_mode=conv_mode, dine_size=dine_size, repa_size=repa_size,
            repa_zoom=repa_zoom, standard_ico=standard_ico,
            cachedir=cachedir)
        self.input_channels = input_channels
        self.cfg = cfg
        self.n_classes = n_classes
        self.batch_norm = batch_norm
        self.n_modules = len(cfg)
        self.final_flt = int(cfg[-1])
        self.top_flatten_dim = len(
            self.ico[self.input_order - self.n_layers + 1].vertices)
        self.top_final = self.final_flt * 7
        self._make_encoder()
        self.avgpool = nn.AdaptiveAvgPool1d((7))
        self.classifier = nn.Sequential(
            nn.Linear(self.top_final, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, n_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, left_x, right_x):
        """ Forward method.

        Parameters
        ----------
        left_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input left cortical texture.
        right_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input right cortical texture.

        Returns
        -------
        out: torch.Tensor
        """
        logger.debug("SphericalVGG forward pass")
        logger.debug(debug_msg("left cortical", left_x))
        logger.debug(debug_msg("right cortical", right_x))
        x = torch.cat(
            (self.enc_left_conv(left_x), self.enc_right_conv(right_x)), dim=1)
        logger.debug(debug_msg("lh/rh path", x))
        for mod in self.enc_w_conv.children():
            if isinstance(mod, IcoPool):
                x = mod(x)[0]
            else:
                x = mod(x)
        logger.debug(debug_msg("features", x))
        x = self.avgpool(x)
        logger.debug(debug_msg("avg pooling", x))
        x = torch.flatten(x, 1)
        logger.debug(debug_msg("flat", x))
        x = self.classifier(x)
        logger.debug(debug_msg("classifier", x))
        return x

    def _initialize_weights(self):
        """ Init model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_encoder(self):
        """ Method to create the encoding layers.
        """
        input_channels = self.input_channels
        order = self.input_order
        self.enc_left_conv = nn.Sequential()
        self.enc_right_conv = nn.Sequential()
        self.enc_w_conv = nn.Sequential()
        first_layer = True
        for idx in range(self.n_modules):
            if self.cfg[idx] == "M":
                order -= 1
                if first_layer:
                    first_layer = False
                    input_channels *= 2
                pooling = IcoPool(
                    down_neigh_indices=self.ico[order + 1].neighbor_indices,
                    down_indices=self.ico[order + 1].down_indices,
                    pooling_type="max")
                self.enc_w_conv.add_module("pooling_{0}".format(idx), pooling)
            elif first_layer:
                lconv = self.sconv(
                    input_channels, (self.cfg[idx] // 2),
                    self.ico[order].conv_neighbor_indices)
                self.enc_left_conv.add_module("lconv_{0}".format(idx), lconv)
                if self.batch_norm:
                    lbn = nn.BatchNorm1d(self.cfg[idx] // 2)
                    self.enc_left_conv.add_module("lbn_{0}".format(idx), lbn)
                lrelu = nn.ReLU(inplace=True)
                self.enc_left_conv.add_module("lrelu_{0}".format(idx), lrelu)
                rconv = self.sconv(
                    input_channels, (self.cfg[idx] // 2),
                    self.ico[order].conv_neighbor_indices)
                self.enc_right_conv.add_module("rconv_{0}".format(idx), rconv)
                if self.batch_norm:
                    rbn = nn.BatchNorm1d(self.cfg[idx] // 2)
                    self.enc_right_conv.add_module("rbn_{0}".format(idx), rbn)
                rrelu = nn.ReLU(inplace=True)
                self.enc_right_conv.add_module("rrelu_{0}".format(idx), rrelu)
                input_channels = self.cfg[idx] // 2
            else:
                conv = self.sconv(
                    input_channels, self.cfg[idx],
                    self.ico[order].conv_neighbor_indices)
                self.enc_w_conv.add_module("conv_{0}".format(idx), conv)
                if self.batch_norm:
                    bn = nn.BatchNorm1d(self.cfg[idx])
                    self.enc_w_conv.add_module("bn_{0}".format(idx), bn)
                relu = nn.ReLU(inplace=True)
                self.enc_w_conv.add_module("relu_{0}".format(idx), relu)
                input_channels = self.cfg[idx]


class SphericalGVGG(nn.Module):
    """ Spherical Grided VGG architecture.

    Notes
    -----
    Debuging messages can be displayed by changing the log level using
    ``setup_logging(level='debug')``.

    See Also
    --------
    SphericalVGG
    """
    def __init__(self, input_channels, cfg, n_classes, input_dim=194,
                 hidden_dim=4096, batch_norm=False, init_weights=True):
        """ Init class.

        Parameters
        ----------
        input_channels: int
            the number of input channels.
        cfg: list
            the definition of layers where 'M' stands for max pooling.
        n_classes: int
            the number of class in the classification problem.
        input_dim: int, default 192
            the size of the converted 3-D surface to the 2-D grid.
        hidden_dim: int, default 4096
            the 2-layer classification MLP number of hidden dims.
        batch_norm: bool, default False
            wether or not to use batch normalization after a convolution
            layer.
        init_weights: bool, default True
            initialize network weights.
        """
        logger.debug("SphericalGVGG init...")
        super(SphericalGVGG, self).__init__()
        self.input_channels = input_channels
        self.cfg = cfg
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.batch_norm = batch_norm
        self.n_modules = len(cfg)
        self.n_layers = cfg.count("M")
        self.final_flt = int(cfg[-2])
        self.top_flatten_dim = int(self.input_dim / (2 ** self.n_layers))
        self.top_final = self.final_flt * 7 ** 2
        self._make_encoder()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(self.top_final, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, n_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, left_x, right_x):
        """ Forward method.

        Parameters
        ----------
        left_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input left cortical texture.
        right_x: Tensor (samples, <input_channels>, azimuth, elevation)
            input right cortical texture.

        Returns
        -------
        out: torch.Tensor
        """
        logger.debug("SphericalGVGG forward pass")
        logger.debug(debug_msg("left cortical", left_x))
        logger.debug(debug_msg("right cortical", right_x))
        x = torch.cat(
            (self.enc_left_conv(left_x), self.enc_right_conv(right_x)), dim=1)
        logger.debug(debug_msg("lh/rh path", x))
        x = self.enc_w_conv(x)
        logger.debug(debug_msg("features", x))
        x = self.avgpool(x)
        logger.debug(debug_msg("avg pooling", x))
        x = torch.flatten(x, 1)
        logger.debug(debug_msg("flat", x))
        x = self.classifier(x)
        logger.debug(debug_msg("classifier", x))
        return x

    def _initialize_weights(self):
        """ Init model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_encoder(self):
        """ Method to create the encoding layers.
        """
        input_channels = self.input_channels
        self.enc_left_conv = nn.Sequential()
        self.enc_right_conv = nn.Sequential()
        self.enc_w_conv = nn.Sequential()
        first_layer = True
        for idx in range(self.n_modules):
            if self.cfg[idx] == "M":
                if first_layer:
                    first_layer = False
                    input_channels *= 2
                pooling = nn.MaxPool2d(kernel_size=2, stride=2)
                self.enc_w_conv.add_module("pooling_{0}".format(idx), pooling)
            elif first_layer:
                lconv = IcoSpMaConv(
                    in_feats=input_channels,
                    out_feats=(self.cfg[idx] // 2),
                    kernel_size=3, pad=1)
                self.enc_left_conv.add_module("lconv_{0}".format(idx), lconv)
                if self.batch_norm:
                    lbn = nn.BatchNorm2d(self.cfg[idx] // 2)
                    self.enc_left_conv.add_module("lbn_{0}".format(idx), lbn)
                lrelu = nn.ReLU(inplace=True)
                self.enc_left_conv.add_module("lrelu_{0}".format(idx), lrelu)
                rconv = IcoSpMaConv(
                    in_feats=input_channels,
                    out_feats=(self.cfg[idx] // 2),
                    kernel_size=3, pad=1)
                self.enc_right_conv.add_module("rconv_{0}".format(idx), rconv)
                if self.batch_norm:
                    rbn = nn.BatchNorm2d(self.cfg[idx] // 2)
                    self.enc_right_conv.add_module("rbn_{0}".format(idx), rbn)
                rrelu = nn.ReLU(inplace=True)
                self.enc_right_conv.add_module("rrelu_{0}".format(idx), rrelu)
                input_channels = self.cfg[idx] // 2
            else:
                conv = IcoSpMaConv(
                    in_feats=input_channels,
                    out_feats=self.cfg[idx],
                    kernel_size=3, pad=1)
                self.enc_w_conv.add_module("conv_{0}".format(idx), conv)
                if self.batch_norm:
                    bn = nn.BatchNorm2d(self.cfg[idx])
                    self.enc_w_conv.add_module("bn_{0}".format(idx), bn)
                relu = nn.ReLU(inplace=True)
                self.enc_w_conv.add_module("relu_{0}".format(idx), relu)
                input_channels = self.cfg[idx]


def class_factory(klass_name, klass_params, destination_module_globals):
    """ Dynamically define a class.

    In order to make the class publicly accessible, we assign the result of
    the function to a variable dynamically using globals().

    Parameters
    ----------
    klass_name: str
        the class name that will be created.
    klass_params: dict
        the class specific parameters.
    """
    class SphericalVGGBase(SphericalVGG):
        cfg = None
        batch_norm = False

        def __init__(self, input_channels, n_classes, input_order=5,
                     conv_mode="DiNe", dine_size=1, repa_size=5, repa_zoom=5,
                     hidden_dim=4096, init_weights=True, standard_ico=False,
                     cachedir=None):
            if self.cfg is None:
                raise ValueError("Please specify a configuration first.")
            SphericalVGG.__init__(
                self,
                cfg=self.cfg,
                batch_norm=self.batch_norm,
                input_channels=input_channels,
                n_classes=n_classes,
                input_order=input_order,
                conv_mode=conv_mode,
                dine_size=dine_size,
                repa_size=repa_size,
                repa_zoom=repa_zoom,
                hidden_dim=hidden_dim,
                init_weights=init_weights,
                standard_ico=standard_ico,
                cachedir=cachedir)

    class SphericalGVGGBase(SphericalGVGG):
        cfg = None
        batch_norm = False

        def __init__(self, input_channels, n_classes, input_dim=194,
                     hidden_dim=4096, init_weights=True):
            if self.cfg is None:
                raise ValueError("Please specify a configuration first.")
            SphericalGVGG.__init__(
                self,
                cfg=self.cfg,
                batch_norm=self.batch_norm,
                input_channels=input_channels,
                n_classes=n_classes,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                init_weights=init_weights)

    klass_params.update({
        "__module__": destination_module_globals["__name__"],
        "_id":  destination_module_globals["__name__"] + "." + klass_name
    })
    _klass_name = "Spherical" + klass_name
    destination_module_globals[_klass_name] = type(
        _klass_name, (SphericalVGGBase, ), klass_params)
    _klass_name = "SphericalG" + klass_name
    destination_module_globals[_klass_name] = type(
        _klass_name, (SphericalGVGGBase, ), klass_params)

    klass_params["batch_norm"] = True
    _klass_name = "Spherical" + klass_name + "BN"
    destination_module_globals[_klass_name] = type(
        _klass_name, (SphericalVGGBase, ), klass_params)
    _klass_name = "SphericalG" + klass_name + "BN"
    destination_module_globals[_klass_name] = type(
        _klass_name, (SphericalGVGGBase, ), klass_params)


CFGS = {
    "VGG11": {
        "cfg": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512,
                "M"]
    },
    "VGG13": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512,
                512, "M"]
    },
    "VGG16": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512,
                "M", 512, 512, 512, "M"]
    },
    "VGG19": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512,
                512, 512, "M", 512, 512, 512, 512, "M"]
    }
}


destination_module_globals = globals()
for klass_name, klass_params in CFGS.items():
    class_factory(klass_name, klass_params, destination_module_globals)
