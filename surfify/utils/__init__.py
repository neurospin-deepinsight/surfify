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
import logging
import warnings
from .sampling import (
    interpolate, neighbors, downsample, neighbors_rec, icosahedron,
    number_of_ico_vertices, get_rectangular_projection)


# Global parameters
LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}


def get_logger():
    """ Setup the logger.

    Returns
    -------
    logger: logging.Logger
        return a logger.
    """
    return logging.getLogger("surfify")


def setup_logging(level="info", logfile=None):
    """ Setup the logging.

    Parameters
    ----------
    logfile: str, default None
        the log file.
    """
    logger = get_logger()
    logging_format = logging.Formatter(
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - "
        "%(message)s", "%Y-%m-%d %H:%M:%S")
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[-1])
    level = LEVELS.get(level, None)
    if level is None:
        raise ValueError("Unknown logging level.")
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging_format)
    logger.addHandler(stream_handler)
    if logfile is not None:
        file_handler = logging.FileHandler(logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
    if level != logging.DEBUG:
        warnings.simplefilter("ignore", DeprecationWarning)


def debug_msg(name, tensor):
    """ Format a debug message.

    Parameters
    ----------
    name: str
        the tensor name in the displayed message.
    tensor: Tensor
        a pytorch tensor.

    Returns
    -------
    msg: str
        the formated debug message.
    """
    return "  {3}: {0} - {1} - {2}".format(
        tensor.shape, tensor.get_device(), tensor.dtype, name)
