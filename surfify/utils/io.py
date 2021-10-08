# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Spherical i/o utilities.
"""

# Imports
import os
import sys
import gzip
import shutil
import inspect
import nibabel
import numpy as np
from joblib import Memory
from logging import getLogger

logger = getLogger("surfify")


def ungzip(path):
    """ Extract GNU zipped archive file.

    Parameters
    ----------
    path: str
        the archive file to be opened: must be a .gz file.

    Returns
    -------
    out_path: str
        the generated file at the same location without the .gz extension.
    """
    assert path.endswith(".gz")
    dest_path = path.replace(".gz", "")
    if not os.path.isfile(dest_path):
        with gzip.open(path, "rb") as f_in:
            with open(dest_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    return dest_path


def read_gifti(surf_file):
    """ Read a surface geometry stored in GIFTI format.

    Parameters
    ----------
    surf_file: str
        the input GIFTI surface file.

    Returns
    -------
    vertices: array (N, 3)
        the N vertices of the surface.
    triangles: array (M, 3)
        the M triangles that defines the surface geometry.
    """
    image = nibabel.load(surf_file)
    nb_of_surfs = len(image.darrays)
    if nb_of_surfs != 2:
        raise ValueError(
            "'{0}' does not contain a valid mesh.".format(surf_file))
    vertices = image.darrays[0].data
    triangles = image.darrays[1].data
    return vertices, triangles


def read_freesurfer(surf_file):
    """ Read a surface geometry stored in FreeSurfer format.

    Parameters
    ----------
    surf_file: str
        the input FreeSurfer surface file.

    Returns
    -------
    vertices: array (N, 3)
        the N vertices of the surface.
    triangles: array (M, 3)
        the M triangles that defines the surface geometry.
    """
    vertices, traingles = nibabel.freesurfer.read_geometry(surf_file)
    return vertices, traingles


def write_gifti(vertices, triangles, surf_file):
    """ Write a surface geometry in GIFTI format.

    Parameters
    ----------
    vertices: array (N, 3)
        the N vertices of the surface to be saved.
    triangles: array (M, 3)
        the M triangles that defines the surface geometry to be saved.
    surf_file: str
        the path to the generated GIFTI surface file.
    """
    vertices_array = nibabel.gifti.GiftiDataArray(
        data=vertices,
        intent=nibabel.nifti1.intent_codes["NIFTI_INTENT_POINTSET"])
    triangles_array = nibabel.gifti.GiftiDataArray(
        data=triangles,
        intent=nibabel.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"])
    gii = nibabel.gifti.GiftiImage(darrays=[vertices_array, triangles_array])
    nibabel.gifti.write(gii, surf_file)


def write_freesurfer(vertices, triangles, surf_file):
    """ Write a surface geometry in FreeSurfer format.

    Parameters
    ----------
    vertices: array (N, 3)
        the N vertices of the surface to be saved.
    triangles: array (M, 3)
        the M triangles that defines the surface geometry to be saved.
    surf_file: str
        the path to the generated FreeSurfer surface file.
    """
    nibabel.freesurfer.io.write_geometry(surf_file, vertices, triangles,
                                         create_stamp="",
                                         volume_info=None)


class HidePrints(object):
    """ This function securely redirect the standard outputs and errors. The
    resulting object can be used as a context manager. On completion of the
    context the default context is restored.
    """
    def __init__(self, hide_err=False):
        """ Init class.

        Parameters
        ----------
        hide_err: bool, default False
            optionally hide the standard errors.
        """
        self.hide_err = hide_err

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        if self.hide_err:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        if self.hide_err:
            sys.stderr.close()
            sys.stderr = self._original_stderr


def compute_and_store(func, path):
    """ Decorator allowing to compute and store a function's output to
    access it faster on next same calls of the wrapped function

    Parameters
    ----------
    func: callable
        function to cache. It will receive arguments of the wrapped
        function that have the same name as its arguments when
        executed.
    path: string
        path to store the function's output.

    Returns
    -------
    decorator: callable
        the decorator that can use the cached function.
    """
    memory = Memory(path, verbose=0)
    cached_func = memory.cache(func)
    params = inspect.signature(func).parameters
    logger.debug("compute_and_store decorator's params : ", params)
    def decorate(func):
        def wrapped(*args, **kwargs):
            logger.debug("wrapped function args : ", len(args))
            logger.debug("wrapped function kwargs : ", kwargs.keys())
            wrapped_params = inspect.signature(func).parameters
            logger.debug("wrapped function params : ", wrapped_params)
            common_args = {
                name: list(wrapped_params).index(name)
                for name in params if name in wrapped_params}
            # we want to use arguments that are common to both function by name
            cached_func_args = dict((name, kwargs[name]) for name in common_args
                                     if name in kwargs.keys())
            to_remove = []
            for name in common_args:
                if name not in cached_func_args.keys():
                    # if the param's index is lower than the len of args, then it uses
                    # the default value
                    if common_args[name] < len(args):
                        cached_func_args[name] = args[common_args[name]]
                        to_remove.append(common_args[name])
                    else:
                        cached_func_args[name] = wrapped_params[name].default

            # If a param was previously added, we want to remove it from args so that
            # the wrapped function only receive it once through the kwargs
            to_remove = np.array(to_remove)
            to_remove.sort()
            args = list(args)
            for i in to_remove:
                del args[i]
                to_remove -= 1

            kwargs.update(cached_func_args)
            new_kwargs = cached_func(**cached_func_args)
            kwargs.update(new_kwargs)

            response = func(*args, **kwargs)
            return response
        return wrapped
    return decorate
