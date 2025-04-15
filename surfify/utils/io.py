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


def decompose_cifti(cifti_file, raw=False):
    """ Decompose CIFTI data.

    Parameters
    ----------
    cifti_file: str
        the path to a CIFTI image.
    raw: bool, default False
        if set return raw data as stored in the CIFTI file.

    Returns
    -------
    vol: array
        the raw/organized volume data.
    surf_left: array
        the raw/organized left surface data.
    surf_right: array
        the raw/organized right surface data.
    """
    img = nibabel.load(cifti_file)
    data = img.get_fdata(dtype=np.float32)
    hdr = img.header
    axes = [hdr.get_axis(idx) for idx in range(img.ndim)]
    select_axes = [axis for axis in axes
                   if isinstance(axis, nibabel.cifti2.BrainModelAxis)]
    assert len(select_axes) == 1
    brain_models = select_axes[0]
    return (volume_from_cifti(data, brain_models, raw),
            surf_data_from_cifti(data, brain_models,
                                 "CIFTI_STRUCTURE_CORTEX_LEFT", raw),
            surf_data_from_cifti(data, brain_models,
                                 "CIFTI_STRUCTURE_CORTEX_RIGHT", raw))


def surf_data_from_cifti(data, axis, surf_name, raw=False):
    """ Load CIFTI surface data (see here <https://nbviewer.org/github/
    neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/
    NiBabel.ipynb>`_).
    """
    assert isinstance(axis, nibabel.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():
        if name == surf_name:
            data = data.T[data_indices]
            if raw:
                return data
            vtx_indices = model.vertex
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:],
                                 dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def volume_from_cifti(data, axis, raw=False):
    """ Load CIFTI volume data (see here <https://nbviewer.org/github/
    neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/
    NiBabel.ipynb>`_).
    """
    assert isinstance(axis, nibabel.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]
    if raw:
        return data
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)
    vol_data = np.zeros(axis.volume_shape + data.shape[1:],
                        dtype=data.dtype)
    vol_data[vox_indices] = data
    return nibabel.Nifti1Image(vol_data, axis.affine)


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
        with gzip.open(path, "rb") as f_in, open(dest_path, "wb") as f_out:
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


class HidePrints:
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


def compute_and_store(func, cachedir=None):
    """ Decorator allowing to compute and store a function's output to
    access them faster on the next calls of the wrapped function.

    Notes
    -----
    The decorator input function and the decorated function must have
    overlaping arguments. The decorator input function must also returns
    a dictionnary containing the items to be stored.

    Parameters
    ----------
    func: callable
        function to cache. It will receive arguments of the wrapped
        function that have the same name as its arguments when
        executed.
    cachedir: str, default None
        the path of the base directory to use as a data store or None.
        If None is given, no caching is done and the Memory object is
        completely transparent.

    Returns
    -------
    decorator: callable
        the decorated function that can use the cached function outputs.
    """
    memory = Memory(cachedir, verbose=0)
    cached_func = memory.cache(func)
    params = inspect.signature(func).parameters
    logger.debug("compute_and_store decorator's params : {}".format(params))

    def decorate(func):
        def wrapped(*args, **kwargs):
            logger.debug("wrapped function args : {}".format(len(args)))
            logger.debug("wrapped function kwargs : {}".format(kwargs.keys()))
            wrapped_params = inspect.signature(func).parameters
            logger.debug("wrapped function params : {}".format(wrapped_params))
            common_args = {
                name: list(wrapped_params).index(name)
                for name in params if name in wrapped_params}
            # we want to use arguments that are common to both function by name
            cached_func_args = dict(
                (name, kwargs[name]) for name in common_args if name in kwargs)
            for name in common_args:
                if name not in cached_func_args:
                    # if the param's index is lower than the len of args and is
                    # not entered as kwargs, then it uses the default value
                    if common_args[name] < len(args):
                        cached_func_args[name] = args[common_args[name]]
                    else:
                        cached_func_args[name] = wrapped_params[name].default

            logger.debug("cached function kwargs : {}".format(
                cached_func_args))
            if len(params) != len(cached_func_args):
                raise ValueError(
                    "The decorator input function and the decorated function "
                    "must have overlaping arguments.")
            new_kwargs = cached_func(**cached_func_args)
            if not isinstance(new_kwargs, dict):
                raise ValueError(
                    "The decorator input function must also returns a "
                    "dictionnary containing the items to be stored.")
            kwargs.update(new_kwargs)

            logger.debug("wrapped function args : {}".format(args))
            logger.debug("wrapped function kwargs : {}".format(kwargs))
            response = func(*args, **kwargs)
            return response
        return wrapped

    return decorate
