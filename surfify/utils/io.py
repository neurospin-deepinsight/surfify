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
import nibabel


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
