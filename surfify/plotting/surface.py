# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Utility functions to display surfaces.
"""

# Imports
import numpy as np
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_trisurf(vertices, triangles, texture=None, vmin=None,
                 vmax=None, colorbar=True, fig=None, ax=None,
                 is_label=False, alpha=1, edgecolors="black",
                 linewidths=1, color_map=cm.coolwarm):
    """ Display a triangular surface.

    Parameters
    ----------
    vertices: array (N, 3)
        the surface vertices.
    triangles: array (M, 3)
        the surface triangles.
    texture: array (N,), default None
        a texture to display on the surface.
    vmin: float, default None
        minimum value to map.
    vmax: float, default None
        maximum value to map.
    colorbar: bool, default True
        display a colorbar.
    fig: Figure, default None
        the matplotlib figure.
    ax: Axes3D, default None
        axis to display the surface plot.
    is_label: bool, default False
        optionally specify that the texture contains labels in order to
        use most representative neighboor interpolation.
    alpha: float, default 1
        number between 0 and 1 defining the opacity of the faces. Not use
        when texture is None, where alpha is fixed at 0.1
    edgecolors: string, default 'black'
        name of the matplotlib color to use for the edges
    linewidths: float, default 1
        edge line width.
    color_map: Mapping, default cm.coolwarm
        container for colormaps.
    """

    # Parameters
    if texture is not None and vmin is None:
        vmin = texture.min()
    if texture is not None and vmax is None:
        vmax = texture.max()
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    if texture is not None:
        if is_label:
            texture = np.asarray([
                np.argmax(np.bincount(texture[tri].astype(int)))
                for tri in triangles])
        else:
            texture = np.asarray([np.mean(texture[tri]) for tri in triangles])

    # Display tri surface
    triangle_vertices = np.array([vertices[tri] for tri in triangles])
    if texture is not None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        facecolors = color_map(norm(texture))
        polygon = Poly3DCollection(triangle_vertices, facecolors=facecolors,
                                   edgecolors=edgecolors, alpha=alpha,
                                   linewidths=linewidths)
    else:
        polygon = Poly3DCollection(triangle_vertices, facecolors="white",
                                   edgecolors=edgecolors, alpha=0.1,
                                   linewidths=linewidths)
    ax.add_collection3d(polygon)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Add colorbar
    if texture is not None:
        m = cm.ScalarMappable(cmap=color_map, norm=norm)
        m.set_array(texture)
        if colorbar:
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    # Get rid of the panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
