# -*- coding: utf-8 -*-
"""
Spherical augmentations
=======================

Credit: A Grigis & C Ambroise

A simple example on how to use augmentations in the spherical domain.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
<<<<<<< HEAD
from surfify.utils import icosahedron, neighbors, min_depth_to_get_n_neighbors
from surfify.plotting import plot_trisurf
from surfify.augmentation import (
    SurfCutOut, SurfNoise, SurfBlur, SurfRotation, HemiMixUp, GroupMixUp,
    Transformer, interval)

vertices, triangles = icosahedron(order=3)
neighs = neighbors(vertices, triangles, direct_neighbor=True)
max_depth = min_depth_to_get_n_neighbors(np.ceil(len(vertices) / 4))


def display(vertices, triangles, texture, aug, add_axis=False, *args,
            **kwargs):
    """ Display augmented data.
    """
    fig, axs = plt.subplots(
        2, 2, subplot_kw={"projection": "3d", "aspect": "auto"},
        figsize=(10, 10))
    axs = axs.flatten()
    plot_trisurf(vertices, triangles, texture, ax=axs[0], fig=fig,
                 alpha=0.3, colorbar=False, edgecolors="white",
                 linewidths=0.2)
    if add_axis:
        axs[0].plot([0, 0], [0, 0], [-1, 1], c="red")
        axs[0].plot([0, 0], [-1, 1], [0, 0], c="blue")
        axs[0].plot([-1, 1], [0, 0], [0, 0], c="green")
    for idx in range(1, len(axs)):
        _texture = aug(texture, *args, **kwargs)
        if add_axis:
            axs[idx].plot([0, 0], [0, 0], [-1, 1], c="red")
            axs[idx].plot([0, 0], [-1, 1], [0, 0], c="blue")
            axs[idx].plot([-1, 1], [0, 0], [0, 0], c="green")
        plot_trisurf(vertices, triangles, _texture, ax=axs[idx], fig=fig,
                     alpha=0.3, colorbar=False, edgecolors="white",
                     linewidths=0.2)
    fig.tight_layout()

=======
from surfify.utils import icosahedron, neighbors, setup_logging
from surfify.plotting import plot_trisurf
from surfify.augmentation import (SphericalRandomRotation, SphericalRandomCut,
                                  SphericalBlur, SphericalNoise)
from surfify import datasets

setup_logging(level="info", logfile=None)

coords, triangles = icosahedron(order=4)
neighs = neighbors(coords, triangles, direct_neighbor=True)
print(coords.shape)
print(triangles.shape)
>>>>>>> b088a8b (lots of stuff : spherical and grided vae, new augmentations, plot new augmentations, new function to compute the right number of ring for surf cutout)

#############################################################################
# SurfCutOut
# ----------
#

texture = np.array([1, ] * len(vertices))
aug = SurfCutOut(vertices, triangles, neighs=neighs,
                 patch_size=interval((2, max_depth), int),
                 n_patches=interval((1, 3), int),
                 sigma=1)
display(vertices, triangles, texture, aug)


#############################################################################
# SurfNoise
# ---------
#

texture = np.random.uniform(0, 3, len(vertices))
aug = SurfNoise(sigma=interval((1, 3), float))
display(vertices, triangles, texture, aug)


#############################################################################
# SurfBlur
# -----------
#

<<<<<<< HEAD
texture = np.random.uniform(0, 2, len(vertices))
aug = SurfBlur(vertices, triangles, sigma=interval((0.1, 1), float))
display(vertices, triangles, texture, aug)

=======
tri_texture = np.array([[1, 1]]*len(coords)).T

augmentations = []
print("initializing random cut augmentations...")
for idx, depth in enumerate(range(1, 14, 3)):
    aug = SphericalRandomCut(
        coords, triangles, neighs=None, patch_size=depth,
        n_patches=(9 - idx * 2), random_size=False)
    augmentations.append(aug)

fig, ax = plt.subplots(2, 3, subplot_kw={
        "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
colorbar = False
plot_trisurf(coords, triangles, tri_texture[0], fig=fig, ax=ax[0, 0],
             alpha=0.3, colorbar=colorbar, edgecolors="white",
             linewidths=0.2)#, is_label=True)
for idx in range(5):
    if idx == 4:
        colorbar = True
    augmented_texture = augmentations[idx](tri_texture)
    plot_trisurf(coords, triangles, augmented_texture[0],
                 ax=ax[(idx + 1) // 3, (idx + 1) % 3], fig=fig,
                 alpha=0.3, colorbar=colorbar, edgecolors="white",
                 linewidths=0.2)#, is_label=True)
fig.tight_layout()
>>>>>>> b088a8b (lots of stuff : spherical and grided vae, new augmentations, plot new augmentations, new function to compute the right number of ring for surf cutout)

#############################################################################
# SurfRotation
# ------------
#

texture = np.array([1, ] * len(vertices))
aug = SurfRotation(vertices, triangles, phi=interval((5, 180), float), theta=0,
                   psi=0)
texture[neighs[0]] = 0
display(vertices, triangles, texture, aug, add_axis=True)

<<<<<<< HEAD

#############################################################################
# HemiMixUp
# ---------
#

texture = np.random.uniform(0, 3, len(vertices))
controlateral_texture = np.random.uniform(0, 3, len(vertices))
aug = HemiMixUp(prob=interval((0.2, 0.5), float), n_vertices=len(vertices))
display(vertices, triangles, texture, aug,
        controlateral_data=controlateral_texture)


#############################################################################
# GroupMixUp
# ----------
#

texture = np.random.uniform(0, 3, len(vertices))
group_textures = np.random.uniform(0, 3, (10, len(vertices)))
aug = GroupMixUp(prob=interval((0.2, 0.5), float), n_vertices=len(vertices))
display(vertices, triangles, texture, aug, group_data=group_textures,
        n_samples=1)


#############################################################################
# GroupMixUp group
# ----------------
#

textures = np.random.uniform(0, 3, (100, len(vertices)))
neigh_ind = GroupMixUp.groupby(textures, n_neighbors=4, n_components=30)
print(neigh_ind)


#############################################################################
# Transformer
# -----------
#

texture = np.random.uniform(0, 3, len(vertices))
aug1 = SurfCutOut(vertices, triangles, neighs=neighs,
                  patch_size=interval((1, max_depth), int),
                  n_patches=interval((1, 3), int))
aug2 = SurfNoise(sigma=interval((1, 3), float))
aug3 = SurfBlur(vertices, triangles, sigma=interval((0.1, 1), float))
trans = Transformer()
trans.register(aug1, probability=.75)
trans.register(aug2, probability=.75)
trans.register(aug3, probability=.5)
display(vertices, triangles, texture, trans)
=======
# We also set the vertices neighbouring vertex 0 to 1
tri_texture[neighs[0]] = [1, 1]
fig, ax = plt.subplots(2, 2, subplot_kw={
        "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
ax[0, 0].plot([0, 0], [0, 0], [-1, 1], c="red")
ax[0, 0].plot([0, 0], [-1, 1], [0, 0], c="blue")
ax[0, 0].plot([-1, 1], [0, 0], [0, 0], c="green")
colorbar = False
plot_trisurf(coords, triangles, tri_texture[:, 0], fig=fig, ax=ax[0, 0],
             alpha=0.3, colorbar=colorbar, edgecolors="white",
             linewidths=0.2)
for idx, angle in enumerate(range(90, 271, 90)):
    if idx == 2:
        colorbar = True
    augmented_texture = augmentations[idx](tri_texture)
    ax[(idx + 1) // 2, (idx + 1) % 2].plot([0, 0], [0, 0], [-1, 1], c="red")
    ax[(idx + 1) // 2, (idx + 1) % 2].plot([0, 0], [-1, 1], [0, 0], c="blue")
    ax[(idx + 1) // 2, (idx + 1) % 2].plot([-1, 1], [0, 0], [0, 0], c="green")
    print("Angle rotation {}: {}".format(idx, augmentations[idx].angles))
    plot_trisurf(coords, triangles, augmented_texture[:, 0],
                 ax=ax[(idx + 1) // 2, (idx + 1) % 2], fig=fig,
                 colorbar=colorbar, alpha=0.3, edgecolors="white",
                 linewidths=0.2)
fig.tight_layout()

#############################################################################
# Spherical blur
# -----------
#
# Display blured textures.

tri_texture = np.array([[2, 2], [0, 0]]*int(len(coords) / 2)).T

print("initializing blur augmentation...")
augmentations = []
for i in range(5):
    aug = SphericalBlur(
        coords, triangles, sigma=(i + 1) / 5, fixed_sigma=True)
    augmentations.append(aug)

fig, ax = plt.subplots(2, 3, subplot_kw={
        "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
colorbar = False
plot_trisurf(coords, triangles, tri_texture[0], fig=fig, ax=ax[0, 0],
             alpha=1, colorbar=colorbar, edgecolors="black", linewidths=0.2)
for idx in range(5):
    if idx == 4:
        colorbar = True
    augmented_texture = augmentations[idx](tri_texture)
    plot_trisurf(coords, triangles, augmented_texture[0],
                 ax=ax[(idx + 1) // 3, (idx + 1) % 3], fig=fig,
                 colorbar=colorbar, edgecolors="black", linewidths=0.2,
                 vmax=2, vmin=0)
fig.tight_layout()

#############################################################################
# Spherical nosie
# -----------
#
# Display blured textures.

tri_texture = np.array([[0, 0]]*len(coords)).T

print("initializing blur augmentation...")
augmentations = []
for i in range(5):
    aug = SphericalNoise(
        sigma=(i + 1) * 2 / 5, fixed_sigma=True)
    augmentations.append(aug)

fig, ax = plt.subplots(2, 3, subplot_kw={
        "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
colorbar = False
plot_trisurf(coords, triangles, tri_texture[0], fig=fig, ax=ax[0, 0],
             alpha=1, colorbar=colorbar, edgecolors="black", linewidths=0.2)
for idx in range(5):
    if idx == 4:
        colorbar = True
    augmented_texture = augmentations[idx](tri_texture)
    plot_trisurf(coords, triangles, augmented_texture[0],
                 ax=ax[(idx + 1) // 3, (idx + 1) % 3], fig=fig,
                 colorbar=colorbar, edgecolors="black", linewidths=0.2,
                 vmax=3, vmin=0)
fig.tight_layout()

>>>>>>> b088a8b (lots of stuff : spherical and grided vae, new augmentations, plot new augmentations, new function to compute the right number of ring for surf cutout)

plt.show()
