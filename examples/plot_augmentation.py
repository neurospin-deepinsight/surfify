# -*- coding: utf-8 -*-
"""
Spherical augmentations
=======================

Credit: A Grigis & C Ambroise

A simple example on how to use augmentations in the spherical domain.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from surfify.utils import icosahedron, neighbors
from surfify.plotting import plot_trisurf
from surfify.augmentation import (
    SurfCutOut, SurfNoise, SurfRotation, HemiMixUp, GroupMixUp, Transformer,
    interval)

vertices, triangles = icosahedron(order=3)
neighs = neighbors(vertices, triangles, direct_neighbor=True)


#############################################################################
# SurfCutOut
# ----------
#

texture = np.array([1, ] * len(vertices))
aug = SurfCutOut(vertices, triangles, neighs=neighs,
                 patch_size=interval((1, 4), int),
                 n_patches=interval((1, 3), int))
fig, axs = plt.subplots(
    2, 2, subplot_kw={"projection": "3d", "aspect": "auto"}, figsize=(10, 10))
axs = axs.flatten()
for idx in range(len(axs)):
    _texture = aug(texture)
    plot_trisurf(vertices, triangles, _texture, ax=axs[idx], fig=fig,
                 alpha=0.3, colorbar=False, edgecolors="white")


#############################################################################
# SurfNoise
# ---------
#

texture = np.random.uniform(0, 3, len(vertices))
aug = SurfNoise(sigma=interval((1, 3), float))
fig, axs = plt.subplots(
    2, 2, subplot_kw={"projection": "3d", "aspect": "auto"}, figsize=(10, 10))
axs = axs.flatten()
for idx in range(len(axs)):
    _texture = aug(texture)
    plot_trisurf(vertices, triangles, _texture, ax=axs[idx], fig=fig,
                 alpha=1, colorbar=False, edgecolors="white")


#############################################################################
# SurfRotation
# ------------
#

texture = np.array([1, ] * len(vertices))
aug = SurfRotation(vertices, triangles, phi=interval((5, 180), float), theta=0,
                   psi=0)
texture[neighs[0]] = 0
fig, axs = plt.subplots(
    2, 2, subplot_kw={"projection": "3d", "aspect": "auto"}, figsize=(10, 10))
axs = axs.flatten()
for idx in range(len(axs)):
    _texture = aug(texture)
    axs[idx].plot([0, 0], [0, 0], [-1, 1], c="red")
    axs[idx].plot([0, 0], [-1, 1], [0, 0], c="blue")
    axs[idx].plot([-1, 1], [0, 0], [0, 0], c="green")
    plot_trisurf(vertices, triangles, _texture, ax=axs[idx], fig=fig,
                 alpha=1, colorbar=False, edgecolors="white")


#############################################################################
# HemiMixUp
# ---------
#

texture = np.random.uniform(0, 3, len(vertices))
controlateral_texture = np.random.uniform(0, 3, len(vertices))
aug = HemiMixUp(prob=interval((0.2, 0.5), float), n_vertices=len(vertices))
fig, axs = plt.subplots(
    2, 2, subplot_kw={"projection": "3d", "aspect": "auto"}, figsize=(10, 10))
axs = axs.flatten()
for idx in range(len(axs)):
    _texture = aug(texture, controlateral_texture)
    plot_trisurf(vertices, triangles, _texture, ax=axs[idx], fig=fig,
                 alpha=1, colorbar=False, edgecolors="white")


#############################################################################
# GroupMixUp
# ----------
#

texture = np.random.uniform(0, 3, len(vertices))
group_textures = np.random.uniform(0, 3, (10, len(vertices)))
aug = GroupMixUp(prob=interval((0.2, 0.5), float), n_vertices=len(vertices))
fig, axs = plt.subplots(
    2, 2, subplot_kw={"projection": "3d", "aspect": "auto"}, figsize=(10, 10))
axs = axs.flatten()
for idx in range(len(axs)):
    _texture = aug(texture, group_textures)
    plot_trisurf(vertices, triangles, _texture, ax=axs[idx], fig=fig,
                 alpha=1, colorbar=False, edgecolors="white")


#############################################################################
# GroupMixUp group
# ----------------
#

textures = np.random.uniform(0, 3, (100, len(vertices)))
neigh_ind = GroupMixUp.groupby(textures, n_neighbors=4)
print(neigh_ind)


#############################################################################
# Transformer
# -----------
#

texture = np.random.uniform(0, 3, len(vertices))
aug1 = SurfCutOut(vertices, triangles, neighs=neighs,
                  patch_size=interval((1, 4), int),
                  n_patches=interval((1, 3), int))
aug2 = SurfNoise(sigma=interval((1, 3), float))
trans = Transformer()
trans.register(aug1, probability=.5)
trans.register(aug2, probability=.5)
fig, axs = plt.subplots(
    2, 2, subplot_kw={"projection": "3d", "aspect": "auto"}, figsize=(10, 10))
axs = axs.flatten()
for idx in range(len(axs)):
    _texture = trans(texture)
    plot_trisurf(vertices, triangles, _texture, ax=axs[idx], fig=fig,
                 alpha=1, colorbar=False, edgecolors="white")

plt.show()
