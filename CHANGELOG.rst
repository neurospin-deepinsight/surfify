.. -*- mode: rst -*-

Version 0.2.0
=============

:Date: work in progress

Added
-----

* dynamic adaptation of the RePa zoom in `SphericalBase` to control the size
  of the generated grid on the spherical surface.
* add a `fusion_level` parameter on VGG and VAE models in order to control
  the left and right hemispheres fusion level.
* RandomAugmentation: aplly an augmentation with random parameters defined in
  intervals.
* SurfNoise, HemiMixUp, GroupMixUp: new spherical augmentations.
* Transformer: register augmentation tranformations and apply them with custom
  probabilities.

Changed
-------

* SphericalRandomCut -> SurfCutOut
* SphericalRandomRotation -> SurfRotation

Deprecated
----------

Fixed
-----

* normalize weights in RePa.
* DiNe: change how missing nodes are handle during the neighborhood generation.

Contributors
------------

The following people contributed to this release (from ``git shortlog -ns v0.2.0``)::


Version 0.1.0
=============

:Date: October 15, 2021

Added
-----

* RePa: Rectangular Patch convolution method.
* DiNe: Direct Neighbor convolution method.
* SpMa: Spherical Mapping convolution method.
* SphericalUNet, SphericalGUNet: spherical UNet with RePa, DiNe and SpMa.
* SphericalVAE, SphericalGVAE: spherical VAE with RePa, DiNe and SpMa.
* SphericalVGG, SphericalVGG: spherical VGG with RePa, DiNe and SpMa.
* SphericalRandomCut, SphericalRandomRotation: spherical augmentations.
* plot_trisurf: utility function to plot icosahedron surfaces.

Changed
-------

Deprecated
----------

Fixed
-----

Contributors
------------

The following people contributed to this release (from ``git shortlog -ns v0.1.0``)::

   55  Antoine Grigis
   20  CorentinAmbroise
