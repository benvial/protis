#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


"""
Photonic crystal slab
=====================

Metasurface with holes.
"""

# sphinx_gallery_thumbnail_number = 1

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import protis as pt

lattice = pt.Lattice([[1.0, 0], [0, 1.0]], discretization=(2**9, 2**9))
epsilon = lattice.ones() * 1
circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
epsilon[circ] = 8.9
sim = pt.Simulation(lattice, k=(0, 0), epsilon=epsilon, nh=33)
sim.build_epsilon_hat()
M = sim.epsilon_hat

cols = [
    "#4AB08C",
    "#f0f0f0",
    "#e2e2e2",
    "#b1a9a9",
    "#583d66",
    "#35253d",
    "#8c4ab0",
]
cmap = colors.ListedColormap(cols)


x = range(sim.nh)
x1, x2 = np.meshgrid(range(sim.nh), range(sim.nh), indexing="ij")
y = np.log10(np.abs(M))
y = np.fliplr(y)
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.pcolormesh(x1, x2, y.T, cmap=cmap, ec="white", lw=0.8)
plt.axis("off")
plt.axis("equal")
plt.savefig(
    "bg.svg",
    bbox_inches="tight",
    pad_inches=0,
)
