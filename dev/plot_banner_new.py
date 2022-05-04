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

plt.close("all")
plt.ion()
#
# lattice = pt.Lattice([[1.0, 0], [0, 1.0]], discretization=(2**9, 2**9))
# epsilon = lattice.ones() * 1
# circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
# epsilon[circ] = 8.9
# sim = pt.Simulation(lattice, k=(0, 0), epsilon=epsilon, nh=33)
# sim.build_epsilon_hat()
# M = sim.epsilon_hat
#
cols = [
    "#b04a65",
    "#f0f0f0",
    "#e2e2e2",
    "#b1a9a9",
    "#583d66",
    "#35253d",
    "#8c4ab0",
]
cmap = colors.ListedColormap(cols)
#
#
# x = range(sim.nh)
# x1, x2 = np.meshgrid(range(sim.nh), range(sim.nh), indexing="ij")
# y = np.log10(np.abs(M))
# y = np.fliplr(y)
# fig, ax = plt.subplots(figsize=(4, 4))
# im = ax.pcolormesh(x1, x2, y.T, cmap=cmap, ec="white", lw=0.8)
# plt.axis("off")
# plt.axis("equal")


a = 1
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)


epsilon = lattice.ones() * 1
hole = lattice.square((0.5, 0.5), 0.33)
epsilon[hole] = 11.4
mu = 1


def k_space_path(Nb):
    K = np.linspace(0, np.pi / a, Nb)
    bands = np.zeros((3 * Nb - 3, 2))
    bands[:Nb, 0] = K
    bands[Nb : 2 * Nb, 0] = K[-1]
    bands[Nb : 2 * Nb - 1, 1] = K[1:]
    bands[2 * Nb - 1 : 3 * Nb - 3, 0] = bands[2 * Nb - 1 : 3 * Nb - 3, 1] = np.flipud(
        K
    )[1:-1]
    return bands, K


def full_model(bands, nh=100):
    sim = pt.Simulation(lattice, epsilon=epsilon, mu=mu, nh=nh)
    BD = {}
    sims = {}
    for polarization in ["TE", "TM"]:
        ev_band = []
        sims_ = []
        for kx, ky in bands:
            sim.k = kx, ky
            sim.solve(polarization)
            ev_norma = sim.eigenvalues * a / (2 * np.pi)
            ev_band.append(ev_norma)
            sims_.append(sim)
        # append first value since this is the same point
        ev_band.append(ev_band[0])
        BD[polarization] = ev_band
        sims[polarization] = sims_
    BD["TM"] = pt.backend.stack(BD["TM"]).real
    BD["TE"] = pt.backend.stack(BD["TE"]).real
    return BD, sims


bands, K = k_space_path(Nb=49)
BD, sims = full_model(bands, nh=100)


def k_space_path_plot(Nb, K):
    bands_plot = np.zeros(3 * Nb - 2)
    bands_plot[:Nb] = K
    bands_plot[Nb : 2 * Nb - 1] = K[-1] + K[1:]
    bands_plot[2 * Nb - 1 : 3 * Nb - 2] = 2 * K[-1] + 2**0.5 * K[1:]
    return bands_plot


bands_plot = k_space_path_plot(49, K)


plt.figure(figsize=(5.2, 2.5))
plotTE = plt.plot(bands_plot, BD["TE"], "-", c=cols[0], lw=1)

plt.xlim(0, bands_plot[-1])
plt.axis("off")

plotTM = plt.plot(bands_plot, BD["TM"], "-", c=cols[-1], lw=1)
plt.ylim(0, 2.5)
plt.xlim(0, bands_plot[-1])
# plt.axvline(K[-1], c="k", lw=0.3)
# plt.axvline(2 * K[-1], c="k", lw=0.3)
plt.tight_layout()
plt.axis("off")

plt.savefig(
    "bg_alt.svg",
    bbox_inches="tight",
    pad_inches=0,
)


# n = len(sims["TE"])
fig, ax = plt.subplots(12, 24, figsize=(4, 2))
ax = ax.ravel()
j = 0
for polarization in ["TE", "TM"]:
    print(j)
    ieig = 0
    for sim in sims[polarization]:
        if ieig >= sim.nh:
            ieig = 0
        print(ieig)
        v = sim.get_mode(ieig)
        ax[j].imshow(v.real, cmap=cmap)
        ax[j].set_axis_off()
        ieig += 1
        j += 1

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(
    "bg.svg",
    bbox_inches="tight",
    pad_inches=0,
)
