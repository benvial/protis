#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


"""
Calculating topological invariants
===================================

2D photonic crystal comprised of a square lattice of
yttrium-iron-garnet (YIG) rods for transverse-magnetic modes.
"""


import matplotlib.pyplot as plt
import numpy as np

import protis as pt

# plt.ion()
# plt.close("all")


##############################################################################
# Reference results are taken from  :cite:p:`Joannopoulos2008` (Chapter 5 Fig. 2).

a = 1
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
r = 0.11 * a
polarization = "TM"

##############################################################################
# Define the permittivity and permeability

rod = lattice.circle(center=(0.5, 0.5), radius=r)
epsilon = lattice.ones()
epsilon[rod] = 15
muxx = lattice.ones()
muxx[rod] = 14
muyy = lattice.ones()
muyy[rod] = 14
muxy = lattice.ones() * 0
muxy[rod] = 12.4j
muyx = lattice.ones() * 0
muyx[rod] = -12.4j
muzz = lattice.ones()

mu = pt.block_z_anisotropic(muxx, muxy, muyx, muyy, muzz)
# mu = pt.block_z_anisotropic(1, 0, 0, 1, 1)
# mu=np.eye(3)

# mu = muxx

# test = pt.get_block(mu[:2,:2], 0, 0, 2**8)
#
# xxs
##############################################################################
# We define here the wavevector path:

Nb = 21
K = np.linspace(0, np.pi / a, Nb)
bands = np.zeros((3 * Nb - 3, 2))
bands[:Nb, 0] = K
bands[Nb : 2 * Nb, 0] = K[-1]
bands[Nb : 2 * Nb - 1, 1] = K[1:]
bands[2 * Nb - 1 : 3 * Nb - 3, 0] = bands[2 * Nb - 1 : 3 * Nb - 3, 1] = np.flipud(K)[
    1:-1
]


##############################################################################
# Calculate the band diagram:

sim = pt.Simulation(lattice, epsilon=epsilon, mu=mu, nh=250)

band_diag = []
for kx, ky in bands:
    sim.k = kx, ky
    sim.solve(polarization, vectors=False)
    ev_norma = sim.eigenvalues * a / (2 * np.pi)
    band_diag.append(ev_norma)
# append first value since this is the same point
band_diag.append(band_diag[0])


bands_plot = np.zeros(3 * Nb - 2)
bands_plot[:Nb] = K
bands_plot[Nb : 2 * Nb - 1] = K[-1] + K[1:]
bands_plot[2 * Nb - 1 : 3 * Nb - 2] = 2 * K[-1] + 2**0.5 * K[1:]

band_diag = pt.backend.stack(band_diag).real


##############################################################################
# Plot the bands:

plt.figure(figsize=(2.2, 3.2))

plotTM = plt.plot(bands_plot, band_diag, c="#4199b0")
plt.ylim(0, 0.8)
plt.xlim(0, bands_plot[-1])
plt.xticks(
    [0, K[-1], 2 * K[-1], bands_plot[-1]], ["$\Gamma$", "$X$", "$M$", "$\Gamma$"]
)
plt.axvline(K[-1], c="k", lw=0.3)
plt.axvline(2 * K[-1], c="k", lw=0.3)
plt.ylabel(r"Frequency $\omega a/2\pi c$")
plt.tight_layout()
plt.show()
