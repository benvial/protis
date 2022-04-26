#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


"""
Band diagram of 2D photonic crystal
===================================

Calculation of the band diagram of a two-dimensional photonic crystal.
"""



import matplotlib.pyplot as plt
import numpy as np
import protis as pt


##############################################################################
# Reference results are taken from  :cite:p:`Joannopoulos2008` (Chapter 5 Fig. 2).
#
# The structure is a square lattice of dielectric
# columns, with radius r and dielectric constant :math:`\varepsilon`.
# The material is invariant along the z direction  and periodic along
# :math:`x` and :math:`y` with lattice constant :math:`a`.
# We will define the lattie using the class :class:`~protis.Lattice`

a = 1
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)


##############################################################################
# Define the permittivity
epsilon = lattice.ones() * 1
hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
epsilon[hole] = 8.9

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

nh = 100
BD = {}
for polarization in ["TE", "TM"]:
    ev_band = []
    q = 0
    for kx, ky in bands:
        sim = pt.Simulation(lattice, (kx, ky), epsilon=epsilon,mu=1, nh=100)
        a = sim.lattice.basis_vectors[0][0]
        neig = 6
        k0, v = sim.solve(polarization)
        ev_norma = k0[:neig] * a / (2 * np.pi)
        # plt.plot(q*np.ones(neig),ev_norma,"ob")
        ev_band.append(ev_norma)
        
        

        q += (kx**2 + ky**2) ** 0.5
        plt.pause(0.1)
    # append first value since this is the same point
    ev_band.append(ev_band[0])

    BD[polarization] = ev_band

bands_plot = np.zeros(3 * Nb - 2)
bands_plot[:Nb] = K
bands_plot[Nb : 2 * Nb - 1] = K[-1] + K[1:]
bands_plot[2 * Nb - 1 : 3 * Nb - 2] = 2 * K[-1] + 2**0.5 * K[1:]


BD["TM"] = pt.backend.stack( BD["TM"]).real
BD["TE"] = pt.backend.stack( BD["TE"]).real


##############################################################################
# Plot the bands:

plt.figure(figsize=(3.2, 2.5))

plotTM = plt.plot(bands_plot, BD["TM"], c="#4199b0")
plotTE = plt.plot(bands_plot, BD["TE"], c="#cf5268")
plt.annotate("TM modes", (1, 0.05), c="#4199b0")
plt.annotate("TE modes", (0.33, 0.33), c="#cf5268")
plt.ylim(0, 0.8)
plt.xlim(0, bands_plot[-1])
plt.xticks(
    [0, K[-1], 2 * K[-1], bands_plot[-1]], ["$\Gamma$", "$X$", "$M$", "$\Gamma$"]
)
plt.axvline(K[-1], c="k", lw=0.3)
plt.axvline(2 * K[-1], c="k", lw=0.3)
plt.ylabel(r"Frequency $\omega a/2\pi c$")
plt.tight_layout()
