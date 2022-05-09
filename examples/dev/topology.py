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

pt.set_backend("scipy")

plt.ion()
plt.close("all")


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

sim = pt.Simulation(lattice, epsilon=epsilon, mu=mu, nh=100)

band_diag = []
for kx, ky in bands:
    sim.k = kx, ky
    sim.solve(polarization, vectors=False, sparse=True, neig=6, sigma=1e-12)
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

#
# ##############################################################################
# # Berry phase
# bk = pt.backend
#
#
# def inner(phi1, phi2, coeff, x, y):
#     return bk.trapz(bk.trapz(bk.conj(phi1) * coeff * phi2, y, axis=-1), x, axis=-1)
#
#
# def moment_x(phi1, phi2, coeff, x, y):
#     return bk.trapz(bk.trapz(x * bk.conj(phi1) * coeff * phi2, y, axis=-1), x, axis=-1)
#
#
# x, y = lattice.grid
# x = lattice.grid[0, :, 0]
# y = lattice.grid[1, 0, :]
# coeff = sim.epsilon
#
# nk = 20
# nmode = 3
# Kx = np.linspace(-pt.pi / a, pt.pi / a, nk)
# Ky = np.linspace(-pt.pi / a, pt.pi / a, nk)
# thetan_kx = []
# for ikx, kx in enumerate(Kx):
#     thetan_ky = []
#     thetan_ky = bk.zeros((nmode, nk), dtype=complex)
#     for iky, ky in enumerate(Ky):
#         print(ikx, iky)
#         sim.k = kx, ky
#         sim.solve(polarization, vectors=True, sparse=True, neig=3, sigma=1e-12)
#         for imode in range(nmode):
#             phi = sim.get_mode(imode)
#             norma = inner(phi, phi, coeff, x, y)
#             phi /= norma**0.5
#             thetan_ky[imode, iky] = moment_x(phi, phi, coeff, x, y)
#     thetan_kx.append(bk.trapz(thetan_ky, Ky))
#
# thetan_kx = bk.array(thetan_kx)
#
# plt.figure()
# plt.plot(Kx, thetan_kx)
#
#
#
#
# ###############################################
#
# def inner(phi1, phi2, x, y):
#     return bk.trapz(bk.trapz(bk.conj(phi1) * coeff * phi2, y, axis=-1), x, axis=-1)
#
#
#
# nk = 20
# nmode = 3
#
# x, y = lattice.grid
# x = lattice.grid[0, :, 0]
# y = lattice.grid[1, 0, :]
# Kx = np.linspace(-pt.pi / a, pt.pi / a, nk)
# Ky = np.linspace(-pt.pi / a, pt.pi / a, nk)
# modes = bk.zeros((nk, nk, nmode,*lattice.discretization), dtype=complex)
#
# for ikx, kx in enumerate(Kx):
#     for iky, ky in enumerate(Ky):
#         print(ikx, iky)
#         sim.k = kx, ky
#         sim.solve(polarization, vectors=True, sparse=True, neig=3, sigma=1e-12)
#         for imode in range(nmode):
#             phi = sim.get_mode(imode)
#             norma = inner(phi, phi, x, y)
#             phi /= norma**0.5
#             modes[ikx, iky, imode] = phi
#
#
#
# Ms = []
# for ikx, kx in enumerate(Kx):
#     M = bk.zeros((nk,nmode,nmode), dtype=complex)
#     for iky, ky in enumerate(Ky):
#         print(ikx, iky)
#         for imode in range(nmode):
#             phi1 = modes[ikx, iky, imode]
#             for jmode in range(nmode):
#                 if iky == nk-1:
#                     break
#                 phi2 = modes[ikx, iky+1, jmode]
#                 M[iky,imode,jmode] = inner(phi1, phi2, x, y)
#     Ms.append(M)
#
# thetas = []
# for ikx, M in enumerate(Ms):
#     W = bk.eye(nmode)
#     for iky, ky in enumerate(Ky):
#         if iky == nk-1:
#             break
#         W = W @ M[iky]
#     w = bk.linalg.eigvals(W)
#     theta = -bk.imag(bk.log(w))
#     thetas.append(theta)
#
#
# thetas = bk.array(thetas)
#
# plt.figure()
# plt.plot(Kx, thetas)
