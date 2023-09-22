#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


"""
Reduced Bloch mode expansion
=============================

Calculation of the band diagram of a two-dimensional photonic crystal.
"""


import matplotlib.pyplot as plt
import numpy as np

import protis as pt

plt.close("all")
plt.ion()

##############################################################################
# Reference results are taken from :cite:p:`Hussein2009`.

a = 1
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)


##############################################################################
# Define the permittivity
epsilon = lattice.ones() * 1
hole = lattice.square((0.5, 0.5), 0.33)
epsilon[hole] = 11.4
mu = 1


##############################################################################
# Full model:


def full_model(bands, nh=100):
    t0 = pt.tic()
    sim = pt.Simulation(lattice, epsilon=epsilon, mu=mu, nh=nh)
    BD = {}
    for polarization in ["TE", "TM"]:
        ev_band = []
        for kx, ky in bands:
            print(kx, ky)
            sim.k = kx, ky
            sim.solve(polarization, vectors=False)
            ev_norma = sim.eigenvalues * a / (2 * np.pi)
            ev_band.append(ev_norma)
        # append first value since this is the same point
        # ev_band.append(ev_band[0])
        BD[polarization] = ev_band
    BD["TM"] = pt.backend.stack(BD["TM"]).real
    BD["TE"] = pt.backend.stack(BD["TE"]).real
    t_full = pt.toc(t0, verbose=False)
    return BD, t_full, sim


##############################################################################
# Reduced Bloch mode expansion


def rbme_model(bands, nh=100, Nmodel=2, N_RBME=8):
    t0 = pt.tic()
    sim = pt.Simulation(lattice, epsilon=epsilon, mu=mu, nh=nh)
    q = pt.pi / a
    if Nmodel == 2:
        bands_RBME = [(0, 0), (q, 0), (q, q)]
    elif Nmodel == 3:
        bands_RBME = [(0, 0), (q / 2, 0), (q, 0), (q, q / 2), (q, q), (q / 2, q / 2)]
    else:
        raise ValueError
    rbme = {
        polarization: sim.get_rbme_matrix(N_RBME, bands_RBME, polarization)
        for polarization in ["TE", "TM"]
    }
    BD_RBME = {}
    for polarization in ["TE", "TM"]:
        ev_band = []
        for kx, ky in bands:
            print(kx, ky)
            sim.k = kx, ky
            sim.solve(polarization, vectors=False, rbme=rbme[polarization])
            ev_norma = sim.eigenvalues * a / (2 * np.pi)
            ev_band.append(ev_norma)
        # append first value since this is the same point
        # ev_band.append(ev_band[0])
        BD_RBME[polarization] = ev_band
    BD_RBME["TM"] = pt.backend.stack(BD_RBME["TM"]).real
    BD_RBME["TE"] = pt.backend.stack(BD_RBME["TE"]).real
    t_rbme = pt.toc(t0, verbose=False)
    return BD_RBME, t_rbme, sim


bk = pt.backend
Nbz = 51
N_RBME = 8
Nmodel = 2

bandsx = bk.linspace(-pt.pi / a, pt.pi / a, Nbz)
bandsy = bk.linspace(-pt.pi / a, pt.pi / a, Nbz)

bandsx1, bandsy1 = bk.meshgrid(bandsx, bandsy, indexing="ij")
bands = bk.vstack([bandsx1.ravel(), bandsy1.ravel()]).T

# BD_RBME, t_rbme, sim_rbme = rbme_model(bands, nh=100, Nmodel=Nmodel, N_RBME=N_RBME)
# neig_rbme = BD_RBME["TE"].shape[-1]
# test = BD_RBME["TM"].reshape(Nbz,Nbz,neig_rbme)

BD, t_full, sim_full = full_model(bands, nh=100)

# neig_max = sim_full.nh

full_bd = {}
nmax = 6
for polarization in ["TE", "TM"]:
    # neig_rbme = BD[polarization].shape[-1]
    full_bd[polarization] = BD[polarization].reshape(Nbz, Nbz, sim_full.nh)
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    for i in range(nmax):
        ax.plot_surface(
            bandsx1,
            bandsy1,
            full_bd[polarization][:, :, i],
            cmap="viridis",
            edgecolor="none",
        )

    plt.title(polarization)

    ax.set_box_aspect([1, 1, 2])
    ax.set_zlim(0, 1.0)
    plt.tight_layout()
    plt.pause(0.01)

#
# BD, t_full, sim_full = full_model(bands, nh=100)
#
# for polarization in ["TE", "TM"]:
#     test = BD[polarization].reshape(Nbz,Nbz,sim_full.nh)
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#
#     nmax=6
#     for i in range(nmax):
#         ax.plot_surface(bandsx1, bandsy1, test[:,:,i],
#                         cmap='viridis', edgecolor='none')
#
#     plt.title(polarization)
# ax.set_box_aspect([1,1,2])
# ax.set_zlim(0,1.)
# plt.tight_layout()

polarization = "TM"
# polarization = "TE"

omega = full_bd[polarization]
#
# vx = bk.gradient(omega, axis=0)
# vy = bk.gradient(omega, axis=1)
#
# kx = bk.gradient(bandsx1, axis=0)
# ky = bk.gradient(bandsy1, axis=1)
#
#
# vx = bk.array([vx[:, :, i] / kx for i in range(vx.shape[-1])])
# vy = bk.array([vy[:, :, i] / kx for i in range(vy.shape[-1])])
# v = (vx**2 + vy**2) ** 0.5
#

from scipy.interpolate import RectBivariateSpline

Nbz_new = Nbz * 4
tol_omega = 1e-4  # kx[0,0]/Nbz_new
iband = 0
k_interpx, k_interpy = 3, 3


# OMEGAS = bk.linspace(omega[:,:,iband].min(),omega[:,:,iband].max(),130)
OMEGAS = bk.linspace(0.0, 1, 1000)


bandsx_new = bk.linspace(-pt.pi / a, pt.pi / a, Nbz_new)
bandsy_new = bk.linspace(-pt.pi / a, pt.pi / a, Nbz_new)
bandsx_new1, bandsy_new1 = bk.meshgrid(bandsx_new, bandsy_new, indexing="ij")

nmax = 8

# plt.close("all")


DOS = []
for iband in range(nmax):
    print(iband)
    interp_omega = RectBivariateSpline(
        bandsx, bandsy, omega[:, :, iband], kx=k_interpx, ky=k_interpy
    )
    omega0_new = interp_omega(bandsx_new, bandsy_new)

    # interp_v = RectBivariateSpline(bandsx, bandsy, v[iband], kx=k_interpx, ky=k_interpy)
    # v_new = interp_v(bandsx_new, bandsy_new)
    #

    vx = interp_omega(bandsx_new, bandsy_new, dx=1, dy=0)
    vy = interp_omega(bandsx_new, bandsy_new, dx=0, dy=1)
    v_new = (vx**2 + vy**2) ** 0.5

    # plt.clf()
    # plt.imshow(v_new)
    # plt.pause(0.01)
    #
    DOS_i = []

    for omega_i in OMEGAS:
        # print(omega_i)
        cond = bk.abs(omega0_new - omega_i) < tol_omega
        if not bk.any(cond):
            I = 0
        else:
            u = bk.where(cond, 1 / (1e-3 + v_new), 0)
            I = bk.trapz(bk.trapz(u, bandsy_new), bandsx_new)
        DOS_i.append(I)
        # plt.clf()
        # u = bk.where(bk.abs(omega0_new - omega_i)<tol_omega,v_new,0)
        # plt.figure()
        # plt.imshow(u)
    DOS.append(DOS_i)

DOS = bk.array(DOS)

DOS_TOT = bk.sum(DOS, axis=0)

# plt.close("all")
# plt.figure()


plt.plot(OMEGAS, DOS_TOT / (2 * bk.pi / a) ** 2, "-")
#
# plt.figure()
# plt.plot(OMEGAS, DOS_TOT , "-")
# plt.plot(OMEGAS, OMEGAS*3 , "-")
# plt.plot(OMEGAS, DOS_TOT / (2 * bk.pi / a) ** 2, "-")
