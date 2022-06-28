#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import sys

import matplotlib.pyplot as plt
import numpy as np

import protis as pt

pt.set_backend("torch")
from protis.fft import inverse_fourier_transform
from protis.utils import is_anisotropic, is_scalar

bk = pt.backend

no = pt.optimize
plt.close("all")
plt.ion()

polarization = "TE"
a = 1
R = 0.25 * a
nh = 200

lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
epsilon = lattice.ones() * 1
# rod = lattice.circle(center=(0.5, 0.5), radius=0.2)
rod = lattice.ellipse(center=(0.5, 0.5), radii=(R, R * 1.5), rotate=180 / 6)
epsilon[rod] = 9
mu = 1


################################################################
# Optimization parameters
Nx, Ny = lattice.discretization
# plots = True
rfilt = Nx / 50
maxiter = 40
threshold = (0, 12)
stopval = 1e-6
eps_min, eps_max = 1, 9
mode_index = 3
point = "X"
Txx_target = -3
Tyy_target = 11

x, y = lattice.grid

sim = pt.Simulation(lattice, epsilon=epsilon, mu=mu, nh=nh)

norm_eigval = (2 * np.pi) / a

##############################################################################

Nb = 121
K = bk.linspace(0, np.pi / a, Nb)
bands = bk.zeros((3 * Nb - 3, 2))
bands[:Nb, 0] = K
bands[Nb : 2 * Nb, 0] = K[-1]
bands[Nb : 2 * Nb - 1, 1] = K[1:]
bands[2 * Nb - 1 : 3 * Nb - 3, 0] = bands[2 * Nb - 1 : 3 * Nb - 3, 1] = bk.flipud(K)[
    1:-1
]

Nhom = int(Nb / 2)

n_eig_hom = 6
##############################################################################
# Calculate the band diagram:


def symmetrize_pattern(dens, x=True, y=True, s8=False):
    dens = bk.array(dens)
    if x:
        dens = 0.5 * (dens + bk.fliplr(dens))
    if y:
        dens = 0.5 * (dens + bk.flipud(dens))
    if s8:
        dens = 0.5 * (dens + dens.T)
    return dens


def compute_bands(sim):
    ev_band = []
    for kx, ky in bands:
        sim.k = kx, ky
        sim.solve(polarization, vectors=False)
        ev_norma = sim.eigenvalues / norm_eigval
        ev_band.append(ev_norma)
    # append first value since this is the same point
    ev_band.append(ev_band[0])

    return bk.stack(ev_band).real


bands_plot = bk.zeros(3 * Nb - 2)
bands_plot[:Nb] = K
bands_plot[Nb : 2 * Nb - 1] = K[-1] + K[1:]
bands_plot[2 * Nb - 1 : 3 * Nb - 2] = 2 * K[-1] + 2**0.5 * K[1:]

bands_hom = bk.vstack((bands, bands[0]))

# # ##############################################################################
# # # Plot the bands:


def plot_bd(BD):

    plt.figure(figsize=(3.2, 2.5))

    plot = plt.plot(bands_plot, BD, c="#4199b0")
    plt.ylim(0, 1.5)
    plt.xlim(0, bands_plot[-1])
    plt.xticks(
        [0, K[-1], 2 * K[-1], bands_plot[-1]], ["$\Gamma$", "$X$", "$M$", "$\Gamma$"]
    )
    plt.axvline(K[-1], c="k", lw=0.3)
    plt.axvline(2 * K[-1], c="k", lw=0.3)
    plt.ylabel(r"Frequency $\omega a/2\pi c$")
    plt.tight_layout()
    return plot


# print(bk.linalg.inv(T))


def init_point(point):
    if point == "Gamma":
        propagation_vector = (0, 0)
        bands_hom_1 = [bands_hom[:Nhom], bands_hom[-Nhom:]]
        bands_plot_1 = [bands_plot[:Nhom], bands_plot[-Nhom:]]
    elif point == "X":
        propagation_vector = (bk.pi / a, 0)
        i1, i2 = Nb - 1 - int(Nhom / 2), Nb - 1 + int(Nhom / 2)
        bands_hom_1 = [bands_hom[i1:i2]]
        bands_plot_1 = [bands_plot[i1:i2]]
    elif point == "M":
        propagation_vector = (bk.pi / a, bk.pi / a)
        i1, i2 = 2 * Nb - 2 - int(Nhom / 2), 2 * Nb - 2 + int(Nhom / 2)
        bands_hom_1 = [bands_hom[i1:i2]]
        bands_plot_1 = [bands_plot[i1:i2]]
    else:
        raise (ValueError)
    return propagation_vector, bands_hom_1, bands_plot_1


def compute_hfhom(sim, special_points, modes_indexes, polarization, plot=True):

    hom = dict()
    for point in special_points:
        print("**************")
        print(f"{point} point")
        print("**************")

        propagation_vector, bands_hom_1, bands_plot_1 = init_point(point)

        sim.k = propagation_vector
        sim.solve(polarization)

        hom[point] = {}
        eigenfreqs = []
        Teff = []

        for mode_index in modes_indexes:
            k0 = sim.eigenvalues[mode_index].real
            print(f"mode index = {mode_index}")
            print(f"eigenfrequency = {k0}")
            print("----------")
            T = sim.get_hfh_tensor(mode_index, polarization)
            print(T)

            eigenfreqs.append(k0)
            Teff.append(T)

            for b, bp in zip(bands_hom_1, bands_plot_1):
                kappa = [b[:, i] - propagation_vector[i] for i in range(2)]
                q = sum(
                    [T[i, j] * kappa[i] * kappa[j] for i in range(2) for j in range(2)]
                )
                omega_hom = (k0**2 + q) ** 0.5 / norm_eigval
                # omega_hom = (k0 + (T[0,0] * bands[:,0]**2 + T[1,1] * bands[:,1]**2)/(2*k0)) / norm_eigval

                plt.plot(bp, omega_hom, "--", c="#cf5268", lw=1)
                plt.pause(0.1)
                # sys.exit(0)

        hom[point]["k0"] = eigenfreqs
        hom[point]["T"] = Teff

    return hom


special_points = ["Gamma", "X", "M"]


# BD = compute_bands(sim)
# plot_bd(BD)

# hom = compute_hfhom(sim,special_points,range(n_eig_hom),polarization)

# sys.exit(0)
# # plt.savefig(f"bd_{polarization}.eps")
# # plt.savefig(f"bd_{polarization}.png")


propagation_vector, bands_hom_1, bands_plot_1 = init_point(point)


def simu(x, proj_level=None, rfilt=0, sym=True):
    global T
    dens = bk.reshape(x, lattice.discretization)
    if sym:
        dens = symmetrize_pattern(dens)
    density_f = no.apply_filter(dens, rfilt)
    density_fp = (
        no.project(density_f, proj_level) if proj_level is not None else density_f
    )
    epsilon = no.simp(density_fp, eps_min, eps_max, p=1)
    sim = pt.Simulation(lattice, epsilon=epsilon, mu=1, nh=nh)
    sim.k = propagation_vector
    sim.solve(polarization)
    T = sim.get_hfh_tensor(mode_index, polarization)
    # objective = bk.abs(1 + T[1, 1].real / T[0, 0].real) ** 2

    objective = bk.abs(T[0, 0].real - Txx_target) ** 2
    objective += bk.abs(T[1, 1].real - Tyy_target) ** 2

    objective += abs(T[1, 0]) ** 2 + abs(T[0, 1]) ** 2
    nev = sim.eigenvalues / norm_eigval
    alpha = 0.01

    # objective -= alpha * bk.abs(nev[mode_index] - nev[mode_index + 1]) ** 2
    # objective -= alpha * bk.abs(nev[mode_index - 1] - nev[mode_index]) ** 2
    is_grad_pass = x.grad_fn is not None

    if not is_grad_pass:
        # print(T[0, 0].real.tolist(), T[1, 1].real.tolist())
        print(T.real.tolist())
        print(objective.tolist())
        evals = nev[mode_index - 1 : mode_index + 2].real.tolist()
        print(evals)

    return objective


it = 0


def callback(x, y, proj_level, rfilt):
    global it

    dens = bk.reshape(x, lattice.discretization)
    dens = symmetrize_pattern(dens)
    density_f = no.apply_filter(dens, rfilt)
    density_fp = (
        no.project(density_f, proj_level) if proj_level is not None else density_f
    )
    epsilon = no.simp(density_fp, eps_min, eps_max, p=1)

    plt.clf()

    plt.pcolormesh(density_fp, cmap="Greens", vmin=0, vmax=1)
    plt.colorbar()
    plt.axis("scaled")
    plt.axis("off")
    plt.title(
        rf"$T_{{xx}} = {T[0,0]:0.3f}$, $T_{{yy}} = {T[1,1]:0.3f}$, $\Phi={y:0.3e}$"
    )
    plt.tight_layout()
    plt.pause(0.01)

    it += 1
    return y


nvar = Nx * Ny
x0 = bk.array(np.random.rand(nvar))
x0 = bk.reshape(x0, lattice.discretization)
x0 = no.apply_filter(x0, 10 * rfilt)
x0 = no.project(x0, 2**10)
x0 = symmetrize_pattern(x0).ravel()

x0 = lattice.ones()
R = 0.2
# rod = lattice.circle(center=(0.5, 0.5), radius=0.2)
rod = lattice.ellipse(center=(0.5, 0.5), radii=(R, R * 1), rotate=0 * 180 / 6)
x0[rod] = 0
x0 = 1 - x0
x0 = bk.real(x0).ravel()


opt = no.TopologyOptimizer(
    simu,
    x0,
    method="nlopt",
    maxiter=maxiter,
    threshold=threshold,
    stopval=stopval,
    args=(1, rfilt),
    options={},
    callback=callback,
    verbose=True,
)

x_opt, f_opt = opt.minimize()

xbin = bk.array(x_opt.copy())
xbin = bk.reshape(xbin, lattice.discretization)
xbin = symmetrize_pattern(xbin)
xbin = no.apply_filter(xbin, rfilt)
xbin = no.project(xbin, 2 ** (threshold[-1] - 1))
xbin[xbin <= 0.5] = 0
xbin[xbin > 0.5] = 1
xbin = xbin.ravel()


objfinal = simu(xbin, sym=False)
callback(xbin, objfinal, None, 0)


xbin_array = bk.reshape(xbin, lattice.discretization)

epsilon = no.simp(xbin_array, eps_min, eps_max, p=1)
sim = pt.Simulation(lattice, epsilon=epsilon, mu=1, nh=nh)
sim.k = propagation_vector
BD = compute_bands(sim)
plot_bd(BD)
hom = compute_hfhom(sim, [point], [mode_index], polarization)
