#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io

import os
import tempfile
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import protis as pt

plt.ion()
plt.close("all")
warnings.filterwarnings("ignore")

tmpdir = tempfile.mkdtemp(dir=".")


def savefig(base, it):
    name = tmpdir + "/" + base + str(it).zfill(4) + ".png"
    plt.savefig(name)


def pltpause(interval):
    backend = plt.rcParams["backend"]
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


# backend = "autograd"
backend = "torch"
pt.set_backend(backend)

bk = pt.backend
no = pt.optimize


def symmetrize_pattern(dens, x=True, y=True, s8=True):
    dens = bk.array(dens)
    if x:
        dens = 0.5 * (dens + bk.fliplr(dens))
    if y:
        dens = 0.5 * (dens + bk.flipud(dens))
    if s8:
        dens = 0.5 * (dens + dens.T)
    return dens


def k_space_path(Nb, a):
    K = np.linspace(0, np.pi / a, Nb)
    bands = np.zeros((3 * Nb - 3, 2))
    bands[:Nb, 0] = K
    bands[Nb : 2 * Nb, 0] = K[-1]
    bands[Nb : 2 * Nb - 1, 1] = K[1:]
    bands[2 * Nb - 1 : 3 * Nb - 3, 0] = bands[2 * Nb - 1 : 3 * Nb - 3, 1] = np.flipud(
        K
    )[1:-1]
    return bands, K


def k_space_path_plot(Nb, K):
    bands_plot = np.zeros(3 * Nb - 2)
    bands_plot[:Nb] = K
    bands_plot[Nb : 2 * Nb - 1] = K[-1] + K[1:]
    bands_plot[2 * Nb - 1 : 3 * Nb - 2] = 2 * K[-1] + 2**0.5 * K[1:]
    return bands_plot


a = 1
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**8)

eps_min, eps_max = 1, 9

nh = 300
Nb = 101
polarization = "TM"

ieig = 4
center_target = 0.9
alpha = 0.01
rfilt = lattice.discretization[0] / 100
nvar = lattice.discretization[0] * lattice.discretization[1]

r0 = 0.2
x0 = lattice.ones() * 1
hole = lattice.circle(center=(0.5, 0.5), radius=r0)
x0[hole] = 0

# if polarization =="TM":
#     x0 = 1-x0

# x0 = bk.ones(nvar) * 0.5

x0 = bk.array(np.random.rand(nvar))
x0 = bk.reshape(x0, lattice.discretization)
x0 = no.apply_filter(x0, rfilt * 2)
x0 = (x0 - x0.min()) / (x0.max() - x0.min())
x0 = symmetrize_pattern(x0).ravel()

x0 = x0.real.ravel()

bands, K = k_space_path(Nb, a)
bands_plot = k_space_path_plot(Nb, K)
#
# bands = bands[:Nb]
# bands_plot = bands_plot[:Nb]

type_opt = "dispersion"
type_opt = "bandgap"
# type_opt = "dirac"
index1 = slice(0, Nb)
index2 = slice(0)  # slice(None, None)
bands_plot = bk.array(bands_plot)
ks = bk.hstack([bands_plot[index1], bands_plot[index2]])

dispersion = (
    -0.01 * bk.cos(ks * a) + 0.004 * bk.cos(2 * ks * a) - 0.002 * bk.cos(3 * ks * a)
)
omega_c = None

stopval = 1e-6 if type_opt == "dispersion" else None

# fig, ax = plt.subplots(1)

fig_bg, ax_bg = plt.subplots(1, figsize=(3.2, 3.5))
# ax_ins = inset_axes(ax_bg, width="18%", height="18%", loc=8, borderpad=1)

ax_ins = inset_axes(
    ax_bg,
    width="100%",
    height="100%",
    bbox_to_anchor=(0.35, 0.01, 0.16, 0.16),
    bbox_transform=ax_bg.transAxes,
    loc=8,
)

color = "#cf5268" if polarization == "TE" else "#4199b0"
cmap = "Reds" if polarization == "TE" else "Blues"
##############################################################################


def rbme_model(bands, epsilon, polarization, nh=nh, Nmodel=2, N_RBME=8):

    sim = pt.Simulation(lattice, epsilon=epsilon, mu=1, nh=nh)
    q = pt.pi / a
    if Nmodel == 2:
        bands_RBME = [(0, 0), (q, 0), (q, q)]
    elif Nmodel == 3:
        bands_RBME = [(0, 0), (q / 2, 0), (q, 0), (q, q / 2), (q, q), (q / 2, q / 2)]
    else:
        raise ValueError
    rbme = {}
    rbme[polarization] = sim.get_rbme_matrix(N_RBME, bands_RBME, polarization)

    ev_band = []
    for kx, ky in bands:
        sim.k = kx, ky
        sim.solve(polarization, vectors=False, rbme=rbme[polarization])
        ev_norma = sim.eigenvalues * a / (2 * np.pi)
        ev_band.append(ev_norma)
    # append first value since this is the same point

    ev_band.append(ev_band[0])
    BD_RBME = pt.backend.stack(ev_band).real
    return BD_RBME, sim


def simu(x, proj_level=None, rfilt=0, return_bg=False):
    global omega_c
    dens = bk.reshape(x, lattice.discretization)

    dens = symmetrize_pattern(dens)
    density_f = no.apply_filter(dens, rfilt)
    density_fp = (
        no.project(density_f, proj_level) if proj_level is not None else density_f
    )
    epsilon = no.simp(density_fp, eps_min, eps_max, p=1)

    kx, ky = 0, 0
    BD_RBME, sim = rbme_model(bands, epsilon, polarization, nh=nh, Nmodel=2, N_RBME=8)

    # BG = BG/center

    if type_opt == "dispersion":
        test = bk.hstack([BD_RBME[index1, ieig], BD_RBME[index2, ieig]])
        omega_c = bk.mean(test)
        objective = bk.sum((test - dispersion - omega_c) ** 2)
    else:

        # objective1 = -width/center_target * alpha
        # objective2 = (center - center_target)**2/center_target**2
        # objective = objective1 * alpha + objective2* (1 - alpha)

        width = bk.min(BD_RBME[:, ieig + 1]) - bk.max(BD_RBME[:, ieig])
        center = (bk.min(BD_RBME[:, ieig + 1]) + bk.max(BD_RBME[:, ieig])) / 2
        objective = bk.abs(center - center_target) / width
        objective = 1 / width
        # objective = bk.abs(center - center_target) / width

        # objective = (BD_RBME[2*Nb, ieig ] - BD_RBME[2*Nb, ieig + 1])**2 #+ (BD_RBME[:, ieig -1] - BD_RBME[:, ieig ])**2

        # objective = 1/width

    is_grad_pass = objective.grad_fn is not None

    if not is_grad_pass:
        BD_plot = BD_RBME.detach().numpy()
        density_fp_plot = density_fp.detach().numpy()
        plt.sca(ax_bg)
        ax_bg.clear()
        plot_bd(BD_plot, density_fp_plot, ax_bg, ax_ins)

    pltpause(0.1)

    if return_bg:
        return objective, BD_RBME
    else:
        return objective


##############################################################################


def bg_metrics(BG, verbose=False):
    BG = bk.array(BG)
    width = bk.min(bk.real(BG[:, ieig + 1]) - bk.max(BG[:, ieig]))
    center = (bk.min(bk.real(BG[:, ieig + 1])) + bk.max(bk.real(BG[:, ieig]))) / 2
    if verbose:
        print("bandgap center", center.tolist())
        print("bandgap width", width.tolist())
        print("relative width", (width / center).tolist())
    return width, center


itplot = 0


def plot_bd(BD_RBME, density_fp, ax, ax_ins):
    global itplot

    ax.set_title(f"{polarization} modes", c=color)
    if type_opt == "dispersion":
        # omega_c = bk.mean(bk.array(BD_RBME[:, ieig]))
        test = np.hstack([BD_RBME[index1, ieig], BD_RBME[index2, ieig]])
        omega_c = np.mean(test)
        ax_bg.plot(ks, dispersion + omega_c, "--k")
        ax_bg.set_ylim(0.9 * omega_c, 1.1 * omega_c)
    else:
        width, center = bg_metrics(BD_RBME, verbose=True)
        y1 = np.min(BD_RBME[:, ieig + 1].real)
        y0 = np.max(BD_RBME[:, ieig].real)
        # print(y0,y1)
        if y1 > y0:
            ax.fill_between(bands_plot, y1, y0, alpha=0.1, color=color, lw=0)
            ax.hlines(
                center,
                bands_plot[0],
                bands_plot[-1],
                alpha=0.5,
                color=color,
                lw=0.4,
                ls="--",
            )
            ax.annotate(
                rf"$\Delta\omega/\omega_0={100*width/center:0.1f}$%",
                (bands_plot[1], center * (1.02)),
            )
            ax.annotate(
                r"$\tilde{\omega_0}=" + rf"{center:0.3f}$",
                (bands_plot[-1] * 0.84, center * (1.02)),
            )
        ax.set_ylim(0, 1.2)
    plotTE_RBME = ax.plot(bands_plot, BD_RBME, "-", c=color)

    ax.set_xlim(0, bands_plot[-1])
    ax.set_xticks(
        [0, K[-1], 2 * K[-1], bands_plot[-1]], ["$\Gamma$", "$X$", "$M$", "$\Gamma$"]
    )
    ax.axvline(K[-1], c="k", lw=0.3)
    ax.axvline(2 * K[-1], c="k", lw=0.3)
    # ax.set_xticks([0, K[-1]], ["$\Gamma$", "$X$"])
    ax.set_ylabel(r"Normalized frequency $\tilde{\omega} = \omega a/2\pi c$")
    # ax.legend([plotTE_RBME[0]], ["full", "2-point RBME"], loc=(0.31, 0.02))

    plt.tight_layout()

    ax_ins.clear()
    # plt.sca(ax)
    ax_ins.imshow(density_fp, cmap=cmap)
    ax_ins.set_axis_off()

    savefig("bd", itplot)
    itplot += 1


it = 0


def callback(x, y, proj_level, rfilt):
    global it
    print(f"iteration {it}")
    dens = bk.reshape(x, lattice.discretization)
    dens = symmetrize_pattern(dens)
    density_f = no.apply_filter(dens, rfilt)
    density_fp = no.project(density_f, proj_level)
    # plt.figure()
    ax.clear()
    plt.sca(ax)
    plt.imshow(density_fp, cmap="Reds")
    ax.axis("off")
    plt.suptitle(f"iteration {it}, objective = {y:.5f}")
    # plt.tight_layout()
    pltpause(0.1)
    it += 1


callback = None

opt = no.TopologyOptimizer(
    simu,
    x0,
    method="nlopt",
    maxiter=20,
    stopval=stopval,
    args=(1, rfilt),
    options={},
    callback=callback,
    verbose=True,
)

x_opt, f_opt = opt.minimize()
x_opt = bk.array(x_opt)

dens = bk.reshape(x_opt, lattice.discretization)

dens = symmetrize_pattern(dens)
density_f = no.apply_filter(dens, rfilt)
density_bin = density_f
density_bin[density_f < 0.5] = 0
density_bin[density_f >= 0.5] = 1
x_bin = density_bin.ravel()

epsilon_bin = no.simp(density_bin, eps_min, eps_max, p=1)

obj, BD_RBME = simu(x_bin, proj_level=None, rfilt=0, return_bg=True)

from nannos.plot import plot_layer

grid = lattice.unit_grid
# bk.stack((grid[0], grid[1]))
lattice.basis_vectors = bk.array(lattice.basis_vectors, dtype=bk.float32)
plt.figure()
ims = plot_layer(
    lattice, grid, epsilon_bin.real, nper=4, cmap=cmap, show_cell=True, cellstyle="y--"
)
plt.axis("off")
savefig("final_bd", it)


from protis.isocontour import get_isocontour

x = bk.linspace(0, 1, lattice.discretization[0])
y = bk.linspace(0, 1, lattice.discretization[1])

from skimage import measure

# contours = get_isocontour(*grid, density_bin.numpy(),0.)
d = lattice.discretization

im = density_bin.numpy()


npad = 5
im1 = np.pad(im, (npad, npad), constant_values=1)

contours = measure.find_contours(im1, 0.0, fully_connected="high")

x1 = np.pad(x, (npad, npad), constant_values=(-npad * a / d[0], a + npad * a / d[0]))
y1 = np.pad(y, (npad, npad), constant_values=(-npad * a / d[1], a + npad * a / d[1]))


for contour in contours:
    contour[:, 0] = (
        (x.max() - x.min()) * contour[:, 0] / len(x) + x.min() - npad * a / d[0]
    )
    contour[:, 1] = (
        (y.max() - y.min()) * contour[:, 1] / len(y) + y.min() - npad * a / d[1]
    )

    contour[contour > 0.99] = 1
    contour[contour < 0.01] = 0


plt.figure()
plt.pcolormesh(x1, y1, im1, cmap="Reds")
plt.axis("off")
plt.axis("scaled")


plt.figure()
plt.pcolormesh(x, y, im, cmap="Reds")
plt.axis("off")
plt.axis("scaled")

for contour in contours:
    plt.plot(contour[:, 0], contour[:, 1], "-b")
    plt.tight_layout()


np.savez("contour.npz", contours=contours, density_bin=density_bin)


os.system(f"convert -delay 20 -loop 0 {tmpdir}/bd*.png {tmpdir}/animation_bd.gif")
os.system(f"convert {tmpdir}/animation_bd.gif {tmpdir}/animation_bd.mp4")
