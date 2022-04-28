#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np

import protis as pt

plt.ion()
plt.close("all")


backend = "autograd"
backend = "torch"
pt.set_backend(backend)

bk = pt.backend
no = pt.optimize


a = 1
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**8)


eps_min, eps_max = 1, 9

ieig = 4
Nb = 21

polarization = "TM"

rfilt = lattice.discretization[0] / 50

nvar = lattice.discretization[0] * lattice.discretization[1]

r0 = 0.5
x0 = lattice.ones() * 1
hole = lattice.circle(center=(0.5, 0.5), radius=r0)
x0[hole] = 0


x0 = x0.real.ravel()
##############################################################################


def symmetrize_pattern(dens, x=True, y=True, s8=False):
    dens = bk.array(dens)
    if x:
        dens = 0.5 * (dens + bk.fliplr(dens))
    if y:
        dens = 0.5 * (dens + bk.flipud(dens))
    if s8:
        dens = 0.5 * (dens + dens.T)
    return dens


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


bands, K = k_space_path(Nb=Nb)


def rbme_model(bands, epsilon, polarization, nh=100, Nmodel=2, N_RBME=8):

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
    dens = bk.reshape(x, lattice.discretization)

    dens = symmetrize_pattern(dens)
    density_f = no.apply_filter(dens, rfilt)
    density_fp = (
        no.project(density_f, proj_level) if proj_level is not None else density_f
    )
    epsilon = no.simp(density_fp, eps_min, eps_max, p=1)

    kx, ky = 0, 0
    BD_RBME, sim = rbme_model(bands, epsilon, polarization, nh=100, Nmodel=2, N_RBME=8)
    BG = bk.min(bk.real(BD_RBME[:, ieig + 1] - BD_RBME[:, ieig]))
    center = (
        bk.min(bk.real(BD_RBME[:, ieig + 1])) + bk.max(bk.real(BD_RBME[:, ieig]))
    ) / 2
    # BG = BG/center
    if return_bg:
        return -BG, BD_RBME
    else:
        return -BG


#
# x0 = bk.ones(nvar) * 0.5
x0 = bk.array(np.random.rand(nvar))
x0 = bk.reshape(x0, lattice.discretization)
x0 = no.apply_filter(x0, rfilt * 2)
x0 = (x0 - x0.min()) / (x0.max() - x0.min())
x0 = symmetrize_pattern(x0).ravel()

#
# f = simu(x0)
# g = pt.grad(simu)
# df = g(x0)
#
# print(df)
#
# plt.figure()
# plt.imshow(df.real)
#
it = 0
fig, ax = plt.subplots(1)


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
    plt.pause(0.1)
    it += 1


opt = no.TopologyOptimizer(
    simu,
    x0,
    method="nlopt",
    maxiter=20,
    stopval=None,
    args=(1, rfilt),
    options={},
    callback=callback,
    verbose=True,
)

x_opt, f_opt = opt.minimize()

x_opt = bk.array(x_opt)

x_opt[x_opt < 0.5] = 0
x_opt[x_opt >= 0.5] = 1

obj, BD_RBME = simu(x_opt, proj_level=None, rfilt=0, return_bg=True)


def k_space_path_plot(Nb, K):
    bands_plot = np.zeros(3 * Nb - 2)
    bands_plot[:Nb] = K
    bands_plot[Nb : 2 * Nb - 1] = K[-1] + K[1:]
    bands_plot[2 * Nb - 1 : 3 * Nb - 2] = 2 * K[-1] + 2**0.5 * K[1:]
    return bands_plot


bands_plot = k_space_path_plot(Nb, K)


##############################################################################

BG = bk.min(bk.real(BD_RBME[:, ieig + 1] - BD_RBME[:, ieig]))
center = (bk.min(bk.real(BD_RBME[:, ieig + 1])) + bk.max(bk.real(BD_RBME[:, ieig]))) / 2

print("bandgap center", center.tolist())
print("bandgap width", BG.tolist())
print("relative width", (BG / center).tolist())


color = "#cf5268" if polarization == "TE" else "#4199b0"


plt.figure(figsize=(3.2, 2.5))
plotTE_RBME = plt.plot(bands_plot, BD_RBME, "-", c=color)
plt.ylim(0, 1.2)
plt.xlim(0, bands_plot[-1])
plt.xticks(
    [0, K[-1], 2 * K[-1], bands_plot[-1]], ["$\Gamma$", "$X$", "$M$", "$\Gamma$"]
)
plt.axvline(K[-1], c="k", lw=0.3)
plt.axvline(2 * K[-1], c="k", lw=0.3)
plt.ylabel(r"Frequency $\omega a/2\pi c$")
plt.legend([plotTE_RBME[0]], ["full", "2-point RBME"], loc=(0.31, 0.02))
plt.title(f"{polarization} modes", c=color)
plt.tight_layout()
y1 = bk.min(BD_RBME[:, ieig + 1].real)
y0 = bk.max(BD_RBME[:, ieig].real)
plt.fill_between(bands_plot, y1, y0, alpha=0.1, color=color, lw=0)
# bk.real(BD_RBME[:,ieig + 1] - BD_RBME[:,ieig])
