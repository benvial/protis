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
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**7)


eps_min, eps_max = 1, 8.9

ieig = 4


def simu(x, proj_level=None, rfilt=0):
    dens = bk.reshape(x, lattice.discretization)
    density_f = no.apply_filter(dens, rfilt)
    density_fp = (
        no.project(density_f, proj_level) if proj_level is not None else density_f
    )
    epsilon = no.simp(density_fp, eps_min, eps_max, p=1)
    polarization = "TM"
    kx, ky = 0, 0
    sim = pt.Simulation(lattice, (kx, ky), epsilon=epsilon, mu=1, nh=100)
    a = sim.lattice.basis_vectors[0][0]

    k0, w = sim.solve(polarization, vectors=True)
    ev_norma = bk.real(k0[ieig + 1] - k0[ieig])
    return -ev_norma


nvar = lattice.discretization[0] * lattice.discretization[1]

x0 = lattice.ones() * 1
hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
x0[hole] = 0


x0 = x0.real.ravel()

x0 = bk.ones(nvar) * 0.5
x0 = bk.array(np.random.rand(nvar))

f = simu(x0)
g = pt.grad(simu)
df = g(x0)
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
    density_f = no.apply_filter(dens, rfilt)
    density_fp = no.project(density_f, proj_level)
    # plt.figure()
    ax.clear()
    plt.sca(ax)
    plt.imshow(density_fp, cmap="Blues")
    ax.axis("off")
    plt.suptitle(f"iteration {it}, objective = {y:.5f}")
    # plt.tight_layout()
    plt.pause(0.1)
    it += 1


rfilt = lattice.discretization[0] / 30
opt = no.TopologyOptimizer(
    simu,
    x0,
    method="nlopt",
    maxiter=12,
    stopval=None,
    args=(1, rfilt),
    options={},
    callback=callback,
    verbose=True,
)

x_opt, f_opt = opt.minimize()
