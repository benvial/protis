#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import matplotlib.pyplot as plt
import numpy as np

import protis as pt

bk = pt.backend
plt.close("all")
plt.ion()
a = 1
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
epsilon = lattice.ones() * 1
hole = lattice.square((0.5, 0.5), 0.33)
epsilon[hole] = 11.4
mu = 1
nh = 100
sim = pt.Simulation(lattice, epsilon=epsilon, mu=mu, nh=nh)
polarization = "TE"


def model(bands, nh=100):
    ev_band = []
    for kx, ky in bands:
        sim.k = kx, ky
        sim.solve(polarization, vectors=False)
        ev_norma = sim.eigenvalues * a / (2 * np.pi)
        ev_band.append(ev_norma)
    return ev_band


Nbz = 51
bandsx = bk.linspace(-pt.pi / a, pt.pi / a, Nbz)
bandsy = bk.linspace(-pt.pi / a, pt.pi / a, Nbz)
bandsx1, bandsy1 = bk.meshgrid(bandsx, bandsy, indexing="ij")
bands = bk.vstack([bandsx1.ravel(), bandsy1.ravel()]).T

BD = model(bands)
BD = bk.array(BD)

BD = BD.reshape(Nbz, Nbz, sim.nh)

ieig = 1
ev_target = 0.59

from protis.isocontour import get_isocontour

isocontour = get_isocontour(bandsx, bandsy, BD[:, :, ieig], ev_target, method="protis")
isocontour1 = get_isocontour(
    bandsx, bandsy, BD[:, :, ieig], ev_target, method="skimage"
)


plt.figure()
plt.pcolormesh(bandsx, bandsy, BD[:, :, ieig])
plt.axis("scaled")
plt.colorbar()
plt.plot(isocontour[:, :, 0], isocontour[:, :, 1], ".k")

for contour in isocontour1:
    plt.plot(contour[:, 1], contour[:, 0], "-r")
plt.tight_layout()
