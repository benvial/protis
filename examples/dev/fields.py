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

a = 1
lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**8)
epsilon = lattice.ones() * 1
hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
epsilon[hole] = 8.9

fig, ax = plt.subplots(3, 2, figsize=(2, 3))

q = pt.pi / a
for i, k in enumerate([(0, 0), (q, 0), (q, q)]):
    sim = pt.Simulation(lattice, k=k, epsilon=epsilon, mu=1, nh=100)
    sim.solve("TM", vectors=True)
    for imode in range(2):
        mode = sim.get_mode(imode)
        # phasor = bk.exp(-1j*(sim.k[0]*a+ sim.k[1]*a)/2)
        D = epsilon * mode
        ims = sim.plot(
            D.real,
            nper=(3, 3),
            ax=ax[i, imode],
        )
        sim.plot(
            sim.epsilon.real,
            nper=(3, 3),
            ax=ax[i, imode],
            cmap="Greys",
            alpha=0.1,
        )
        ax[i, imode].set_axis_off()
plt.tight_layout()


fig, ax = plt.subplots(1, 2, figsize=(2, 1))

q = pt.pi / a
sim = pt.Simulation(lattice, k=(q, 0), epsilon=epsilon, mu=1, nh=100)
sim.solve("TE", vectors=True)
for imode in range(2):
    mode = sim.get_mode(imode)
    phasor = bk.exp(-1j * (sim.k[0] * a / 2 + sim.k[1] * a / 2))
    # x,y = lattice.grid
    # phasor =bk.exp(-1j*(sim.k[0]*x+ sim.k[1]*y))

    ims = sim.plot(
        mode.real,
        nper=(3, 3),
        ax=ax[imode],
    )
    sim.plot(
        sim.epsilon.real,
        nper=(3, 3),
        ax=ax[imode],
        cmap="Greys",
        alpha=0.1,
    )
    ax[imode].set_axis_off()
plt.tight_layout()
