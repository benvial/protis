#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import time

import matplotlib.pyplot as plt

import protis as pt

# pt.set_backend("scipy")
pt.set_backend("numpy")
bk = pt.backend
plt.ion()
plt.close("all")

a = 1
lattice = pt.Lattice(
    [[a, 0], [0, a]], discretization=2**9, truncation="parallelogrammic"
)
rod = lattice.circle(center=(0.5, 0.5), radius=0.11)
epsilon = lattice.ones()
epsilon[rod] = 15
mu = lattice.ones()
mu[rod] = 3
polarization = "TM"

nh = 11
sim = pt.Simulation(lattice, epsilon=epsilon, mu=mu, nh=nh)
sim.k = (0.2, 0.3)
A = sim.build_A(polarization)
print(sim.nh, A.shape)

plt.figure()
plt.imshow(A.real)
plt.colorbar()


# lattice1 = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
