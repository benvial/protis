#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import protis as pt


def test_simu():
    bk = pt.backend
    a = 1
    lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
    epsilon = lattice.ones() * 1
    hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
    if pt.get_backend() == "jax":
        epsilon = epsilon.at[hole].set(8.9)  # jax syntax
    else:
        epsilon[hole] = 8.9
    nh = 100
    mu = 1
    bands = [(0, 0), (pt.pi / a, 0), (pt.pi / a, 0)]
    for polarization in ["TE", "TM"]:
        for kx, ky in bands:
            print(kx, ky)
            sim = pt.Simulation(lattice, (kx, ky), epsilon=epsilon, mu=mu, nh=nh)
            a = sim.lattice.basis_vectors[0][0]

            neig = 6
            k0, v = sim.solve(polarization)
            ev_norma = k0[:neig] * a / (2 * pt.pi)
