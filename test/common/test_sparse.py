#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import numpy as np


def test_simu():
    import protis as pt

    pt.set_backend("scipy")
    bk = pt.backend
    a = 1
    lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
    epsilon = lattice.ones() * 1
    hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
    epsilon[hole] = 8.9
    nh = 300
    mu = 1
    bands = [(0.1, 0.1), (pt.pi / a, 0), (pt.pi / a, pt.pi / a)]
    for polarization in ["TE", "TM"]:
        for kx, ky in bands:
            print(kx, ky)
            sim = pt.Simulation(lattice, (kx, ky), epsilon=epsilon, mu=mu, nh=nh)
            neig = 10
            print("dense")
            t0 = pt.tic()
            k0, v = sim.solve(polarization)
            t1 = pt.toc(t0)
            print("sparse")
            t0 = pt.tic()
            k0_sparse, v_sparse = sim.solve(
                polarization,
                sparse=True,
                neig=neig,
            )
            t2 = pt.toc(t0)
            print("seedup = ", t1 / t2)
            k0_sparse = np.sort(k0_sparse.real)
            print(k0[:neig])
            print(k0_sparse)

            assert np.allclose(k0[:neig], k0_sparse, atol=1e-2)
