#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


def run(backend, use_gpu):
    import protis as pt

    print(f"backend: {backend}, GPU: {use_gpu}")
    pt.set_backend(backend)
    pt.use_gpu(use_gpu)
    bk = pt.backend
    pi = pt.pi
    a = 1
    lattice = pt.Lattice(basis_vectors=[[a, 0], [0, a]], discretization=2**9)

    epsilon = lattice.ones() * 1

    circ = lattice.circle(center=(0.5, 0.5), radius=0.2)
    if backend == "jax":
        epsilon = epsilon.at[circ].set(8.9)
    else:
        epsilon[circ] = 8.9

    Gamma = (0, 0)
    X = (pi / a, 0)
    M = (pi / a, pi / a)
    sym_points = [Gamma, X, M, Gamma]

    Nb = 2
    kpath = pt.init_bands(sym_points, Nb)

    sim = pt.Simulation(lattice, epsilon=epsilon, nh=100)

    BD = {}
    for polarization in ["TE", "TM"]:
        ev_band = []
        for kx, ky in kpath:
            sim.k = kx, ky
            sim.solve(polarization, vectors=False)
            ev_norma = sim.eigenvalues * a / (2 * pi)
            ev_band.append(ev_norma)
        BD[polarization] = ev_band


def test_simulation():
    import protis as pt

    for backend in pt.available_backends:
        run(backend, False)
    if pt.HAS_CUDA:
        run("torch", True)
