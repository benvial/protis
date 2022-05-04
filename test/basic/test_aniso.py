#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import numpy as np

import protis as pt
from protis.simulation import *

bk = pt.backend


def test_anisotropy():
    lat = pt.Lattice(((1, 0), (0, 1)), 2**9)
    muxx = 3 * lat.ones()
    epsilonxx = lat.ones()
    cir = lat.circle((0.5, 0.5), 0.2)
    if pt.get_backend() == "jax":
        epsilonxx = epsilonxx.at[cir].set(8.9)  # jax syntax
    else:
        epsilonxx[cir] = 8.9
    epsilon_list = [
        [epsilonxx, epsilonxx * 0, 0 * epsilonxx],
        [epsilonxx * 0, epsilonxx, 0 * epsilonxx],
        [epsilonxx * 0, epsilonxx * 0, epsilonxx],
    ]
    epsilon = block_anisotropic(epsilon_list)
    epsilonz = block_z_anisotropic(
        epsilonxx, 0 * epsilonxx, 0 * epsilonxx, epsilonxx, epsilonxx
    )
    assert np.allclose(epsilon, epsilonz)
    muz = block_z_anisotropic(muxx, 0 * muxx, 0 * muxx, muxx, muxx)

    epsilon_list = [
        [epsilonxx, epsilonxx * 0, 0 * epsilonxx],
        [epsilonxx * 0, epsilonxx, 0 * epsilonxx],
        [epsilonxx * 0, epsilonxx * 3, epsilonxx],
    ]
    epsilon1 = block_anisotropic(epsilon_list)
    assert is_z_anisotropic(epsilon1) == False

    for pola in ["TE", "TM"]:
        print(pola)
        simh = pt.Simulation(lat, epsilon=epsilonxx, mu=3)
        wh = simh.solve(pola, vectors=False)
        simh2 = pt.Simulation(lat, epsilon=epsilonz, mu=3 * bk.eye(3))
        wh2 = simh2.solve(pola, vectors=False)
        simh3 = pt.Simulation(lat, epsilon=epsilonxx, mu=3)
        wh3 = simh3.solve(pola, vectors=False)
        simh4 = pt.Simulation(lat, epsilon=epsilonz, mu=muxx)
        wh4 = simh4.solve(pola, vectors=False)
        simh5 = pt.Simulation(lat, epsilon=epsilonz, mu=muz)
        wh5 = simh4.solve(pola, vectors=False)
        sim = pt.Simulation(lat, epsilon=epsilonxx, mu=3)
        w = sim.solve(pola, vectors=False)
        assert np.allclose(wh, w)
        assert np.allclose(wh2, w)
        assert np.allclose(wh3, w)
        assert np.allclose(wh4, w)
        assert np.allclose(wh5, w)
