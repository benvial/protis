#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import numpy as np

import protis as pt


def test_simu():
    pt.set_backend("scipy")
    bk = pt.backend
    a = 1
    lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
    epsilon = lattice.ones() * 1
    hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
    epsilon[hole] = 8.9
    nh = 800
    mu = 1
    bands = [(0, 0), (pt.pi / a, 0), (pt.pi / a, pt.pi / a)]
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
                sigma=1e-12,
            )
            t2 = pt.toc(t0)
            print("seedup = ", t1 / t2)

            assert np.allclose(k0[:neig], k0_sparse)


#
# np.random.seed(10)
#
# # def test_simu():
# if True:
#     bk = pt.backend
#     a = 1
#     lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
#     epsilon = lattice.ones() * 1
#     hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
#     epsilon[hole] = 8.9
#     nh = 1600
#     mu = 1
#     bands = [(0, 0), (0.01 * pt.pi / a, 0)]
#     for polarization in ["TE", "TM"]:
#         print(f"{polarization} polarization")
#         print("-----------------------------")
#         j = 0
#         for kx, ky in bands:
#             print(kx, ky)
#             sim = pt.Simulation(lattice, (kx, ky), epsilon=epsilon, mu=mu, nh=nh)
#             neig = 1
#             t0 = pt.tic()
#             k0, v = sim.solve(
#                 polarization,
#                 sparse=True,
#                 neig=neig,
#                 sigma=1e-12,
#             )
#             t1 = pt.toc(t0)
#             t0 = pt.tic()
#             v0 = np.random.rand(sim.nh) if j == 0 else v_sparse[:, 0]
#             k0_sparse, v_sparse = sim.solve(
#                 polarization,
#                 sparse=True,
#                 neig=neig,
#                 sigma=1e-12,
#                 v0=v0,
#             )
#             t2 = pt.toc(t0)
#             print("seedup = ", t1 / t2)
#
#             assert np.allclose(k0[:neig], k0_sparse)
#             j += 1


# if True:
# # def test_simu():
#     pt.set_backend("torch")
#     bk = pt.backend
#     a = 1
#     lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
#     epsilon = lattice.ones() * 1
#     hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
#     epsilon[hole] = 8.9
#     nh = 800
#     mu = 1
#     bands = [(0, 0), (pt.pi / a, 0), (pt.pi / a, pt.pi / a)]
#     for polarization in ["TE", "TM"]:
#         for kx, ky in bands:
#             print(kx, ky)
#             sim = pt.Simulation(lattice, (kx, ky), epsilon=epsilon, mu=mu, nh=nh)
#             neig = 10
#             print("dense")
#             t0 = pt.tic()
#             k0, v = sim.solve(polarization)
#             t1 = pt.toc(t0)
#             print("sparse")
#             t0 = pt.tic()
#             k0_sparse, v_sparse = sim.solve(
#                 polarization,
#                 sparse=True,
#                 neig=neig,
#                 largest=False,
#             )
#             t2 = pt.toc(t0)
#             print("seedup = ", t1 / t2)
#
#             assert np.allclose(k0[:neig], k0_sparse)
#
# pt.set_backend("torch")
# bk = pt.backend
# a = 1
# lattice = pt.Lattice([[a, 0], [0, a]], discretization=2**9)
# epsilon = lattice.ones() * 1
# hole = lattice.circle(center=(0.5, 0.5), radius=0.2)
# epsilon[hole] = 8.9
# nh = 800
# mu = 1
# sim = pt.Simulation(lattice, (0, 0), epsilon=epsilon, mu=mu, nh=nh)
# neig = 10
# print("dense")
# polarization = "TE"
#
# k0_sparse, v_sparse = sim.solve(
#     polarization,
#     sparse=True,
#     neig=neig,
#     largest=False,
# )
#
# A = sim.build_A(polarization)
# B = sim.build_B(polarization)
# import torch
# B1 = bk.array(bk.eye(A.shape[0])+A*0j,dtype=bk.complex128)
# X = bk.array(bk.eye(A.shape[0])+A*0j,dtype=bk.complex128)[:,:neig]
# # X = bk.array(np.random.rand(A.shape[0],neig)+1j*np.random.rand(A.shape[0],neig),dtype=bk.complex128)
# torch.lobpcg(A , k=neig,B=B1,X=X)
# torch.matmul(A, B1)
