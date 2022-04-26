#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


from nannos.formulations import fft
from nannos.utils import is_scalar

from . import backend as bk


def is_symmetric(M):
    return bk.allclose(M, M.T)


def is_hermitian(M):
    return bk.allclose(M, bk.conj(M).T)


def eig(M):
    _eig = bk.linalg.eigh if is_hermitian(M) else bk.linalg.eig
    return _eig(M)


def gen_eig(A, B):

    A = bk.array(A + 0j, dtype=bk.complex128)
    if is_scalar(B):
        return eig(A / B)
    else:
        Q = bk.linalg.inv(B)
        return eig(Q @ A)


class Simulation:
    def __init__(self, lattice, k=(0, 0), epsilon=1, mu=1, nh=100):
        self.lattice = lattice
        self.k = k
        self.epsilon = epsilon
        self.mu = mu
        self.nh0 = int(nh)
        nh0 = int(nh)
        if nh0 == 1:
            nh0 = 2
        # Get the harmonics
        self.harmonics, self.nh = self.lattice.get_harmonics(nh0)
        # Check if nh and resolution satisfy Nyquist criteria
        maxN = bk.max(self.harmonics)
        if self.lattice.discretization[0] <= 2 * maxN or (
            self.lattice.discretization[1] <= 2 * maxN and not self.lattice.is_1D
        ):
            raise ValueError(f"lattice discretization must be > {2*maxN}")
        # Buid lattice vectors
        self.k = k
        r = self.lattice.reciprocal
        self.kx = (
            self.k[0] + r[0, 0] * self.harmonics[0, :] + r[0, 1] * self.harmonics[1, :]
        )
        self.ky = (
            self.k[1] + r[0, 1] * self.harmonics[0, :] + r[1, 1] * self.harmonics[1, :]
        )
        self.Kx = bk.diag(self.kx)
        self.Ky = bk.diag(self.ky)

    def _get_toeplitz_matrix(self, u, transverse=False):
        if transverse:
            return [
                [self._get_toeplitz_matrix(u[i, j]) for j in range(2)] for i in range(2)
            ]
        else:
            uft = fft.fourier_transform(u)
            ix = bk.arange(self.nh)
            jx, jy = bk.meshgrid(ix, ix, indexing="ij")
            delta = self.harmonics[:, jx] - self.harmonics[:, jy]
            return uft[delta[0, :], delta[1, :]]

    def build_hat(self, u):
        return u if is_scalar(u) else self._get_toeplitz_matrix(u)

    def build_epsilon_hat(self):
        self.epsilon_hat = self.build_hat(self.epsilon)

    def build_mu_hat(self):
        self.mu_hat = self.build_hat(self.mu)

    def build_A(self, pola):
        def matmuldiag(A, B):
            return bk.einsum("i,ik->ik", bk.array(bk.diag(A)), bk.array(B))

        q = self.mu_hat if pola == "TM" else self.epsilon_hat
        if is_scalar(q):
            self.A = 1 / q * self.Kx @ self.Kx + self.Ky @ self.Ky
        else:
            u = bk.linalg.inv(q)
            kxu = matmuldiag(self.Kx, u)
            kyu = matmuldiag(self.Ky, u)
            self.A = matmuldiag(self.Kx.T, kxu.T).T + matmuldiag(self.Ky.T, kyu.T).T

    def build_B(self, pola):
        self.B = self.epsilon_hat if pola == "TM" else self.mu_hat

    def solve(self, pola):
        self.build_epsilon_hat()
        self.build_mu_hat()
        self.build_A(pola)
        self.build_B(pola)
        w, v = gen_eig(self.A, self.B)
        k0 = (1e-16 + w) ** 0.5  # this is to avoid nan gradients
        i = bk.argsort(bk.real(k0))
        self.eigenvalues = k0[i]
        self.eigenvectors = v[i]

        return self.eigenvalues, self.eigenvectors
