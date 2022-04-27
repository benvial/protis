#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import scipy
from nannos.formulations import fft
from nannos.utils import is_scalar

from . import backend as bk
from . import get_backend
from .reduced import gram_schmidt


def is_symmetric(M):
    return bk.allclose(M, M.T)


def is_hermitian(M):
    return bk.allclose(M, bk.conj(M).T)


def eig(M, vectors=True, hermitian=False):
    if vectors:
        _eig = bk.linalg.eigh if hermitian else bk.linalg.eig
    else:
        _eig = bk.linalg.eigvalsh if hermitian else bk.linalg.eigvals
    return _eig(M)


def gen_eig(A, B, vectors=True):
    if get_backend() == "scipy":
        return _gen_eig_scipy(A, B, vectors=vectors)
    A = bk.array(A + 0j, dtype=bk.complex128)
    if is_scalar(B):
        C = A / B
        return eig(C, vectors=vectors, hermitian=is_hermitian(C))
    else:
        invB = bk.linalg.inv(B)
        C = invB @ A
        return eig(C, vectors=vectors, hermitian=is_hermitian(C))


def _gen_eig_scipy(A, B, vectors=True):
    A = bk.array(A + 0j, dtype=bk.complex128)
    if is_scalar(B):
        if vectors:
            try:
                out = scipy.linalg.eigh(A / B)
            except scipy.linalg.LinAlgError:
                out = scipy.linalg.eig(A / B)
        else:
            try:
                out = scipy.linalg.eigvalsh(A / B)
            except scipy.linalg.LinAlgError:
                out = scipy.linalg.eigvals(A / B)
    else:
        if vectors:
            try:
                out = scipy.linalg.eigh(A, B)
            except scipy.linalg.LinAlgError:
                out = scipy.linalg.eig(A, B)
        else:
            try:
                out = scipy.linalg.eigvalsh(A, B)
            except scipy.linalg.LinAlgError:
                out = scipy.linalg.eigvals(A, B)

    return out


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

        self.build_epsilon_hat()
        self.build_mu_hat()

    @property
    def kx(self):
        r = self.lattice.reciprocal
        return (
            self.k[0] + r[0, 0] * self.harmonics[0, :] + r[0, 1] * self.harmonics[1, :]
        )

    @property
    def ky(self):
        r = self.lattice.reciprocal
        return (
            self.k[1] + r[0, 1] * self.harmonics[0, :] + r[1, 1] * self.harmonics[1, :]
        )

    @property
    def Kx(self):
        return bk.diag(self.kx)

    @property
    def Ky(self):
        return bk.diag(self.ky)

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

    def build_A(self, polarization):
        def matmuldiag(A, B):
            return bk.einsum("i,ik->ik", bk.array(bk.diag(A)), bk.array(B))

        q = self.mu_hat if polarization == "TM" else self.epsilon_hat
        if is_scalar(q):
            A = 1 / q * (self.Kx @ self.Kx + self.Ky @ self.Ky)
        else:
            u = bk.linalg.inv(q)
            # self.A = self.Kx @ u @ self.Kx + self.Ky @ u @ self.Ky
            kxu = matmuldiag(self.Kx, u)
            kyu = matmuldiag(self.Ky, u)
            A = matmuldiag(self.Kx.T, kxu.T).T + matmuldiag(self.Ky.T, kyu.T).T

        self.A = bk.array(A + 0j, dtype=bk.complex128)

    def build_B(self, polarization):
        self.B = self.epsilon_hat if polarization == "TM" else self.mu_hat

    def solve(self, polarization, vectors=True, rbme=None):
        self.build_A(polarization)
        self.build_B(polarization)
        if rbme is None:
            A = self.A
            B = self.B
        else:
            A = bk.conj(rbme.T) @ self.A @ rbme
            B = self.B if is_scalar(self.B) else bk.conj(rbme.T) @ self.B @ rbme
        eign = gen_eig(A, B, vectors=vectors)
        w = eign[0] if vectors else eign
        v = eign[1] if vectors else None
        lmin = bk.min(
            bk.linalg.norm(
                bk.array(self.lattice.basis_vectors, dtype=bk.float64), axis=0
            )
        )
        kmin = 2 * bk.pi / lmin
        eps = 1e-12 * kmin**2
        k0 = (w + eps) ** 0.5  # this is to avoid nan gradients
        i = bk.argsort(bk.real(k0))
        self.eigenvalues = k0[i]
        self.eigenvectors = v[:, i] if vectors else None
        if vectors:
            return self.eigenvalues, self.eigenvectors
        else:
            return self.eigenvalues

    def get_rbme_matrix(self, N_RBME, bands_RBME, polarization):
        v = []
        for kx, ky in bands_RBME:
            self.k = kx, ky
            self.solve(polarization, vectors=True)
            v.append(self.eigenvectors[:, :N_RBME])
        U = bk.hstack(v)
        return gram_schmidt(U)
