#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


from . import backend as bk
from . import get_block
from .eig import gen_eig
from .fft import *
from .plot import *
from .reduced import gram_schmidt
from .utils import *


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

    def _get_toeplitz_matrix(self, u):
        if is_scalar(u):
            return u
        # else:
        # u *= bk.ones((self.nh,self.nh),dtype=bk.complex128)
        if is_anisotropic(u):
            utf = [
                self._get_toeplitz_matrix(u[i, j]) for i in range(2) for j in range(2)
            ]
            utf.append(self._get_toeplitz_matrix(u[2, 2]))
            return block_z_anisotropic(*utf)

        uft = fourier_transform(u)
        ix = bk.arange(self.nh)
        jx, jy = bk.meshgrid(ix, ix, indexing="ij")
        delta = self.harmonics[:, jx] - self.harmonics[:, jy]
        return uft[delta[0, :], delta[1, :]]

    def build_epsilon_hat(self):
        self.epsilon_hat = self._get_toeplitz_matrix(self.epsilon)
        return self.epsilon_hat

    def build_mu_hat(self):
        self.mu_hat = self._get_toeplitz_matrix(self.mu)
        return self.mu_hat

    def build_A(self, polarization):
        def matmuldiag(A, B):
            return bk.einsum("i,ik->ik", bk.array(bk.diag(A)), bk.array(B))

        q = self.mu_hat if polarization == "TM" else self.epsilon_hat
        if is_scalar(q):
            A = 1 / q * (self.Kx @ self.Kx + self.Ky @ self.Ky)
        else:
            if is_anisotropic(q):
                if q.shape == (3, 3):
                    u = bk.linalg.inv(q)
                    A = u[1, 1] * self.Kx @ self.Kx + u[0, 0] * self.Ky @ self.Ky
                    A -= u[1, 0] * self.Kx @ self.Ky + u[0, 1] * self.Ky @ self.Kx
                else:
                    q = block(q[:2, :2])
                    qxx = get_block(q, 0, 0, self.nh)
                    qxy = get_block(q, 0, 1, self.nh)
                    qyx = get_block(q, 1, 0, self.nh)
                    qyy = get_block(q, 1, 1, self.nh)

                    u = bk.linalg.inv(q)
                    uxx = get_block(u, 0, 0, self.nh)
                    uxy = get_block(u, 0, 1, self.nh)
                    uyx = get_block(u, 1, 0, self.nh)
                    uyy = get_block(u, 1, 1, self.nh)

                    kyuxx = matmuldiag(self.Ky, uxx)
                    kxuyy = matmuldiag(self.Kx, uyy)
                    kxuyx = matmuldiag(self.Kx, uyx)
                    kyuxy = matmuldiag(self.Ky, uxy)
                    A = (
                        matmuldiag(self.Kx.T, kxuyy.T).T
                        + matmuldiag(self.Ky.T, kyuxx.T).T
                    )
                    A -= (
                        matmuldiag(self.Kx.T, kyuxy.T).T
                        + matmuldiag(self.Ky.T, kxuyx.T).T
                    )
            else:
                u = bk.linalg.inv(q)
                # self.A = self.Kx @ u @ self.Kx + self.Ky @ u @ self.Ky
                kxu = matmuldiag(self.Kx, u)
                kyu = matmuldiag(self.Ky, u)
                A = matmuldiag(self.Kx.T, kxu.T).T + matmuldiag(self.Ky.T, kyu.T).T

        self.A = bk.array(A + 0j, dtype=bk.complex128)

        return self.A

    def build_B(self, polarization):
        if polarization == "TM":
            if is_scalar(self.epsilon_hat):
                self.B = self.epsilon_hat
            else:
                self.B = (
                    self.epsilon_hat[2, 2]
                    if is_anisotropic(self.epsilon_hat)
                    else self.epsilon_hat
                )

        else:
            if is_scalar(self.mu_hat):
                self.B = self.mu_hat
            else:
                self.B = (
                    self.mu_hat[2, 2] if is_anisotropic(self.mu_hat) else self.mu_hat
                )

        return self.B

    def solve(
        self,
        polarization,
        vectors=True,
        rbme=None,
        return_square=False,
        sparse=False,
        neig=10,
        ktol=1e-12,
        **kwargs,
    ):
        self.build_A(polarization)
        self.build_B(polarization)
        if rbme is None:
            A = self.A
            B = self.B
        else:
            # reduced bloch mode expansion
            A = bk.conj(rbme.T) @ self.A @ rbme
            B = self.B if is_scalar(self.B) else bk.conj(rbme.T) @ self.B @ rbme
        eign = gen_eig(A, B, vectors=vectors, sparse=sparse, neig=neig, **kwargs)
        w = eign[0] if vectors else eign
        v = eign[1] if vectors else None
        lmin = bk.min(
            bk.linalg.norm(
                bk.array(self.lattice.basis_vectors, dtype=bk.float64), axis=0
            )
        )
        kmin = 2 * bk.pi / lmin
        eps = ktol * kmin**2
        k0 = (w + eps) ** 0.5  # this is to avoid nan gradients

        if return_square:
            i = bk.argsort(bk.real(w))
            self.eigenvalues = w[i]
        else:
            i = bk.argsort(bk.real(k0))
            self.eigenvalues = k0[i]
        if rbme is not None and vectors:
            # retrieve full vectors
            v = rbme @ v @ bk.conj(rbme.T)
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

    def get_mode(self, imode):
        v = self.eigenvectors
        V = bk.zeros(self.lattice.discretization, dtype=bk.complex128)
        V[self.harmonics[0], self.harmonics[1]] = v[:, imode]
        mode = inverse_fourier_transform(V)
        # mode /= bk.max(bk.abs(mode))
        return mode

    def plot(
        self,
        field,
        nper=1,
        ax=None,
        cmap="RdBu_r",
        show_cell=False,
        cellstyle="-k",
        **kwargs,
    ):
        return plot2d(
            self.lattice,
            self.lattice.grid,
            field,
            nper,
            ax,
            cmap,
            show_cell,
            cellstyle,
            **kwargs,
        )
