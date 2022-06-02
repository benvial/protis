#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import scipy

from . import backend as bk
from . import get_backend
from .utils import *


def eig(M, vectors=True, hermitian=False):
    if vectors:
        _eig = bk.linalg.eigh if hermitian else bk.linalg.eig
    else:
        _eig = bk.linalg.eigvalsh if hermitian else bk.linalg.eigvals
    return _eig(M)


def _gen_eig_scipy(A, B, vectors=True):
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


def _gen_eig_scipy_sparse(A, B, vectors=True, neig=10, **kwargs):
    if is_scalar(B):
        out = scipy.sparse.linalg.eigsh(
            A / B, k=neig, return_eigenvectors=vectors, **kwargs
        )
    else:
        out = scipy.sparse.linalg.eigsh(
            A, k=neig, M=B, return_eigenvectors=vectors, **kwargs
        )
    return out


def _gen_eig_torch_sparse(A, B, vectors=True, neig=10, **kwargs):
    if is_scalar(B):
        C = A / B
        print(C.dtype)
        print(type(C))
        B1 = bk.array(bk.eye(A.shape[0]) + 0j, dtype=bk.complex128)
        out = bk.lobpcg(A / B, k=neig, B=B1, **kwargs)
    else:
        out = bk.lobpcg(A, k=neig, B=B, **kwargs)
    return out if vectors else out[0]


def gen_eig(A, B, vectors=True, sparse=False, neig=10, **kwargs):
    A = bk.array(A + 0j, dtype=bk.complex128)
    B = bk.array(B + 0j, dtype=bk.complex128)
    _backend = get_backend()
    if sparse and _backend != "scipy":
        raise NotImplementedError(
            "sparse eigenproblems only implemented for scipy backend"
        )
    if _backend == "scipy":
        if sparse:
            return _gen_eig_scipy_sparse(A, B, vectors=vectors, neig=neig, **kwargs)
        else:
            return _gen_eig_scipy(A, B, vectors=vectors)
    # elif _backend == "torch" and sparse:
    #     # this works only for real matrices
    #     return _gen_eig_torch_sparse(A, B, vectors=vectors, neig=neig, **kwargs)

    if is_scalar(B):
        C = A / B
        return eig(C, vectors=vectors, hermitian=is_hermitian(C))
    else:
        # invB = bk.linalg.inv(B)
        # C = invB @ A
        C = bk.linalg.solve(B, A)
        return eig(C, vectors=vectors, hermitian=is_hermitian(C))
