#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


from nannos import is_scalar as _is_scalar
from nannos.layers import is_anisotropic

from .. import backend as bk


def is_scalar(a):
    if hasattr(a, "shape"):
        return len(a.shape) == 0
    else:
        return _is_scalar(a)


def block_anisotropic(a, dim=3):
    l = len(a[0]), len(a[1])
    if l != (dim, dim):
        raise ValueError(f"input shape must be ({dim},{dim},N,N)")
    return bk.stack(
        [bk.array(bk.stack([bk.array(a[j][i]) for i in range(3)])) for j in range(3)]
    )


def block_z_anisotropic(axx, axy, ayx, ayy, azz):
    zer = 0 * axx
    a_list = [
        [axx, axy, zer],
        [ayx, ayy, zer],
        [zer, zer, azz],
    ]
    return block_anisotropic(a_list)


def block(a):
    return bk.hstack(
        [
            bk.array(bk.vstack([bk.array(a[j][i]) for i in range(len(a))]))
            for j in range(len(a))
        ]
    )


def is_z_anisotropic(a):
    if not is_anisotropic(a):
        return False
    else:
        zer = 0 * a[0, 0]
        return (
            bk.allclose(a[0, 2], zer)
            and bk.allclose(a[1, 2], zer)
            and bk.allclose(a[2, 0], zer)
            and bk.allclose(a[2, 1], zer)
        )


def is_symmetric(M):
    return bk.allclose(M, M.T)


def is_hermitian(M):
    return bk.allclose(M, bk.conj(M).T)
