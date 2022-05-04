#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io

# Gram-Schmidt Orthogonization
# https://gist.github.com/iizukak/1287876/edad3c337844fac34f7e56ec09f9cb27d4907cc7

from . import backend as bk


def gram_schmidt(A, norm=True, row_vect=False):
    """Orthonormalizes vectors by gram-schmidt process

    Parameters
    -----------
    A : ndarray,
    Matrix having vectors in its columns

    norm : bool,
    Do you need Normalized vectors?

    row_vect: bool,
    Does Matrix A has vectors in its rows?

    Returns
    -------
    G : ndarray,
    Matrix of orthogonal vectors

    """
    if row_vect:
        # if true, transpose it to make column vector matrix
        A = A.T

    no_of_vectors = A.shape[1]
    G = A[:, 0:1]  # copy the first vector in matrix
    # 0:1 is done to to be consistent with dimensions - [[1,2,3]]

    # iterate from 2nd vector to number of vectors
    for i in range(1, no_of_vectors):

        # calculates weights(coefficents) for every vector in G
        GH = bk.conj(G)
        numerator = A[:, i].T @ GH
        denominator = bk.diag(G.T @ GH)  # to get elements in diagonal
        weights = bk.squeeze(numerator / denominator)

        # projected vector onto subspace G
        projected_vector = bk.sum(weights * G, axis=1, keepdims=True)

        # orthogonal vector to subspace G
        orthogonalized_vector = A[:, i : i + 1] - projected_vector

        # now add the orthogonal vector to our set
        G = bk.hstack((G, orthogonalized_vector))

    if norm:
        # to get orthoNORMAL vectors (unit orthogonal vectors)
        # replace zero to 1 to deal with division by 0 if matrix has 0 vector
        G = G / replace_zero(bk.linalg.norm(G, axis=0))

    if row_vect:
        return G.T

    return bk.array(G)


def replace_zero(array):

    for i in range(len(array)):
        if array[i] == 0:
            array[i] = 1
    return array
