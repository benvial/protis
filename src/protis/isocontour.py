#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

__all__ = ["get_isocontour"]

from skimage import measure

from . import backend as bk


def finterp(A, B, Adata, Bdata, target):
    return A + (B - A) * (target - Adata) / (Bdata - Adata)


class Square:
    A = [0, 0]
    B = [0, 0]
    C = [0, 0]
    D = [0, 0]
    A_data = 0.0
    B_data = 0.0
    C_data = 0.0
    D_data = 0.0

    def GetCaseId(self, threshold):
        caseId = 0
        if self.A_data >= threshold:
            caseId |= 1
        if self.B_data >= threshold:
            caseId |= 2
        if self.C_data >= threshold:
            caseId |= 4
        if self.D_data >= threshold:
            caseId |= 8

        return caseId

    def GetLines(self, Threshold, interp=True, target=1):
        lines = []
        caseId = self.GetCaseId(Threshold)

        if caseId in (0, 15):
            return []

        if caseId in (1, 14, 10):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.B[1]
            qX = self.D[0]
            qY = (self.A[1] + self.D[1]) / 2
            if interp:
                pX = finterp(self.A[0], self.B[0], self.A_data, self.B_data, target)
                qY = finterp(self.A[1], self.D[1], self.A_data, self.D_data, target)

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (2, 13, 5):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.A[1]
            qX = self.C[0]
            qY = (self.A[1] + self.D[1]) / 2
            if interp:
                pX = finterp(self.A[0], self.B[0], self.A_data, self.B_data, target)
                qY = finterp(self.A[1], self.D[1], self.A_data, self.D_data, target)

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (3, 12):
            pX = self.A[0]
            pY = (self.A[1] + self.D[1]) / 2
            qX = self.C[0]
            qY = (self.B[1] + self.C[1]) / 2
            if interp:
                pY = finterp(self.A[1], self.D[1], self.A_data, self.D_data, target)
                qY = finterp(self.B[1], self.C[1], self.B_data, self.C_data, target)

            line = (pX, pY, qX, qY)

            lines.append(line)

        if caseId in (4, 11, 10):
            pX = (self.C[0] + self.D[0]) / 2
            pY = self.D[1]
            qX = self.B[0]
            qY = (self.B[1] + self.C[1]) / 2
            if interp:
                pX = finterp(self.C[0], self.D[0], self.C_data, self.D_data, target)
                qY = finterp(self.B[1], self.C[1], self.B_data, self.C_data, target)

            line = (pX, pY, qX, qY)

            lines.append(line)

        elif caseId in (6, 9):
            pX = (self.A[0] + self.B[0]) / 2
            pY = self.A[1]
            qX = (self.C[0] + self.D[0]) / 2
            qY = self.C[1]
            if interp:
                pX = finterp(self.A[0], self.B[0], self.A_data, self.B_data, target)
                qX = finterp(self.C[0], self.D[0], self.C_data, self.D_data, target)

            line = (pX, pY, qX, qY)

            lines.append(line)

        elif caseId in (7, 8, 5):
            pX = (self.C[0] + self.D[0]) / 2
            pY = self.C[1]
            qX = self.A[0]
            qY = (self.A[1] + self.D[1]) / 2
            if interp:
                pX = finterp(self.C[0], self.D[0], self.C_data, self.D_data, target)
                qy = finterp(self.A[1], self.D[1], self.A_data, self.D_data, target)

            line = (pX, pY, qX, qY)

            lines.append(line)

        return lines


def marching_square(xVector, yVector, Data, threshold, interp=True):
    linesList = []

    Height = len(Data)  # rows
    Width = len(Data[1])  # cols

    if (Width == len(xVector)) and (Height == len(yVector)):
        squares = bk.full((Height - 1, Width - 1), Square())

        sqHeight = squares.shape[0]  # rows count
        sqWidth = squares.shape[1]  # cols count

        for j in range(sqHeight):  # rows
            for i in range(sqWidth):  # cols
                a = Data[j + 1, i]
                b = Data[j + 1, i + 1]
                c = Data[j, i + 1]
                d = Data[j, i]
                A = [xVector[i], yVector[j + 1]]
                B = [xVector[i + 1], yVector[j + 1]]
                C = [xVector[i + 1], yVector[j]]
                D = [xVector[i], yVector[j]]

                squares[j, i].A_data = a
                squares[j, i].B_data = b
                squares[j, i].C_data = c
                squares[j, i].D_data = d

                squares[j, i].A = A
                squares[j, i].B = B
                squares[j, i].C = C
                squares[j, i].D = D

                list = squares[j, i].GetLines(
                    threshold, interp=interp, target=threshold
                )

                linesList = linesList + list
    else:
        raise AssertionError

    return [linesList]


def _get_iso(x, y, im, level):
    iso = marching_square(x, y, im, level, interp=True)
    if iso == [[]]:
        iso = [[[]]]
    return bk.array(iso)[:, :, 0:2]


def _get_iso_skimage(x, y, im, level):
    contours = measure.find_contours(im, level)
    contours = bk.array(contours)
    if len(contours) == 0:
        return contours
    contours[:, :, 0] = (x.max() - x.min()) * contours[:, :, 0] / len(x) + x.min()
    contours[:, :, 1] = (y.max() - y.min()) * contours[:, :, 1] / len(y) + y.min()
    return contours


def get_isocontour(x, y, im, level, method="skimage"):
    if method not in ["skimage", "protis"]:
        raise ValueError(f"Wrong method {method}. choose between `skimage` or `protis`")

    return (
        _get_iso_skimage(x, y, im, level)
        if method == "skimage"
        else _get_iso(x, y, im, level)
    )
