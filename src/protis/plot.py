#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io

__all__ = ["plot2d"]

from nannos.plot import *


def plot2d(
    lattice,
    grid,
    field,
    nper=1,
    ax=None,
    cmap="RdBu_r",
    show_cell=False,
    cellstyle="-k",
    **kwargs,
):

    return plot_layer(
        lattice,
        grid,
        field,
        nper,
        ax,
        cmap,
        show_cell,
        cellstyle,
        **kwargs,
    )
