#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


from protis.threed import *

vecs = (1, 0, 0), (0, 1, 0), (0, 0, 1)
self = Lattice(vecs)

nh = 27
Lk = self.reciprocal

self.get_harmonics(nh)
