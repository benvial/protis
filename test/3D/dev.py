#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


from protis.threed import *

vecs = (1, 0, 0), (0, 1, 0), (0, 0, 1)
self = Lattice(vecs)

nh = 27
Lk = self.reciprocal

self.get_harmonics(nh)
