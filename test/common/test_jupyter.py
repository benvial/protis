#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


def test_jupyter():
    import protis

    vt = protis.VersionTable()
    vt.protis_version_table()
