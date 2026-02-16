#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


def test_metadata(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "importlib.metadata", None)
    import protis


def test_nometadata():
    import importlib

    import protis

    importlib.reload(protis.__about__)


def test_data():
    import protis

    protis.__about__._get_metadata(None)


def test_info():
    import protis

    protis.print_info()
