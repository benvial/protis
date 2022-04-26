#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io

import pytest


def test_backend():
    import protis as pt

    assert pt.get_backend() == "numpy"
    assert pt.BACKEND == "numpy"

    pt.set_backend("scipy")
    assert pt.numpy.__name__ == "numpy"
    assert pt.backend.__name__ == "numpy"
    assert pt.get_backend() == "scipy"
    assert pt.BACKEND == "scipy"

    pt.set_backend("autograd")
    assert pt.numpy.__name__ == "autograd.numpy"
    assert pt.backend.__name__ == "autograd.numpy"
    assert pt.get_backend() == "autograd"
    assert pt.BACKEND == "autograd"

    pt.set_backend("jax")
    assert pt.numpy.__name__ == "jax.numpy"
    assert pt.backend.__name__ == "jax.numpy"
    assert pt.get_backend() == "jax"
    assert pt.BACKEND == "jax"

    pt.set_backend("torch")
    assert pt.numpy.__name__ == "numpy"
    if pt.has_torch():
        assert pt.get_backend() == "torch"
        assert pt.backend.__name__ == "torch"
        assert pt.BACKEND == "torch"

    with pytest.raises(ValueError) as excinfo:
        pt.set_backend("fake")
    assert "Unknown backend" in str(excinfo.value)
    pt.set_backend("numpy")
    assert pt.numpy.__name__ == "numpy"
    assert pt.backend.__name__ == "numpy"
    assert pt.get_backend() == "numpy"
    assert pt.BACKEND == "numpy"


def test_notorch(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    import protis

    protis.set_backend("torch")

    protis.use_gpu(True)
    protis.use_gpu(False)


def test_gpu(monkeypatch):
    import protis

    protis.set_backend("torch")
    protis.use_gpu(True)
    protis.use_gpu(False)
