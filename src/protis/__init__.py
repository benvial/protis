#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io

import os

from .__about__ import __author__, __description__, __version__
from .__about__ import data as _data


def _reload_package():

    import importlib
    import sys

    import protis

    importlib.reload(protis)

    its = [s for s in sys.modules.items() if s[0].startswith("protis")]
    for k, v in its:
        importlib.reload(v)


_nannos_env_var = os.environ.get("NANNOS_BACKEND")
if _nannos_env_var is not None:
    del os.environ["NANNOS_BACKEND"]
import nannos
from nannos import *

del PlaneWave
del excitation
del print_info
del formulations
del Simulation
del simulation
del utils
del layers


def set_backend(backend):
    global _FORCE_BACKEND
    _FORCE_BACKEND = 1
    nannos.set_backend(backend)
    _reload_package()


def use_gpu(boolean):
    nannos.use_gpu(boolean)
    _reload_package()


_backend_env_var = os.environ.get("PROTIS_BACKEND")

if _backend_env_var in available_backends and _backend_env_var is not None:
    if BACKEND != _backend_env_var and not "_FORCE_BACKEND" in globals():
        logger.debug(f"Found environment variable PROTIS_BACKEND={_backend_env_var}")
        set_backend(_backend_env_var)


def print_info():
    print(f"protis v{__version__}")
    print("=============")
    print(__description__)
    print(f"Author: {__author__}")
    print(f"Licence: {_data['License']}")


from .simulation import *
from .utils import *
