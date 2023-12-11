#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io

from IPython import get_ipython
from nannos.utils.time import *

from .helpers import *
from .jupyter import VersionTable

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(VersionTable)
