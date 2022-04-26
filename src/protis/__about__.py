#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


try:
    # Python 3.8
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata


def get_meta(metadata):
    try:
        data = metadata.metadata("protis")
        __version__ = metadata.version("protis")
        __author__ = data.get("author")
        __description__ = data.get("summary")
    except Exception:
        data = dict(License="unknown")
        __version__ = "unknown"
        __author__ = "unknown"
        __description__ = "unknown"
    return __version__, __author__, __description__, data


__version__, __author__, __description__, data = get_meta(metadata)
