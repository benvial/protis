#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of protis
# License: GPLv3
# See the documentation at protis.gitlab.io


import importlib.metadata as metadata


def _get_metadata(metadata):
    """
    Get package metadata.

    Returns
    -------
        tuple: A tuple of package metadata including version, author,
            description, and a dictionary of additional metadata.
    """
    try:
        data = metadata.metadata("protis")
        __version__ = metadata.version("protis")
        __author__ = data.get("Author-email").split("<")[0][:-1]
        __description__ = data.get("summary")
    except Exception:
        data = dict(License="unknown")
        __version__ = "unknown"
        __author__ = "unknown"
        __description__ = "unknown"
    return __version__, __author__, __description__, data


__version__, __author__, __description__, data = _get_metadata(metadata)
