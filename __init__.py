#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===================================
Neuronal activity package (SENeC-I)
===================================

PyNeurActiv pacakage to study the simulated activity of neural networks.
This package is part of the broader SENeC initiative for the study of neuronal
cultures and devices.


Content
=======

`analysis`
	Tools to analyze data related to neuronal activity, especially in link
    with simulations involving [NEST][nest] or [NNGT][nngt].
`io`
    Input/output functions to load and plot, sometimes based on [Neo][neo].
`lib`
    Generic tools used throughout the modules.
`models`
    Main module containing theoretical and numerical models to predict the
    activity of neuronal populations.
"""

import logging as _logging
import sys as _sys


# ----------------------- #
# Requirements and config #
# ----------------------- #

# Python > 2.6
assert _sys.hexversion > 0x02060000, "PyNeurActiv requires Python > 2.6"

__version__ = "0.1.0"

# logging
_log = _logging.getLogger(__name__)

if not _log.handlers:
    logConsoleFormatter = _logging.Formatter(
        '[%(levelname)s @ %(name)s]: %(message)s')
    consoleHandler = _logging.StreamHandler()
    consoleHandler.setFormatter(logConsoleFormatter)
    consoleHandler.setLevel(_logging.INFO)
    _log.addHandler(consoleHandler)


# ------- #
# Modules #
# ------- #

from . import analysis
from . import data_io
from . import lib
from . import models


__all__ = [
    "analysis",
    "data_io",
    "lib",
    "models",
    "Recorder",
]

try:
    import matplotlib
    from . import plot
    __all__.append('plot')
except ImportError:
    pass

try:
    from .analysis import Recorder
    __all__.append('Recorder')
except ImportError:
    pass
