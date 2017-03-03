"""
===================================
Neuronal activity package (SENeC-I)
===================================

PyNeurActv pacakage to study the simulated activity of neural networks.
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
"""

from __future__ import absolute_import
import sys


# ----------------------- #
# Requirements and config #
# ----------------------- #

# Python > 2.6
assert sys.hexversion > 0x02060000, "PyNeurActiv requires Python > 2.6"


# ------- #
# Modules #
# ------- #

from . import analysis
from . import io
from . import lib

__all__ = [
    "analysis",
    "io",
    "lib",
    "models",
]
