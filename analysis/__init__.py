#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of the PyNeurActiv project, which aims at providing tools
# to study and model the activity of neuronal cultures.
# Copyright (C) 2017 SENeC Initiative
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
===============
Analysis module
===============

Tools to work on simulated or recorded data.

Content
=======
"""

from . import activity_properties as _ap
from . import array_searching as _as

from .activity_properties import *
from .array_searching import *


__all__ = _ap.__all__ + _as.__all__


try:
    from .custom_recording import Recorder
    __all__.append("Recorder")
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Could not load `Recorder` class: " + str(e))

