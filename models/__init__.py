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
=============
Models module
=============

Module containing theoretical or computational models to study or predict the
behavior of neuronal populations.


Content
=======
"""

from .fardet_2017_synchronous_bursting import Fardet2017_SynchroBurst

__all__ = [
    "Fardet2017_SynchroBurst",
    "Simulator_SynchroBurst",
]

try:
    from .simulator_synchronous_bursting import Simulator_SynchroBurst
    __all__.append('Simulator_SynchroBurst')
except ImportError:
    pass
