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

""" Tools to compute the properties of the activity """

import numpy as np


def find_idx_nearest(array, values):
    '''
    Find the indices of the nearest elements of `values` in `array`.
    Both ``array`` and ``values`` should be ``numpy.array``s and `array` MUST be
    sorted in increasing order.

    Parameters
    ----------
    array : reference list or np.ndarray
    values : double, list or array of values to find in `array`

    Returns
    -------
    idx : int or array representing the index of the closest value in `array`
    '''
    idx = np.searchsorted(array, values, side="left") # get the interval
    # return the index of the closest
    if isinstance(values, float) or isinstance(values, int):
        if idx == len(array):
            return idx-1
        else:
            return idx-(np.abs(values-array[idx-1]) < np.abs(values-array[idx]))
    else:
        # find where it is idx_max+1
        overflow = (idx == len(array))
        idx[overflow] -= 1
        # for the others, find the nearest
        tmp = idx[~overflow]
        idx[~overflow] = tmp - ( np.abs(values[~overflow]-array[tmp-1])
                                 < np.abs(values[~overflow]-array[tmp]) )
        return idx
