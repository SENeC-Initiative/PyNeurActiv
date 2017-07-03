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

""" Tools for signal processing """

import numpy as np
import scipy.signal as sps


def _smooth(data, kernel_size, std, mode='same'):
    '''
    Convolve an array by a Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel array in bins.
    std : float
        Width of the Gaussian (also in bins).

    Returns
    -------
    convolved array.
    '''
    kernel = sps.gaussian(kernel_size, std)
    kernel /= np.sum(kernel)
    return sps.convolve(data, kernel, mode=mode)


def find_extrema(x, which='max'):
    '''
    Return the positions of the extremal values in a 1D-array.

    Note
    ----
    In the case of N consecutive equal values that would be local maxima or
    minima, only the last position is returned.
    Peaks located at the extremities of `x` are not considered.

    Parameters
    ----------
    x : array
        Array where the extrema will be searched.
    which : str, optional (default: 'max')
        Type of extrema that are considered, either 'max' or 'min'.
    '''
    lx = len(x)
    comparator = np.greater_equal if which == 'max' else np.less_equal
    checker = np.greater if which == 'max' else np.less
    peaks = sps.argrelextrema(x, comparator=comparator)[0].astype(int)
    keep = np.ones(len(peaks), dtype=bool)
    if len(peaks) > 1:
        keep = np.where(
            (np.diff(peaks) > 1)
            & checker(x[peaks[:-1]], x[peaks[:-1] + 1]))[0]
        add_last = ([peaks[-1]] if (peaks[-1] < lx - 1
            and checker(x[peaks[-1]], x[min(lx - 1, peaks[-1] - 1)])) else [])
        return np.array(peaks[keep].tolist() + add_last, dtype=int)
    elif len(peaks) == 1 and (peaks[0] == 0 or peaks[0] == lx - 1):
            return []
    return peaks
