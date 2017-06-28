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
