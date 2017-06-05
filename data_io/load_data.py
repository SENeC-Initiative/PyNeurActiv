#!/usr/bin/env cython
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

""" Loading data """

import numpy as np
from scipy.io import loadmat


__all__ = [ 'load_abf', 'load_fig' ]


def load_abf(filename):
    ''' Use ``neo`` to load data from a .abf file '''
    assert filename.endswith('abf'), "Not an .abf file"
    from neo import io
    r = io.AxonIO(filename=filename)
    bl = r.read_block(lazy=False, cascade=True)
    return bl


def load_fig(filename, get_properties=False, get_labels=False):
    ''' Use :func:`scipy.io.loadmat` to load data from a Matlab .fig file '''
    d = loadmat(filename, squeeze_me=True, struct_as_record=False)
    ax1 = d['hgS_070000'].children
    if np.size(ax1) > 1:
        ax1 = ax1[0]

    xs, ys, properties, labels = [], [], {}, {}

    counter = 0
    for line in ax1.children:
        if line.type == 'graph2d.lineseries':
            properties["marker"] = "%s" % line.properties.Marker
            properties["linestyle"] = "%s" % line.properties.LineStyle
            properties["color"] =  line.properties.Color
            xs.append(line.properties.XData)
            ys.append(line.properties.YData)
        elif line.type == 'text':
            if counter < 1:
                labels["x"] = "%s" % line.properties.String
            elif counter < 2:
                labels["y"] = "%s" % line.properties.String

    lst_return = [xs, ys]
    if get_properties:
        lst_return.append(properties)
    if get_labels:
        lst_return.append(labels)
    return tuple(lst_return)
