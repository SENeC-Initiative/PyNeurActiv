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

""" Plotting data """

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

from .load_data import *


__all__ = [ 'plot_neo', 'plot_abf', 'plot_fig' ]


def plot_neo(block, xunit=None, show=True):
    ''' Plot the data present in a ``neo.Block`` object '''
    # get the data units (current or voltage)
    units = []
    for s in block.segments:
        units.append(tuple(sg.dimensionality.string for sg in s.analogsignals))
    tunit = block.segments[0].analogsignals[0].times.dimensionality.string
    xunit = tunit if xunit is None else xunit
    set_units = tuple(set(units))
    # make a figure for each identical set of units
    for unit_set in set_units:
        # one subplot per unit
        fig, axes = plt.subplots(len(unit_set), sharex=True)
        if not hasattr(axes, "__len__"):
            axes = [axes]
        # plot them if the unit set correspond to the current figure
        for s, uset in zip(block.segments, units):
            if uset == unit_set:
                for i, asig in enumerate(s.analogsignals):
                    times = asig.times.rescale(xunit).magnitude
                    asig = asig.rescale(uset[i]).magnitude
                    axes[i].plot(times, asig)
        for ax, unit in zip(axes, uset):
            ylabel = "V_m ({})" if "V" in unit else "Current ({})"
            ax.set_ylabel(ylabel.format(unit))
            ax.set_xlabel("Time ({})".format(xunit))
    if show:
        plt.show()


def plot_abf(filename, xunit=None, show=True):
    ''' Plot data from a .abf file using neo '''
    bl = load_abf(filename)
    plot_neo(bl, xunit=xunit, show=show)

def plot_fig1(filename, show=True):
    ''' Plot data from a Matlab .fig file '''
    fig = plt.figure()
    x, y, prop, labels = load_fig(filename,get_properties=True,get_labels=True)
    plt.plot(x, y, **prop)
    plt.xlabel(labels.get("x", "x"))
    plt.ylabel(labels.get("y", "y"))
    if show:
        plt.show()


def plot_fig(filename, show=True):
    ''' Plot a Matlab .fig '''
    d = loadmat(filename, squeeze_me=True, struct_as_record=False)
    ax1 = d['hgS_070000'].children
    if np.size(ax1) > 1:
        ax1 = ax1[0]
    # plot
    counter = 0
    for line in ax1.children:
        if line.type == 'graph2d.lineseries':
            marker = "%s" % line.properties.Marker
            linestyle = "%s" % line.properties.LineStyle
            r,g,b = line.properties.Color
            x = line.properties.XData
            y = line.properties.YData
            plt.plot(x,y,marker,linestyle=linestyle, color=(r,g,b))
        elif line.type == 'text':
            if counter < 1:
                plt.xlabel("%s" % line.properties.String, fontsize=16)
                counter += 1
            elif counter < 2:
                plt.ylabel("%s" % line.properties.String, fontsize = 16)
                counter += 1
    if show:
        plt.show()
