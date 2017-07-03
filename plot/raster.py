#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
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

# nest_plot.py

# This file is part of the NNGT module
# Distributed as a free software, in the hope that it will be useful, under the
# terms of the GNU General Public License.

""" Utility functions to plot Nactivity """

from itertools import cycle

import numpy as np

from PyNeurActiv.analysis import (ActivityRecord, get_b2, firing_rate,
                                  neurons_sorter)
from .plot_tools import _markers
from ..lib import nonstring_container


#-----------------------------------------------------------------------------#
# Plotting the activity
#------------------------
#

def raster_plot(activity, network=None, limits=None, sort=None, normalize=1.,
                decimate=None, hist=True, mark_network_bursts=True,
                mark_spike_patterns=True, title=None, axis=None, label=None,
                markers=None, show=True):
    '''
    Plot the monitored activity.
    
    Parameters
    ----------
    activity : :class:`~PyNeurActiv.analysis.ActivityRecord` or (N, 2) array
        Recorded spikes and senders, either from `raster_analysis` or directly
        from a 2d array. If an `ActivityRecord` is provided, then the names
        of the columns should be stated if they differ from default
        'neuron'/'time'.
    network : :class:`~nngt.Network` or subclass, optional (default: None)
        Network for which the activity was monitored.
    limits : tuple, optional (default: None)
        Time limits of the plot (if not specified, times of first and last
        spike for raster plots).
    show : bool, optional (default: True)
        Whether to show the plot right away or to wait for the next plt.show().
    sort : str or list, optional (default: None)
        Sorting method that will be used to attribute a new index to each
        neuron. See :func:`~PyNeurActiv.analysis.neurons_sorter`.
    normalize : float, optional (default: None)
        Normalize the recorded results by a given float.
    decimate : int or list of ints, optional (default: None)
        Represent only a fraction of the spiking neurons; only one neuron in
        `decimate` will be represented (e.g. setting `decimate` to 5 will lead
        to only 20% of the neurons being represented). If a list is provided,
        it must have one entry per NeuralGroup in the population.
    hist : bool, optional (default: True)
        Whether to display the histogram when plotting spikes rasters.
    title : str, optional (default: None)
        Title of the plot.
    fignum : int, optional (default: None)
        Plot the activity on an existing figure (from ``figure.number``).
    label : str, optional (default: None)
        Add a label to the plot.

    Warning
    -------
    Sorting with "firing_rate" only works if NEST gids form a continuous
    integer range.

    Returns
    -------
    axis : list
        List of the figure numbers.
    '''
    import matplotlib.pyplot as plt
    if not isinstance(activity, ActivityRecord):
        datadict = {'neuron': activity[:, 0], 'time': activity[:, 1]}
        activity = ActivityRecord(datadict)
    senders, times = activity.data.T
    senders = senders.astype(int)

    num_spikes = len(times)
    neurons = np.unique(senders).astype(int)
    num_neurons = len(neurons)
    from itertools import cycle
    colors = cycle(('r','g','b','k','y','gray'))

    # markers
    if markers is None and mark_network_bursts:
        markers = _markers
    elif not nonstring_container(markers):
        markers = cycle([markers])
    else:
        markers = cycle(['o'])

    # decimate if necessary
    if decimate is not None:
        idx_keep = np.where(np.mod(senders, decimate) == 0)[0]
        senders = senders[idx_keep]
        times = times[idx_keep]

    # sorting
    sorter = np.arange(neurons[-1] + 1, dtype=int)
    if sort == "spikes":
        sorter = activity._sort
    elif sort is not None:
        sorter = neurons_sorter(
            neurons, sort, data=activity.data, network=network)

    if len(times):
        if axis is None:
            _, axis = plt.subplots()

        ylabel = "Neuron ID"
        xlabel = "Time (ms)"
        show_burst = mark_spike_patterns * ('individual_burst' in activity)
        cburst = 'r' if show_burst else 'b'
        cspike = 'b'

        delta_t = 0.01*(times[-1]-times[0])

        burst = np.ones(num_spikes, dtype=bool)
        interburst = np.zeros(num_spikes, dtype=bool)
        iburst = np.ones(num_spikes, dtype=bool)
        ispike = np.zeros(num_spikes, dtype=bool)
        descriptor1, descriptor2 = np.ones(num_spikes), np.ones(num_spikes)
        try:
            num_events = int(np.nanmax(activity.array('network_burst')))
            descriptor1 = activity.array('network_burst')
            descriptor2 = activity.array('network_interburst')
        except:
            num_events = 1
        if show_burst:
            iburst = ~np.isnan(activity.array('individual_burst'))
        for c, ev, m in zip(colors, range(num_events), markers):
            burst = np.isclose(descriptor1, ev + 1)
            interburst = np.isclose(descriptor2, ev + 1)
            # burst (all spikes if no bursts)
            bb_idx = np.where(iburst & burst)[0]
            bs_idx = np.where(~iburst & burst)[0]
            axis.plot(times[bb_idx], sorter[senders[bb_idx]],
                      ls='', marker=m, c=cburst)
                      #~ ls='', marker=m, c=c)
            axis.plot(times[bs_idx], sorter[senders[bs_idx]],
                      ls='', marker=m, c=cspike)
                      #~ ls='', marker=m, c=c)
            # interburst (empty if no bursts)
            ib_idx = np.where(iburst & interburst)[0]
            is_idx = np.where(~iburst & interburst)[0]
            axis.plot(times[ib_idx], sorter[senders[ib_idx]],
                      ls='', marker=m, c=cburst, fillstyle='none')
                      #~ ls='', marker=m, c=c, fillstyle='none')
            axis.plot(times[is_idx], sorter[senders[is_idx]],
                      ls='', marker=m, c=cspike, fillstyle='none')
                      #~ ls='', marker=m, c=c, fillstyle='none')

        axis.set_ylabel(ylabel)
        axis.set_xlabel(xlabel)
        if limits is not None:
            axis.set_xlim(limits)
        else:
            axis.set_xlim([times[0]-delta_t, times[-1]+delta_t])
        axis.legend(bbox_to_anchor=(1.1, 1.2))

        if title is None:
            title = 'Raster plot'
        plt.title(title)
        if show:
            plt.show()
        return axis
