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

try:
    import cPickle as pickle
except:
    import pickle
import logging
import weakref
from itertools import chain

import numpy as np
import scipy.sparse as ssp
from scipy.interpolate import interp1d

from .array_searching import find_idx_nearest
from .bayesian_blocks import bayesian_blocks
from ..lib.signal_processing import _smooth
from ..lib import nonstring_container, find_extrema


__all__ = [
    "ActivityRecord",
    "activity_types",
    "analyze_raster",
    "data_from_nest",
    "firing_rate",
    "get_spikes",
    "get_b2",
    "interburst_properties",
    "neurons_sorter",
    "raster_analysis",
    "read_pickle",
    "spiking_properties",
]


logger = logging.getLogger(__name__)


# ---------------- #
# Test for Pandas  #
# ---------------- #

try:
    import pandas as pd
    _with_pandas = True
    RecordParent = pd.DataFrame
except ImportError:
    _with_pandas = False
    RecordParent = dict


# ------------------------------------ #
# Record class for activity properties #
# ------------------------------------ #

class ActivityRecord(RecordParent):

    '''
    Class to record the properties of the simulated activity.
    '''

    def __init__(self, datadict, phases=None, spikes_data=('neuron', 'time'),
                 properties=None, parameters=None, sort=None, network=None,
                 **kwargs):
        '''
        Initialize the instance using `spike_data` (store proxy to an optional
        `network`) and compute the properties of provided data.

        Parameters
        ----------
        datadict : dict of 1D arrays
            Dictionary containing at least the indices of the spiking entities
            as well as the spike times.
        spikes_data : 2-tuple, optional (default: ('neuron', 'time'))
            Tuple containing the keys for the indices of the spiking entities
            and for the spike times.
        properties : dict
            Values of the different properties of the activity (e.g.
            "firing_rate", "IBI"...).
        parameters : dict, optional (default: None)
            Parameters used to compute the phases.
        sort : 1d array of length `neuron_max_id + 1`, optional (default: None)
            Array containing the rank associated to each neuron according to
            the number of spikes it fired inside and outside bursts.
            (more spikes in bursts lower the rank, more spikes out of bursts
            increases it)
        **kwargs : dict
            Keyword arguments. For instance the global firing rate can be
            passed as the entry 'fr' associated to a (fr, time) tuple.

        Note
        ----
        The firing rate is computed as num_spikes / total simulation time, the
        period is the sum of an IBI and a bursting period.
        '''
        super(ActivityRecord, self).__init__(datadict)
        self._data_keys = spikes_data
        self._parameters = parameters
        self._phases = None if phases is None else phases.copy()
        self._properties = None if properties is None else properties.copy()
        self._sort = sort
        self.network = None if network is None else weakref.ref(network)
        if 'fr' in kwargs:
            self._fr = np.array(kwargs['fr'][0])
            self._fr_times = kwargs['fr'][1]
        else:
            self._fr, self._fr_times = None, None
        self._compute_properties()

    def __getstate__(self):
        ''' Necessary for pickling '''
        try:
            parent_state = super(RecordParent, self).__getstate__()
        except AttributeError:
            parent_state = super(RecordParent, self).__dict__.copy()
        own_state = self.__dict__.copy()
        return (parent_state, own_state)

    def __setstate__(self, state):
        ''' Necessary for pickling '''
        parent_state , own_state = state
        self.__dict__ = own_state
        try:
            super(RecordParent, self).__setstate__(parent_state)
        except AttributeError:
            super(RecordParent, self).__dict__ = parent_state

    @property
    def data(self, sort=False):
        '''
        Returns the (N, 2) array of (senders, spike times).
        '''
        sender, time = self._data_keys
        d = None
        if _with_pandas:
            d = np.array([self[sender].values, self[time].values]).T
        else:
            d = np.array([self[sender], self[time]]).T
        if sort:
            if self._sort is None:
                raise RuntimeError('Cannot sort because no sorting data was '
                                   'provided.')
            d[:, 0] = self._sort[d[:, 0]]
        return d

    @property
    def parameters(self):
        ''' Returns the parameters used to compute the properties '''
        return self._parameters

    @property
    def phases(self):
        '''
        Returns
        -------
        A dict with the detected phases, which can be among:

        - "network_burst" for periods of high activity where a large fraction
          of the network is recruited.
        - "quiescent" for periods of low activity
        - "mixed" for firing rate in between "quiescent" and "network_burst".
        - "local_burst" for periods of high activity but where only a small
          fraction of the network is recruited.
        - "unknown" for uncharacterized phases.

        Note
        ----
        Phases that are not present in the activity are not added to the dict.
        See `parameters` for details on the conditions used to differenciate
        these phases.
        '''
        return None if self._phases is None else self._phases.copy()

    def array(self, key):
        if _with_pandas:
            return self[key].values
        else:
            return self[key]

    def firing_rate(self, kernel_center=None, kernel_std=None,
                    resolution=None):
        '''
        Global firing rate of the network.

        Returns
        -------
        fr, times
        '''
        kw = ('kernel_center', 'kernel_std', 'resolution')
        val = (kernel_center, kernel_std, resolution)
        default = val == (None, None, None)
        if not (self._fr is not None and default):
            kwargs = {k: v for k, v in zip(kw, val) if v is not None}
            self._fr, self._fr_times = firing_rate(
                self[self._data_keys[1]], **kwargs)
        return self._fr, self._fr_times

    def neuron_ranking(self, neurons, sort='spikes'):
        '''
        Sort the neurons according to `sort`, which can be either their
        'spikes'-rank (difference between the number of spike outside and
        inside bursts, i.e. number of spikes if there is no bursting activity),
        or their B2 coefficient.

        Parameters
        ----------
        neurons : array-like of length N
            Array containing neuron ids.
        sort : str, optional (default: 'rank')
            Sorting method that will be used to attribute a new index to each
            neuron.

        Returns
        -------
        new_indices : array-like of length N
            Entry `i` contains a new index corresponding to the rank of
            `neurons[i]` among the sorted neurons.
        '''
        if sort == 'spikes':
            if self._sort is None:
                raise RuntimeError("Cannot use 'spikes'-sort because no "
                                   "sorting data was provided.")
            return self._sort[neurons]
        elif sort in ('b2', 'B2'):
            sender, time = self._data_keys
            set_neurons = np.unique(self[sender])
            b2 = get_b2(
                senders=self[sender], spike_times=self[time])[set_neurons]
            sorter = np.arange(0, set_neurons[-1] + 1, dtype=int)
            sorted_idx = np.argsort(np.argsort(b2))
            sorter[set_neurons] = sorted_idx
            return sorter[neurons]

    def properties(self, which='average'):
        '''
        Returns the properties of the activity.
        Contains the following entries:

        - "firing_rate": average value in Hz for 1 neuron in the network.
        - "network_bursting": True if there were bursts of activity detected.

        If `which` is 'list' or 'both', also contains:

        - "burst_durations", "IBIs", and "ISIs" in ms, if "network_bursting".
        - "SpBs" (Spikes per Bursts): average number of spikes per neuron
          during a burst, if "network_bursting".

        If `which` is 'average' or 'both', also contains the 'mean_X' entries,
        where 'X' is the singular of the previously listed entries.

        Parameters
        ----------
        which : str, optional (default: 'average')
            Type of properties that should be returned; either the 'average'
            values over the whole activity, the detailed 'list', or 'both'.

        Returns
        -------
        properties : dict
        '''
        properties = {'network_bursting': self._properties['network_bursting']}
        if self._properties is None:
            return None
        elif which == 'both':
            properties = self._properties.copy()
        elif which == 'average':
            properties.update({
                k:v for k, v in self._properties.items()
                if not nonstring_container(v)
            })
        elif which == 'list':
            properties.update({
                k:v for k, v in self._properties.items()
                if nonstring_container(v)
            })
        else:
            raise RuntimeError('Invalid `which` argument: {}'.format(which))
        return properties

    def simplify():
        raise NotImplementedError("Will hopefully be implemented one day.")

    def to_pickle(self, filename):
        with open(filename, 'w') as f:
            pickle.dump(self, f)

    def _compute_properties(self, skip_bursts=0):
        fr, _ = self.firing_rate()
        avg_fr = np.average(fr)
        phases = {'network_burst': []} if self.phases is None else self.phases
        self._properties = _compute_properties(
            self.data, phases, avg_fr, skip_bursts)


# ------------------- #
# Get data from NEST  #
# ------------------- #

def data_from_nest(recorders):
    ''' Return spike and variables' data '''
    import nest
    # spikes
    data = nest.GetStatus(recorders[0])[0]["events"]
    spike_times = data["times"]
    senders = data["senders"]
    time_var, data_var = [], []
    # variables
    if len(recorders) > 1:
        base_data2 = nest.GetStatus(recorders[1])
        data2 = [d["events"] for d in base_data2]
        time_var = np.array(data2[0]["times"])
        data_var = {key: [] for key in data2[0]
                    if key not in ("senders", "times")}
        for d in data2:
            for key, val in d.items():
                if key not in ("senders", "times"):
                    data_var[key].append(val)
    return spike_times, senders, time_var, data_var


def get_spikes(recorder=None, spike_times=None, senders=None, skip=None,
               network=None):
    '''
    Return a 2D sparse matrix, where:

    - each row i contains the spikes of neuron i
    - each column j contains the times of the jth spike for all neurons

    Parameters
    ----------
    recorder : tuple, optional (default: None)
        Tuple of NEST gids, where the first one should point to the
        spike_detector which recorded the spikes.
    spike_times : array-like, optional (default: None)
        If `recorder` is not provided, the spikes' data can be passed directly
        through their `spike_times` and the associated `senders`.
    senders : array-like, optional (default: None)
        `senders[i]` corresponds to the neuron which fired at `spike_times[i]`.
    skip : double, optional (default: None)
        Number of ms that should be skipped (keep only the spikes that occur
        after this duration).
    network : :class`nngt.Network`, optional (default: None)
        Network for which the activity was recorded. If provided, the neurons
        will be registered via their NNGT ids instead of their NEST gids.

    Example
    -------
    >> get_spikes()

    >> get_spikes(recorder)

    >> times = [1.5, 2.68, 125.6]
    >> neuron_ids = [12, 0, 65]
    >> get_spikes(spike_times=times, senders=neuron_ids)

    Returns
    -------
    CSR matrix containing the spikes sorted by neuron (rows) and time
    (columns).
    '''
    # get spikes
    skip = 0. if skip is None else skip
    if recorder is not None:
        import nest
        data = nest.GetStatus(recorder[0])[0]["events"]
        spike_times = data["times"]
        senders = (data["senders"] if network is None
                   else network.id_from_nest_gid(data["senders"]))
    elif spike_times is None and senders is None:
        import nest
        nodes = nest.GetNodes(
            (0,), properties={'model': 'spike_detector'})
        data = nest.GetStatus(nodes[0])[0]["events"]
        spike_times = data["times"]
        senders = (data["senders"] if network is None
                   else network.id_from_nest_gid(data["senders"]))
    # create the sparse matrix
    data = {n: 0 for n in set(senders)}
    row_idx = []
    col_idx = []
    times = []
    for time, neuron in zip(spike_times, senders):
        if time > skip:
            row_idx.append(neuron)
            col_idx.append(data[neuron])
            times.append(time)
            data[neuron] += 1
    return ssp.csr_matrix((times, (row_idx, col_idx)))


def get_b2(recorder=None, spike_times=None, senders=None):
    '''
    Return an array containing the B2 coefficient for each neuron, as defined
    in van Elburg, van Ooyen, 2004
    (http://doi.org/10.1016/j.neucom.2004.01.086).

    Parameters
    ----------
    recorder : tuple, optional (default: None)
        Tuple of NEST gids, where the first one should point to the
        spike_detector which recorded the spikes.
    spike_times : array-like, optional (default: None)
        If `recorder` is not provided, the spikes' data can be passed directly
        through their `spike_times` and the associated `senders`.
    senders : array-like, optional (default: None)
        `senders[i]` corresponds to the neuron which fired at `spike_times[i]`.

    Note
    ----
    This function supposes that neuron GIDs for a continuous set of integers.
    If no arguments are passed to the function, the first spike_recorder
    available in NEST will be used.

    Returns
    -------
    b2 : :class:`numpy.ndarray` of length `max_neuron_id + 1`
        B2 coefficients (neurons for which no spikes happened have a NaN value;
        neurons having only one or two spikes have infinite B2).
    '''
    spikes = get_spikes(recorder, spike_times, senders)
    neurons = np.unique(senders).astype(int)
    b2 = np.full(neurons[-1] + 1, np.NaN)
    for n in neurons:
        isi_n = np.diff(spikes[n].data)
        if len(isi_n) in (0, 1):
            b2[n] = np.inf
        elif len(isi_n) > 1:
            isi2_n = isi_n[:-1] + isi_n[1:]
            avg_isi = np.mean(isi_n)
            if avg_isi != 0.:
                b2[n] = (2*np.var(isi_n) - np.var(isi2_n)) / (2*avg_isi**2)
    return b2


# ------------------------------------- #
# Analyze properties from spike trains  #
# ------------------------------------- #

def raster_analysis(raster, collective=0.1, limits=None, network=None,
                    skip_bursts=0, bins='doane', smooth=True, tolerance=0.05,
                    num_steps=1000, sender='neuron', time='time', axis=None,
                    show=False, **kwargs):
    '''
    Return the activity types for a given raster.

    Warning
    -------
    This function expects the spike times to be sorted!

    Parameters
    ----------
    raster : array-like of shape (N, 2) or str
        Either an array containing the ids (first row) of the spiking neurons
        and the corresponding times (second row), the gids of NEST recorders,
        or the path to a NEST-like recording.
    collective : float, optional (default: 0.1)
        Fraction of the population required for a behaviour to be considered as
        collective (e.g. network burst).
    limits : tuple of floats
        Time limits of the simulation region which should be studied (in ms).
    network : :class:`nngt.Network`, optional (default: None)
        Network on which the recorded activity was simulated.
    skip_bursts : int, optional (default: 0)
        Skip the `skip_bursts` first bursts to consider only the permanent
        regime.
    bins : str, optional (default: 'bayesian')
        Method that should be used to bin the interspikes and find the main
        intervals. Default uses Bayesian blocks, otherwise, any valid `bins`
        value for :func:`numpy.histogram` can be used.
    smooth : bool or float, optional (default: True)
        Smooth the ISI distribution to find the maxima. By default, the bins
        are smoothed by a Gaussian kernel of width the average interspike.
        If smooth is provided as a float, then this value will be taken as the
        width of the Gaussian kernel.
    tolerance : float, optional (default: 0.1)
        Tolerance for testing the validity of bursts : the ISI inside a burst
        should be smaller that `(1 + tolerance)*kernel_std`, where `kernel_std`
        is the standard deviation of the Gaussian used to smooth the firing
        rate.
    num_steps : int, optional (default: 1000)
        Number of steps to descretize and smooth the histogram.
    axis : :class:`matplotlib.axis.Axis` instance, optional (default: new one)
        Existing axis on which the data should be added.
    show : bool, optional (default: False)
        Display the figures.
    sender : str, optional (default: 'neuron')
        Name of the first column, designating the object from which the spike
        originated.
    time : str, optional (default: 'time')
        Name of the 2nd column, designating the spike time.
    **kwargs : additional arguments for the 'bayesian' binning function, such
        as `min_width` or `max_width` to constrain bin size.

    Note
    ----
    Effects of `skip_bursts` and `limits[0]` are cumulative: the
    `limits[0]` first milliseconds are ignored, then the `skip_bursts`
    first bursts of the remaining activity are ignored.

    Returns
    -------
    activity : :class:`~PyNeurActiv.analysis.ActivityRecord`
        Object containing the phases and the properties of the activity
        from these phases.

    Note
    ----
    If bursts are detected, spikes that do not belong to a burst are registred
    as NaN. For that reason, burst and interburst numbers are floats.
    '''
    data = _get_data(raster)
    if limits is None:
        limits = [np.min(data[:, 1]), np.max(data[:, 1])]
    start = np.argwhere(data[:, 1] > limits[0])[0][0]
    stop = np.argwhere(data[:, 1] < limits[1])[-1][0]
    # container
    activity = {
        sender: data[start:stop, 0].astype(int),
        time: data[start:stop, 1]}
    num_spikes = len(data[start:stop, 1])

    # test for bursting through the interspike intervals
    isi = []                               # values of the interspikes
    isi_positions = []                     # idx range in `isi` for each neuron
    spike_positions = []                   # spikes indices in `activity` 
    neurons = np.unique(activity[sender])  # GIDs of the neurons
    num_neurons = len(neurons)             # number of neurons

    for neuron in neurons:
        current_spikes = np.nonzero(activity[sender] == neuron)[0]
        spike_positions.append(current_spikes)
        n = len(isi)
        isi.extend(np.diff(activity[time][current_spikes]))
        isi_positions.append([n, len(isi)])
    isi = np.array(isi)

    # binning
    kwargs['min_width'] = kwargs.get('min_width', np.min(isi))
    if bins == 'bayesian':
        bins = bayesian_blocks(isi, **kwargs)
    counts, bins = np.histogram(isi, bins)

    #~ import matplotlib.pyplot as plt
    #~ plt.figure()
    #~ plt.hist(isi, bins)
    #~ plt.figure()

    if smooth:
        step = bins[-1] / float(num_steps)
        x = np.linspace(0., bins[-1], num_steps)
        y = interp1d(
            bins, list(counts) + [counts[-1]], kind='nearest',
            bounds_error=False, fill_value=0)
        interpolated = y(x)
        sigma = 0.1 * (np.max(isi) - np.min(isi)) if smooth is True else smooth
        bins = x
        sigma_in_step = max(sigma / step, 1.)
        kernel_size = 5*sigma_in_step
        counts = _smooth(interpolated, kernel_size, sigma_in_step)

    # maxima (T_min, ..., T_max) of the histogram
    local_max = find_extrema(counts, 'max')

    #~ plt.plot(bins, counts)
    #~ plt.plot(bins[local_max], counts[local_max], ls="", marker="s", fillstyle='none', c="r")
    
    if len(counts) >= 2 and counts[0] > counts[1]:
        local_max = [0] + local_max.tolist()
    if len(counts) >= 2 and counts[-1] > counts[-2]:
        local_max = list(local_max) + [len(counts) - 1]

    if len(local_max) == 2:
        # we are bursting, so we can assign spikes to a given burst or to an
        # interburst period interbursts
        burst = np.full(num_spikes, np.NaN)       # NaN if not in a burst
        interburst = np.full(num_spikes, np.NaN)  # NaN if not in an interburst
        # Count the spikes in bursts for each neuron
        spks_loc = {
            'neuron': np.array(list(neurons), dtype=int),
            'spks_in_bursts': np.zeros(num_neurons),
            'spks_in_interbursts': np.zeros(num_neurons),
        }
        # use a clustering method to separate burst from interburst: take the
        # average of the ISI and IBI, then cluster at equal distance. Thus, all
        # spikes with ISI < (3*T_min + T_max) / 4 are considered inside a burst
        isi_high = (3*bins[local_max[0]] + bins[local_max[1]]) / 4.
        i = 0
        for isi_pos, spike_pos in zip(isi_positions, spike_positions):
            in_a_burst = isi[isi_pos[0]:isi_pos[1]] < isi_high
            if np.any(in_a_burst):
                pos_first_spikes_burst = _pos_first_spike_burst(in_a_burst)
                # assign each spike to the burst where it belongs and count the
                # number of spikes inside bursts and inside interbursts
                nsb, nsi = _set_burst_num(pos_first_spikes_burst, in_a_burst,
                                          spike_pos, burst, interburst)
                spks_loc['spks_in_bursts'][i] = nsb
                spks_loc['spks_in_interbursts'][i] = nsi
                i += 1
        if np.any(~np.isnan(burst)):
            activity['individual_burst'] = burst
            activity['individual_spike'] = interburst
    elif len(local_max) > 2:
        logger.warning("Complex activity detected, manual processing will be "
                       "necessary.")

    # network burst detection
    kernel_stds = [20., 80.]
    if isinstance(smooth, float):
        kernel_stds.append(smooth)
    elif len(local_max) > 2:
        kernel_stds.append(max(0.1 * bins[local_max[1]], kernel_stds[0]))
    elif len(local_max):
        kernel_stds.append(0.5 * (0.1*bins[local_max[0]] + kernel_stds[0]))
        kernel_stds.append(max(0.1 * bins[local_max[0]], kernel_stds[0]))
    # check whether the `collective` fraction is reached in the peaks of the
    # firing rate
    # We can assign spikes to a given burst or to an
    # interburst period interbursts
    net_burst = np.full(num_spikes, np.NaN)       # NaN if not in a burst
    net_interburst = np.full(num_spikes, np.NaN)  # NaN if not in an interburst
    burst_phases = []
    burst_nb, current_position = 0, 0
    phases, properties = {}, {}

    success = False
    while not success and kernel_stds:
        kernel_std = kernel_stds.pop()
        fr, fr_time = firing_rate(
            activity[time], kernel_std=kernel_std)
        peaks = find_extrema(fr, 'max')

        #~ plt.figure()
        #~ plt.plot(fr_time, fr)
        #~ plt.plot(fr_time[peaks], fr[peaks], ls="", marker="s", fillstyle='none', c="r")

        invalid_burst = False

        for i, p in enumerate(peaks):
            # test whether the fraction is reached between the two half_widths
            half_heights = np.where(fr < 0.5 * fr[p])[0]
            end = np.where(half_heights > p)[0]
            # boundaries
            bstop = half_heights[end[0]] if len(end) else len(fr)
            bstart = half_heights[end[0] - 1] if (len(end) and end[0]) else 0
            # check that no other peak lies between `p` and the boundaries
            if i > 0 and fr_time[bstart] < fr_time[peaks[i - 1]]:
                bstart = int(np.ceil(0.5*(p + peaks[i - 1])))
            if i < len(peaks) - 1 and fr_time[bstop] > fr_time[peaks[i + 1]]:
                bstop = int(np.ceil(0.5*(p + peaks[i + 1])))
            # count the participating neurons
            burst_spikes = np.where(
                (activity[time] > fr_time[bstart])
                 * (activity[time] < fr_time[bstop-1]))[0]
            # test burst validity
            participating = len(np.unique(activity[sender][burst_spikes]))
            invalid_burst = np.any(np.diff(
                activity[time][burst_spikes]) > (1 + tolerance)*kernel_std)
            if participating / float(num_neurons) >= collective:
                if invalid_burst:
                    # reset everything and break
                    burst_phases = []
                    burst_nb, current_position = 0, 0
                    net_burst = np.full(num_spikes, np.NaN)
                    net_interburst = np.full(num_spikes, np.NaN)
                    break
                burst_nb += 1
                net_burst[burst_spikes] = burst_nb
                net_interburst[current_position:burst_spikes[0]] = burst_nb
                burst_phases.append((activity[time][burst_spikes[0]],
                                     activity[time][burst_spikes[-1]]))
                current_position = burst_spikes[-1] + 1

        # save phases and check for success
        phases['network_burst'] = burst_phases
        success = not invalid_burst

    # if bursts where detected, add them to the activity
    if burst_nb:
        if not success:
            logger.warning("Automatic labeling of the bursts failed!")
        activity['network_burst'] = net_burst
        activity['network_interburst'] = net_interburst

    # neuron sorting
    sorter = np.arange(0, np.max(activity[sender]) + 1, dtype=int)
    if 'individual_burst' in activity:
        rank = spks_loc['spks_in_interbursts'] - spks_loc['spks_in_bursts']
        asort_rank = np.argsort(rank)
        sorter[spks_loc['neuron']] = np.argsort(asort_rank)

    return ActivityRecord(
        activity, spikes_data=(sender, time), sort=sorter, phases=phases,
        fr=(fr, fr_time))


def interburst_properties(bursts, current_index, steady_state, times,
                          variables, resolution, result):
    '''
    Find the end of the previous burst, then compute the interburst (IBI)
    duration and the extremal values of the variables during the interburst.
    '''
    current_burst = bursts[current_index + steady_state]
    # Time
    IBI_start = bursts[current_index + steady_state-1][1]
    IBI_end = current_burst[0]
    result["IBI"] += IBI_end - IBI_start
    # time slice of the IBI to array indices
    idx_start = np.argwhere(times >= IBI_start)[0][0]
    idx_end = np.argwhere(times < IBI_end-resolution)[-1][0]
    idx_wmax = np.argwhere(times > current_burst[1])[0][0]
    for varname, varvalues in iter(variables.items()):
        varname = "V" if varname == "V_m" else varname
        result[varname + "_min"] += np.min(varvalues[0][idx_start:idx_end])
        result[varname + "_max"] += np.max(varvalues[0][idx_start:idx_end])


def spiking_properties(burst, spike_times, senders, result):
    '''
    Compute the average and standard deviation of the interspike interval (ISI)
    as well as those of the number of spikes during the burst.
    '''
    # get the spikes inside the burst
    spikes = np.where( (spike_times >= burst[0])*(spike_times <= burst[1]) )[0]
    # get the number and ISI for each spike, then average
    lst_num_spikes = []
    lst_ISI = []
    for sender in set(senders[spikes]):
        subset = np.where(senders[spikes] == sender)[0]
        lst_num_spikes.append(len(subset))
        stimes = spike_times[spikes][subset]
        if len(stimes) > 1:
            lst_ISI.extend(np.diff(stimes))
    result["num_spikes"] += np.average(lst_num_spikes)
    result["std_num_spikes"] += np.std(lst_num_spikes)
    result["ISI"] += np.average(lst_ISI)
    result["std_ISI"] += np.std(lst_ISI)


def firing_rate(spike_times, kernel_center=0., kernel_std=30., resolution=None,
                cut_gaussian=5.):
    '''
    Computes the firing rate from the spike times.
    Firing rate is obtained as the convolution of the spikes with a Gaussian
    kernel characterized by a standard deviation and a temporal shift.

    Parameters
    ----------
    spike_times : array-like
        Array containing the spike times (in ms) from which the firing rate
        will be computed.
    kernel_center : float, optional (default: 0.)
        Temporal shift of the Gaussian kernel, in ms.
    kernel_std : float, optional (default: 30.)
        Characteristic width of the Gaussian kernel (standard deviation) in ms.
    resolution : float, optional (default: `0.1*kernel_std`)
        The resolution at which the firing rate values will be computed.
        Choosing a value smaller than `kernel_std` is strongly advised.
    cut_gaussian : float, optional (default: 5.)
        Range over which the Gaussian will be computed. By default, we consider
        the 5-sigma range. Decreasing this value will increase speed at the
        cost of lower fidelity; increasing it with increase the fidelity at the
        cost of speed.

    Returns
    -------
    fr : array-like
        The firing rate in Hz.
    times : array-like
        The times associated to the firing rate values.
    '''
    if resolution is None:
        resolution = 0.1*kernel_std
    bin_std = kernel_std / float(resolution)
    kernel_size = 2. * cut_gaussian * bin_std
    # generate the times
    delta_T = resolution * 0.5 * kernel_size
    times = np.arange(np.min(spike_times) - delta_T,
                      np.max(spike_times) + delta_T, resolution)
    rate = np.zeros(len(times))
    # counts the spikes at each time
    pos = find_idx_nearest(times, spike_times)
    bins = np.linspace(0, len(times), len(times)+1)
    counts, _ = np.histogram(pos, bins=bins)
    # initialize with delta rate in Hz
    rate += 1000. * counts / (kernel_std*np.sqrt(np.pi))
    fr = _smooth(rate, kernel_size, bin_std, mode='same')
    # translate times
    times += kernel_center
    return fr, times


# ------------------------- #
# Analyse bursting activity #
# ------------------------- #

def activity_types(spike_detector, limits, network=None,
                   phase_coeff=(0.5, 10.), mbis=0.5, mfb=0.2, mflb=0.05,
                   skip_bursts=0, simplify=False, fignums=None, show=False):
    '''
    Analyze the spiking pattern of a neural network.
    .. todo ::
        think about inserting t=0. and t=simtime at the beginning and at the
        end of ``times''.

    Parameters
    ----------
    spike_detector : NEST node(s), (tuple or list of tuples)
        The recording device that monitored the network's spikes
    limits : tuple of floats
        Time limits of the simulation regrion which should be studied (in ms).
    network : :class:`nngt.Network`, optional (default: None)
        Neural network that was analyzed
    phase_coeff : tuple of floats, optional (default: (0.2, 5.))
        A phase is considered `bursting' when the interspike between all spikes
        that compose it is smaller than ``phase_coeff[0] / avg_rate`` (where
        ``avg_rate`` is the average firing rate), `quiescent' when it is
        greater that ``phase_coeff[1] / avg_rate``, `mixed' otherwise.
    mbis : float, optional (default: 0.5)
        Maximum interspike interval allowed for two spikes to be considered in
        the same burst (in ms).
    mfb : float, optional (default: 0.2)
        Minimal fraction of the neurons that should participate for a burst to
        be validated (i.e. if the interspike is smaller that the limit BUT the
        number of participating neurons is too small, the phase will be
        considered as `localized`).
    mflb : float, optional (default: 0.05)
        Minimal fraction of the neurons that should participate for a local
        burst to be validated (i.e. if the interspike is smaller that the limit
        BUT the number of participating neurons is too small, the phase will be
        considered as `mixed`).
    skip_bursts : int, optional (default: 0)
        Skip the `skip_bursts` first bursts to consider only the permanent
        regime.
    simplify: bool, optional (default: False)
        If ``True``, `mixed` phases that are contiguous to a burst are
        incorporated to it.
    return_steps : bool, optional (default: False)
        If ``True``, a second dictionary, `phases_steps` will also be returned.
        @todo: not implemented yet
    fignums : list, optional (default: [])
        Indices of figures on which the periods can be drawn.
    show : bool, optional (default: False)
        Whether the figures should be displayed.

    .. note :
        Effects of `skip_bursts` and `limits[0]` are cumulative: the 
        `limits[0]` first milliseconds are ignored, then the `skip_bursts`
        first bursts of the remaining activity are ignored.

    Returns
    -------
    phases : dict
        Dictionary containing the time intervals (in ms) for all four phases
        (`bursting', `quiescent', `mixed', and `localized`) as lists.
        E.g: ``phases["network_burst"]`` could give ``[[123.5,334.2],
        [857.1,1000.6]]``.
    '''
    import nest
    if fignums is None:
        fignums = []
    # check if there are several recorders
    senders, times = [], []
    if True in nest.GetStatus(spike_detector, "to_file"):
        for fpath in nest.GetStatus(spike_detector, "record_to"):
            data = _get_data(fpath)
            times.extend(data[:, 1])
            senders.extend(data[:, 0])
    else:
        for events in nest.GetStatus(spike_detector, "events"):
            times.extend(events["times"])
            senders.extend(events["senders"])
        idx_sort = np.argsort(times)
        times = np.array(times)[idx_sort]
        senders = np.array(senders)[idx_sort]
    # compute phases and properties
    data = np.array((senders, times))
    phases, fr = _analysis(times, senders, limits, network=network,
              phase_coeff=phase_coeff, mbis=mbis, mfb=mfb, mflb=mflb,
              simplify=simplify)
    properties = _compute_properties(data.T, phases, fr, skip_bursts)
    kwargs = {
        "limits": limits,
        "phase_coeff": phase_coeff,
        "mbis": mbis,
        "mfb": mfb,
        "mflb": mflb,
        "simplify": simplify
    }
    # plot if required
    if show:
        _plot_phases(phases, fignums)
    datadict = {'neuron': data[0], 'time': data[1]}
    return ActivityRecord(datadict, phases, properties=properties)


def analyze_raster(raster, limits=None, network=None,
                   phase_coeff=(0.5, 10.), mbis=0.5, mfb=0.2, mflb=0.05,
                   skip_bursts=0, skip_ms=0., simplify=False, fignums=None,
                   show=False):
    '''
    Return the activity types for a given raster.

    Parameters
    ----------
    raster : array-like or str
        Either an array containing the ids of the spiking neurons and the
        corresponding time, or the path to a NEST .gdf recording.
    limits : tuple of floats
        Time limits of the simulation region which should be studied (in ms).
    network : :class:`nngt.Network`, optional (default: None)
        Network on which the recorded activity was simulated.
    phase_coeff : tuple of floats, optional (default: (0.2, 5.))
        A phase is considered `bursting' when the interspike between all spikes
        that compose it is smaller than ``phase_coeff[0] / avg_rate`` (where
        ``avg_rate`` is the average firing rate), `quiescent' when it is
        greater that ``phase_coeff[1] / avg_rate``, `mixed' otherwise.
    mbis : float, optional (default: 0.5)
        Maximum interspike interval allowed for two spikes to be considered in
        the same burst (in ms).
    mfb : float, optional (default: 0.2)
        Minimal fraction of the neurons that should participate for a burst to
        be validated (i.e. if the interspike is smaller that the limit BUT the
        number of participating neurons is too small, the phase will be
        considered as `localized`).
    mflb : float, optional (default: 0.05)
        Minimal fraction of the neurons that should participate for a local
        burst to be validated (i.e. if the interspike is smaller that the limit
        BUT the number of participating neurons is too small, the phase will be
        considered as `mixed`).
    skip_bursts : int, optional (default: 0)
        Skip the `skip_bursts` first bursts to consider only the permanent
        regime.
    simplify: bool, optional (default: False)
        If ``True``, `mixed` phases that are contiguous to a burst are
        incorporated to it.
    fignums : list, optional (default: [])
        Indices of figures on which the periods can be drawn.
    show : bool, optional (default: False)
        Whether the figures should be displayed.

    .. note :
        Effects of `skip_bursts` and `limits[0]` are cumulative: the
        `limits[0]` first milliseconds are ignored, then the `skip_bursts`
        first bursts of the remaining activity are ignored.

    Returns
    -------
    activity : ActivityRecord
        Object containing the phases and the properties of the activity
        from these phases.
    '''
    data = _get_data(raster) if isinstance(raster, str) else raster
    if limits is None:
        limits = [np.min(data[:, 1]), np.max(data[:, 1])]
    if fignums is None:
        fignums = []
    kwargs = {
        "limits": limits,
        "phase_coeff": phase_coeff,
        "mbis": mbis,
        "mfb": mfb,
        "mflb": mflb,
        "simplify": simplify
    }
    # compute phases and properties
    phases, fr = _analysis(data[:, 1], data[:, 0], limits, network=network,
              phase_coeff=phase_coeff, mbis=mbis, mfb=mfb, mflb=mflb,
              simplify=simplify)
    properties = _compute_properties(data.T, phases, fr, skip_bursts)
    # plot if required
    if show:
        import matplotlib.pyplot as plt
        if fignums:
            _plot_phases(phases, fignums)
        else:
            fig, ax = plt.subplots()
            ax.scatter(data[:, 1], data[:, 0])
            _plot_phases(phases, [fig.number])
    return ActivityRecord(data, phases, properties, kwargs)


# ------ #
# Tools  #
# ------ #

def neurons_sorter(neurons, sort, data=None, network=None):
    '''
    Sort the neurons according to the `sort` property.

    If `sort` is "spikes", "firing_rate" or "B2", then data must contain the
    `senders` and `times` as a (M, 2) array.

    Parameters
    ----------

    neurons : array-like
        Ids of the neurons that should be sorted.
    sort : str
        Property according to which the neurons should be sorted: it will be
        used to attribute a new index to each neuron. If no `network` was
        provided, it can be either the neurons' 'spikes'-rank (difference
        between the number of spike outside and inside bursts, i.e. number of
        spikes if there is no bursting activity), their "firing_rate" or their
        "B2" coefficient.
        If `network` was provided, can be among topological property
        ("in-degree", "out-degree", "total-degree" or "betweenness"), or a
        user-defined list of sorted neuron ids.
        Sorting is performed by increasing value of the `sort` property from
        bottom to top inside each group.
    data : (M, 2) array, optional (default: None)
        Senders and spike times, required for "spikes", "firing_rate" and "B2".
        First row is senders, second is times.
    network : :class:`~nngt.Network` or subclass, optional (default: None)
        Network for which the activity was monitored. Required for topological
        sorting.

    Returns
    -------
    For N neurons, labeled from ``ID_MIN`` to ``ID_MAX``, returns a`sorting`
    array of size ``ID_MAX``, where ``sorting[ids]`` gives the sorted ids of
    the neurons, i.e. an integer between 1 and N.
    '''
    neurons = np.unique(neurons).astype(int)
    num_neurons = len(neurons)
    min_id = np.min(neurons)
    max_id = np.max(neurons)
    sorting = np.zeros(max_id + 1)
    attribute = None
    if isinstance(sort, str):
        sorted_ids = None
        if sort == "firing_rate":
            # compute number of spikes per neuron
            spikes = np.bincount(data[:, 0].astype(int))
            if spikes.shape[0] < max_id:
                spikes.resize(max_id)  # we need one entry per neuron
            # sort them (neuron with least spikes arrives at min_id)
            sorted_ids = np.argsort(spikes)
        elif sort in ("b2", "B2"):
            attribute = get_b2(spike_times=data[:, 1], senders=data[:, 0])
            sorted_ids = np.argsort(attribute)
            # check for non-spiking neurons
            num_b2 = attribute.shape[0]
            if num_b2 < num_neurons:
                spikes = np.bincount(data[:, 0])
                non_spiking = np.where(spikes[min_id] == 0)[0]
                sorted_ids.resize(num_neurons)
                for i, n in enumerate(non_spiking):
                    sorted_ids[sorted_ids >= n] += 1
                    sorted_ids[num_b2 + i] = n
        else:
            from nngt.analysis import node_attributes
            attribute = node_attributes(network, sort)
            sorted_ids = np.argsort(attribute)
        num_sorted = 1
        if network is not None:
            from nngt.lib.sorting import _sort_groups
            _, sorted_groups = _sort_groups(network.population)
            for group in sorted_groups:
                gids = neurons[group.id_list]
                order = np.argsort(sorted_ids[group.id_list])
                sorting[gids] = num_sorted + order
                num_sorted += len(group.id_list)
        else:
            sorting = np.argsort(sorted_ids)
    else:
        sorting[neurons] = np.argsort(sort)
    return sorting.astype(int)


def read_pickle(filename):
    '''
    Read a pickled :class:`PyNeurActiv.analysis.ActivityRecord` and return it.
    '''
    with open(filename, 'r') as f:
        return pickle.load(f)


def _get_data(source):
    '''
    Returns the (times, senders) array.

    Parameters
    ----------
    source : list or str
        Indices of spike detectors or path to the .gdf files.
    
    Returns
    -------
    data : 2D array of shape (N, 2)
    '''
    data = [[],[]]
    is_string = isinstance(source, str)
    if is_string:
        source = [source]
    elif nonstring_container(source) and isinstance(source[0], str):
        is_string = True
    if is_string:
        for path in source:
            tmp = np.loadtxt(path)
            data[0].extend(tmp[:, 0])
            data[1].extend(tmp[:, 1])
    elif nonstring_container(source) and np.array(source).ndim == 2:
        source = np.array(source)
        assert 2 in source.shape, 'Invalid `source`; enter a string, a ' +\
                                  'list of str, a (N, 2) or (2, N) array,' +\
                                  'or a list of NEST-recorder gids.'
        if source.shape[0] == 2:
            return source.T
        return source
    else:
        import nest
        events = nest.GetStatus(source, "events")
        for ev in events:
            data[0].extend(ev["senders"])
            data[1].extend(ev["times"])
    data = np.array(data).T
    idx_sort = np.argsort(data[:, 1])
    return data[idx_sort, :]


def _find_phases(times, phases, lim_burst, lim_quiet, simplify):
    '''
    Find the time limits of the different phases.
    '''
    diff = np.diff(times).tolist()[::-1]
    i = 0
    previous = { "network_burst": -2, "mixed": -2, "quiescent": -2 }
    while diff:
        tau = diff.pop()
        while True:
            if tau < lim_burst: # bursting phase
                if previous["network_burst"] == i-1:
                    phases["network_burst"][-1][1] = times[i+1]
                else:
                    if simplify and previous["mixed"] == i-1:
                        start_mixed = phases["mixed"][-1][0]
                        phases["network_burst"].append(
                            [start_mixed, times[i+1]])
                        del phases["mixed"][-1]
                    else:
                        phases["network_burst"].append([times[i], times[i+1]])
                previous["network_burst"] = i
                i+=1
                break
            elif tau > lim_quiet:
                if previous["quiescent"] == i-1:
                    phases["quiescent"][-1][1] = times[i+1]
                else:
                    phases["quiescent"].append([times[i], times[i+1]])
                previous["quiescent"] = i
                i+=1
                break
            else:
                if previous["mixed"] == i-1:
                    phases["mixed"][-1][1] = times[i+1]
                    previous["mixed"] = i
                else:
                    if simplify and previous["network_burst"] == i-1:
                        phases["network_burst"][-1][1] = times[i+1]
                        previous["network_burst"] = i
                    else:
                        phases["mixed"].append([times[i], times[i+1]])
                        previous["mixed"] = i
                i+=1
                break


def _check_burst_size(phases, senders, times, network, mflb, mfb):
    '''
    Check that bursting periods involve at least a fraction mfb of the neurons.
    '''
    transfer, destination = [], {}
    n = len(set(senders)) if network is None else network.node_nb()
    for i,burst in enumerate(phases["network_burst"]):
        idx_start = np.where(times==burst[0])[0][0]
        idx_end = np.where(times==burst[1])[0][0]
        participating_frac = len(set(senders[idx_start:idx_end])) / float(n)
        if participating_frac < mflb:
            transfer.append(i)
            destination[i] = "mixed"
        elif participating_frac < mfb:
            transfer.append(i)
            destination[i] = "local_burst"
    for i in transfer[::-1]:
        phase = phases["network_burst"].pop(i)
        phases[destination[i]].insert(0, phase)
    remove = []
    i = 0
    while i < len(phases['mixed']):
        mixed = phases['mixed'][i]
        j=i+1
        for span in phases['mixed'][i+1:]:
            if span[0] == mixed[1]:
                mixed[1] = span[1]
                remove.append(j)
            elif span[1] == mixed[0]:
                mixed[0] = span[0]
                remove.append(j)
            j+=1
        i+=1
    remove = list(set(remove))
    remove.sort()
    for i in remove[::-1]:
        del phases["mixed"][i]


def _analysis(times, senders, limits, network=None,
              phase_coeff=(0.5, 10.), mbis=0.5, mfb=0.2, mflb=0.05,
              simplify=False):
    # prepare the phases and check the validity of the data
    phases = {
        "network_burst": [],
        "mixed": [],
        "quiescent": [],
        "localized_burst": []
    }
    num_spikes, avg_rate = len(times), 0.
    if num_spikes:
        num_neurons = (len(np.unique(senders)) if network is None
                       else network.node_nb())
        # set the studied region
        if limits[0] >= times[0]:
            idx_start = np.where(times >= limits[0])[0][0]
            times = times[idx_start:]
            senders = senders[idx_start:]
        if limits[1] <= times[-1]:
            idx_end = np.where(times <= limits[1])[0][-1]
            times = times[:idx_end]
            senders = senders[:idx_end]
        # get the average firing rate to differenciate the phases
        simtime = limits[1] - limits[0]
        lim_burst, lim_quiet = 0., 0.
        avg_rate = num_spikes / float(simtime)
        lim_burst = max(phase_coeff[0] / avg_rate, mbis)
        lim_quiet = min(phase_coeff[1] / avg_rate, 10.)
        # find the phases
        _find_phases(times, phases, lim_burst, lim_quiet, simplify)
        _check_burst_size(phases, senders, times, network, mflb, mfb)
        avg_rate *= 1000. / float(num_neurons)
    return phases, avg_rate


def _compute_properties(data, phases, fr, skip_bursts):
    '''
    Compute the properties from the spike times and phases.

    Parameters
    ----------
    data : 2D array, shape (N, 2)
        Spike times and senders.
    phases : dict
        The phases.
    fr : double
        Firing rate.

    Returns
    -------
    prop : dict
        Properties of the activity. Contains the following pairs:

        - "firing_rate": average value in Hz for 1 neuron in the network.
        - "network_bursting": True if there were bursts of activity detected.
        - "burst_duration", "ISI", and "IBI" in ms, if "network_bursting" is
          True.
        - "SpB": average number of spikes per burst for one neuron.
    '''
    prop = {}
    times = data[:, 1]
    # firing rate (in Hz, normalized for 1 neuron)
    prop["firing_rate"] = fr
    num_bursts = len(phases["network_burst"])
    prop["num_network_bursts"] = num_bursts
    init_val = 0. if num_bursts > skip_bursts else np.NaN
    if num_bursts:
        prop["network_bursting"] = True
        prop.update({
            "burst_durations": [],
            "IBIs": [],
            "ISIs": [],
            "SpBs": [],
            "participations": [],
            "burst_duration_avg": init_val,
            "IBI_avg": init_val,
            "ISI_avg": init_val,
            "SpB_avg": init_val,
            "participation_avg": init_val,
            "period_avg": init_val,
            "burst_duration_std": init_val,
            "IBI_std": init_val,
            "ISI_std": init_val,
            "SpB_std": init_val,
            "participation_std": init_val,
            "period_std": init_val})
    else:
        prop["network_bursting"] = False
    for i, burst in enumerate(phases["network_burst"]):
        if i >= skip_bursts:
            # burst_duration
            prop["burst_durations"].append(burst[1] - burst[0])
            # IBI
            if i > 0:
                end_older_burst = phases["network_burst"][i-1][1]
                prop["IBIs"].append(burst[0]-end_older_burst)
            # get num_spikes inside the burst, divide by num_neurons
            idxs = np.where((times >= burst[0]) & (times <= burst[1]))[0]
            num_spikes = len(times[idxs])
            num_neurons = len(set(data[:, 0][idxs]))
            prop["SpBs"].append(num_spikes / float(num_neurons))
            prop["participations"].append(num_neurons)
            # ISI
            if prop["SpBs"][-1] >= 2:
                prop["ISIs"].append(
                    num_neurons * (burst[1] - burst[0]) / float(num_spikes))
            else:
                prop["ISIs"].append(np.NaN)
    # averaging
    no_avg = ("network_bursting", "firing_rate", "num_network_bursts")
    for key, val in prop.items():
        if nonstring_container(val):
            prop[key[:-1] + '_avg'] = np.average(val)
            prop[key[:-1] + '_std'] = np.std(val)
    if num_bursts > skip_bursts:
        prop["period_avg"] = prop["IBI_avg"] + prop["burst_duration_avg"]
        if num_bursts > skip_bursts + 1:
            prop["period_std"] = np.std(
                prop["IBIs"] + prop["burst_durations"][1:])
    return prop


def _plot_phases(phases, fignums):
    import matplotlib.pyplot as plt
    colors = ('r', 'orange', 'g', 'b')
    names = ('network_burst', 'mixed', 'localized', 'quiescent')
    for fignum in fignums:
        fig = plt.figure(fignum)
        for ax in fig.axes:
            for phase, color in zip(names, colors):
                for span in phases[phase]:
                    ax.axvspan(span[0], span[1], facecolor=color,
                               alpha=0.2)
    plt.show()


def _pos_first_spike_burst(in_a_burst):
    '''
    Returns the index of the first spike of each burst in `isi`.

    .. warning::
        This function expects bursts to be present (nonzero values must exist
        in `in_a_burst`).
    '''
    # positions of interspikes in a burst
    bursting = np.nonzero(in_a_burst)[0]
    # first the interburst in the bursting array: it is the position for which
    # the ISI are not contiguous (index jump > 1)
    diff_b = np.diff(bursting)
    pos_ibi = np.concatenate(([-1], np.nonzero(diff_b > 1)[0]))
    # recover the spike position from the location of the interburst in
    # `bursting`; since it contains the ISI indices, get next index.
    return bursting[pos_ibi + 1]


def _set_burst_num(pos_first_spikes_burst, in_a_burst, spike_pos, burst,
                   interburst):
    '''
    Set the value of the burst associated to each spike.
    `spike_pos` is reauired to convert the spike indices to their absolute
    index if `collective` is False.
    The function fills the `burst` and `interburst` arrays.
    '''
    last_idx = len(in_a_burst)  # gives last spike_pos entry
    i, nsb, nsi, start = 0, 0, 0, 0

    for i, idx in enumerate(chain(pos_first_spikes_burst, [last_idx])):
        ''' Test with numpy
        # get the indices of the spikes in a burst that are before
        # the current interburst and after the previous one
        in_burst_i = in_a_burst[start:idx]
        spks_burst_i = np.nonzero(in_burst_i)[0] + start
        num_in_burst = len(spks_burst_i) + 1
        num_in_interburst = idx + 1 - start - num_in_burst

        # The first spike for which the interspike registers out is still in
        # the burst, so we add it again
        # For the last region, there is one last spike after the last
        # interspike: either in_burst[-1] is True and it is in the burst, or
        # it is False and it is in the interburst
        add_first = (len(spks_burst_i) and idx != last_idx
                     and spks_burst_i[-1] - start < len(in_burst_i) - 1)
        add_last = idx == last_idx and num_in_burst
        if add_first:
            in_burst_i[spks_burst_i[-1] + 1 - start] = True
            num_in_burst += 1
            num_in_interburst -= 1
        elif add_last:
            num_in_burst += 1
        else:
            num_in_burst -= 1

        # create the containers with the right length
        spks_burst_i = np.zeros(num_in_burst, dtype=int)
        spks_interburst_i = np.zeros(num_in_interburst, dtype=int)

        # fill them
        spks_burst_i[:num_in_burst - 1] = np.nonzero(in_burst_i)[0] + start
        spks_interburst_i[:num_in_interburst] = \
            np.nonzero(~in_burst_i)[0] + start

        if num_in_burst - 1 and add_first:
            spks_burst_i[-1] = len(in_burst_i) + start
        elif add_last:
            spks_burst_i[-1] = spks_burst_i[-2] + 1
        if idx == last_idx and num_in_interburst:
            spks_interburst_i[-1] = len(in_burst_i) + start
        '''

        # get the indices of the spikes in a burst that are before
        # the current interburst and after the previous one
        in_burst_i = in_a_burst[start:idx]
        spks_burst_i = np.nonzero(in_burst_i)[0] + start

        # the first spike for which the interspike registers out is still in
        # the burst, so we add it again
        add_first = (len(spks_burst_i) and idx != last_idx
                     and spks_burst_i[-1] - start < len(in_burst_i) - 1)
        if add_first:
            in_burst_i[spks_burst_i[-1] + 1 - start] = True
            spks_burst_i = list(spks_burst_i) + [spks_burst_i[-1] + 1]
        spks_interburst_i = np.nonzero(~in_burst_i)[0] + start

        assert len(in_burst_i) == len(spks_burst_i) + len(spks_interburst_i)

        # for the last region, there is one last spike after the last
        # interspike: either in_burst[-1] is True and it is in the burst, or
        # it is False and it is in the interburst
        if idx == last_idx:
            if in_burst_i[-1]:
                spks_burst_i = list(spks_burst_i) + [len(in_burst_i) + start]
            elif np.any(in_burst_i):
                spks_burst_i = list(spks_burst_i) + [spks_burst_i[-1] + 1]
            else:
                spks_interburst_i = spks_interburst_i.tolist() +\
                                    [len(in_burst_i) + start]

        # set burst indices
        burst[spike_pos[spks_burst_i]] = i + 1

        # get the indices of the spikes that are in the interburst
        interburst[spike_pos[spks_interburst_i]] = i + 1

        # update spikes counts and start index
        nsb += len(spks_burst_i)
        nsi += len(spks_interburst_i)
        start = idx

    return nsb, nsi
