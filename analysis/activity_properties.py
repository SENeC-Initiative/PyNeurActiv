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

import weakref
import logging
from itertools import chain

import numpy as np
import scipy.sparse as ssp
from scipy.interpolate import interp1d
from scipy.signal import argrelmin, argrelmax

from .array_searching import find_idx_nearest
from .bayesian_blocks import bayesian_blocks
from ..lib.signal_processing import _smooth
from ..lib import nonstring_container


__all__ = [
    "ActivityRecord",
    "activity_types",
    "analyze_raster",
    "data_from_nest",
    "firing_rate",
    "raster_analysis",
    "get_spikes",
    "interburst_properties",
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
                 properties=None, parameters=None, sort=None, network=None):
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

        Note
        ----
        The firing rate is computed as num_spikes / total simulation time, the
        period is the sum of an IBI and a bursting period.
        '''
        super(ActivityRecord, self).__init__(datadict)
        self._data_keys = spikes_data
        self._parameters = parameters
        self._phases = None if phases is None else properties.copy()
        self._properties = None if properties is None else properties.copy()
        self._sort = sort
        self.network = None if network is None else weakref.ref(network)
        self._compute_properties()

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

        - "bursting" for periods of high activity where a large fraction of the
          network is recruited.
        - "quiescent" for periods of low activity
        - "mixed" for firing rate in between "quiescent" and "bursting".
        - "localized" for periods of high activity but where only a small
          fraction of the network is recruited.
        - "unknown" for uncharacterized phases.

        Note
        ----
        Phases that are not present in the activity are not added to the dict.
        See `parameters` for details on the conditions used to differenciate
        these phases.
        '''
        return self._phases

    @property
    def properties(self):
        '''
        Returns the properties of the activity.
        Contains the following entries:

        - "firing_rate": average value in Hz for 1 neuron in the network.
        - "bursting": True if there were bursts of activity detected.
        - "burst_duration", "IBI", "ISI", and "period" in ms, if "bursting".
        - "SpB" (Spikes per Burst): average number of spikes per neuron
          during a burst, if "bursting".
        '''
        return self._properties

    def simplify():
        raise NotImplementedError("Will hopefully be implemented one day.")

    def neuron_ranking(self, neurons, sort='spikes'):
        '''
        Sort the neurons according to `sort`, which can be either they
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

    def _compute_properties(self):
        pass
        #~ if self._phases is None:
            #~ pass
        #~ if self._properties is None:
            #~ _, time = self._data_keys
            #~ fr = firing_rate(self._data[time])
            #~ self._properties = _compute_properties(
                #~ self._data, self._phases, fr, skip_bursts=0)


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
    import nest
    # get spikes
    skip = 0. if skip is None else skip
    if recorder is not None:
        data = nest.GetStatus(recorder[0])[0]["events"]
        spike_times = data["times"]
        senders = (data["senders"] if network is None
                   else network.id_from_nest_gid(data["senders"]))
    elif spike_times is None and senders is None:
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

def raster_analysis(raster, limits=None, network=None, skip_bursts=0,
                    bins='bayesian', smooth=True, num_steps=1000, axis=None,
                    show=False, sender='neuron', time='time', **kwargs):
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

    #~ fr = firing_rate(activity[time], kernel_std=30.)
    #~ peaks = argrelmax(fr)
    #~ max_fr = np.max(fr[peaks])
    #~ peaks = [i for i in peaks if fr[i] > 0.2 * max_fr]
    #~ num_bursts = len(peaks)
    #~ bursts = np.argwhere(fr > 0.2 * max_fr)

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

    #~ plt.plot(bins, counts)
    #~ plt.show()

    # maxima (T_min, ..., T_max) of the histogram
    local_max = argrelmax(counts)[0]
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
            pos_first_spikes_burst = _pos_first_spike_burst(in_a_burst)
            # assign each spike to the burst where it belongs and count the
            # number of spikes inside bursts and inside interbursts
            nsb, nsi = _set_burst_num(pos_first_spikes_burst, in_a_burst,
                                      spike_pos, burst, interburst)
            spks_loc['spks_in_bursts'][i] = nsb
            spks_loc['spks_in_interbursts'][i] = nsi
            i += 1
        if np.any(~np.isnan(burst)):
            activity['burst'] = burst
            activity['interburst'] = interburst
    elif len(local_max) > 2:
        logger.warning("Complex activity detected, manual processing will be "
                       "necessary.")

    # neuron sorting
    sorter = np.arange(0, np.max(spks_loc['neuron']) + 1, dtype=int)
    if 'burst' in activity:
        rank = spks_loc['spks_in_interbursts'] - spks_loc['spks_in_bursts']
        asort_rank = np.argsort(rank)
        sorter[spks_loc['neuron']] = np.argsort(asort_rank)

    return ActivityRecord(activity, spikes_data=(sender, time), sort=sorter)


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
    delta_T = resolution * 0.5 * kernel_size,
    times = np.arange(-1. * delta_T, np.max(spike_times) + delta_T, resolution)
    rate = np.zeros(int(np.max(spike_times) / resolution))
    # initialize with delta rate in Hz
    rate[find_idx_nearest(times, spike_times)] += \
        1000. / (kernel_std*np.sqrt(np.pi))
    fr = _smooth(rate, kernel_size, bin_std, mode='full')
    # translate times
    times += kernel_center
    if len(times) > len(fr):
        times = times[:-1]
    elif len(times) < len(fr):
        fr = fr[:-1]
    if not len(times) == len(fr):
        raise RuntimeError("Internal error, please file an issue on the"
                           "GitHub issue tracker, including the versions of "
                           "Python and pna you are using, as well as a "
                           "minimal working example.")
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
        E.g: ``phases["bursting"]`` could give ``[[123.5,334.2],
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
    properties = _compute_properties(data, phases, fr, skip_bursts)
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
    return ActivityRecord(data, phases, properties, kwargs)


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
    previous = { "bursting": -2, "mixed": -2, "quiescent": -2 }
    while diff:
        tau = diff.pop()
        while True:
            if tau < lim_burst: # bursting phase
                if previous["bursting"] == i-1:
                    phases["bursting"][-1][1] = times[i+1]
                else:
                    if simplify and previous["mixed"] == i-1:
                        start_mixed = phases["mixed"][-1][0]
                        phases["bursting"].append([start_mixed, times[i+1]])
                        del phases["mixed"][-1]
                    else:
                        phases["bursting"].append([times[i], times[i+1]])
                previous["bursting"] = i
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
                    if simplify and previous["bursting"] == i-1:
                        phases["bursting"][-1][1] = times[i+1]
                        previous["bursting"] = i
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
    for i,burst in enumerate(phases["bursting"]):
        idx_start = np.where(times==burst[0])[0][0]
        idx_end = np.where(times==burst[1])[0][0]
        participating_frac = len(set(senders[idx_start:idx_end])) / float(n)
        if participating_frac < mflb:
            transfer.append(i)
            destination[i] = "mixed"
        elif participating_frac < mfb:
            transfer.append(i)
            destination[i] = "localized"
    for i in transfer[::-1]:
        phase = phases["bursting"].pop(i)
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
        "bursting": [],
        "mixed": [],
        "quiescent": [],
        "localized": []
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
        - "bursting": True if there were bursts of activity detected.
        - "burst_duration", "ISI", and "IBI" in ms, if "bursting" is True.
        - "SpB": average number of spikes per burst for one neuron.
    '''
    prop = {}
    times = data[1, :]
    # firing rate (in Hz, normalized for 1 neuron)
    prop["firing_rate"] = fr
    num_bursts = len(phases["bursting"])
    init_val = 0. if num_bursts > skip_bursts else np.NaN
    if num_bursts:
        prop["bursting"] = True
        prop.update({
            "burst_duration": init_val,
            "IBI": init_val,
            "ISI": init_val,
            "SpB": init_val,
            "period": init_val})
    else:
        prop["bursting"] = False
    for i, burst in enumerate(phases["bursting"]):
        if i >= skip_bursts:
            # burst_duration
            prop["burst_duration"] += burst[1] - burst[0]
            # IBI
            if i > 0:
                end_older_burst = phases["bursting"][i-1][1]
                prop["IBI"] += burst[0]-end_older_burst
            # get num_spikes inside the burst, divide by num_neurons
            idxs = np.where((times >= burst[0])*(times <= burst[1]))[0]
            num_spikes = len(times[idxs])
            num_neurons = len(set(data[0, :][idxs]))
            prop["SpB"] += num_spikes / float(num_neurons)
            # ISI
            prop["ISI"] += num_neurons * (burst[1] - burst[0])\
                           / float(num_spikes)
    for key in iter(prop.keys()):
        if key not in ("bursting", "firing_rate") and num_bursts > skip_bursts:
            prop[key] /= float(num_bursts - skip_bursts)
    if num_bursts > skip_bursts:
        prop["period"] = prop["IBI"] + prop["burst_duration"]
    if num_bursts and prop["SpB"] < 2.:
        prop["ISI"] = np.NaN
    return prop


def _plot_phases(phases, fignums):
    import matplotlib.pyplot as plt
    colors = ('r', 'orange', 'g', 'b')
    names = ('bursting', 'mixed', 'localized', 'quiescent')
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
