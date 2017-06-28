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

""" Tools to monitor the activity of neurons """

import sys
sys.argv.append('--quiet')
import weakref

import numpy as np
import scipy.signal as sps

import nest

from .activity_properties import firing_rate
from .array_searching import find_idx_nearest


__all__ = ["Recorder"]


# ------------ #
# NNGT groups  #
# ------------ #

try:
    import nngt
    with_nngt = True
except:
    with_nngt = False


# --------------- #
# Recorder class  #
# --------------- #

class Recorder:

    '''
    Intelligent recording device which allows to monitor localized areas.
    
    To get the results, call the :func:`get_recording` function. It will return
    a dictionary with the following form: ::
    
        {   id0: {
                variable0: np.array([...]),
                variable1: np.array([...]),
                "times": np.array([...])
            },
            id1: {
                variable0: np.array([...]),
                variable1: np.array([...]),
                "times": np.array([...])
            },
            ...
        }
    
    Where:
    
    * idX is the identifier of the object on which the recording was performed
      (e.g. the ID of a neuron, or the center of mass (x, y) of an area).
    * variableY is the name of the recorded variable (e.g. "V_m", "spikes"...)
    * "times" is present as soon as a state variable (anythinng but "spikes")
      is recorded, as contains the associated times at which the variable
      values were recorded.
    '''

    @classmethod
    def coarse_grained(cls, record_from, spatial_network, rect, params=None):
        '''
        Creates a grid-based recording where the recorders will integrate the
        signal over an area described by `rect`.
        
        Parameters
        ----------
        record_from : str or list
            Variables that should be recorded, among "spikes", "V_m", or any
            other state parameter allowed by the NEST neuron model.
        spatial_network : :class:`nngt.SpatialNetwork`
            Network containing the neurons and their positions.
        rect : tuple, or :class:`PyNCulture.Shape` object (or in
            ``nngt.geometry``)
            Area that will be used to pave space, and over which the recording
            will be averaged. A tuple of the form (h, w) can also be passed, in
            which case the first entry is the height of the rectangle `rect`,
            and the second is its width.
        params : dict, optional (default: None)
            Additional arameters for the NEST recorder.
        '''
        from shapely.geometry import Point, Polygon
        from shapely.affinity import translate
        assert spatial_network.nest_gid is not None, \
            "Use :func:`~nngt.Network.to_nest` on the network beforehand."
        cg_recorder = cls(
            [], record_from, params=params, network=spatial_network)
        # make the polygon
        if not isinstance(Polygon, rect):
            h, w = rect
            rect = Polygon([(-0.5*w, -0.5*h), (-0.5*w, 0.5*h),
                            (0.5*w, 0.5*h), (0.5*w, -0.5*h)])
        # compute the grid
        xmin, ymin, xmax, ymax = spatial_network.shape.bounds
        s_xmin, s_ymin, s_xmax, s_ymax = rect.bounds
        h, w = (ymax - ymin), (xmax - xmin)
        s_h, s_w = (s_ymax - s_ymin), (s_xmax - s_xmin)
        lines, cols = int(np.ceil(h / s_h)), int(np.ceil(w / s_w))
        border_w = w - s_w*(cols - 1)
        border_h = h - s_h*(lines - 1)
        centroids = np.zeros((lines, cols, 2))
        for i in range(lines):
            centroids[i, :, 0] = np.linspace(
                xmin + 0.5*border_w, xmax - 0.5*border_w, cols)
            centroids[i, :, 1] = np.full(ymin + 0.5*border_w + i*s_h, cols)
        # generate the shape centered around zero
        xshift, yshift = np.array(rect.centroid)
        xx, yy = p.exterior.xy
        exterior = [(x - xshift, y - yshift) for x, y in zip(xx, yy)]
        base_shape = Polygon(exterior)
        groups = []
        for line in centroids:
            for pos in line:
                s = translate(base_shape, pos[0], pos[1])
                r = cls.localized(record_from, spatial_network, s,
                                  params=params, average=False,
                                  rid=tuple(pos))
                groups.extend(r.groups)
                cg_recorder._recorders.append(r)
        cg_recorder._groups = groups
        return cg_recorder

    @classmethod
    def localized(cls, record_from, spatial_network, shape, params=None,
                  average=False):
        '''
        Creates a grid-based recording where the recorders will average the
        signal over an area described by `shape`.
        
        Parameters
        ----------
        record_from : str or list
            Variables that should be recorded, among "spikes", "V_m", or any
            other state parameter allowed by the NEST neuron model.
        spatial_network : :class:`nngt.SpatialNetwork`
            Network containing the neurons and their positions.
        shape : :class:`PyNCulture.Shape` object (or in ``nngt.geometry``)
            Area where the neuronal activity will be recorded (each neuron in
            this area will be recorded separately).
        params : dict, optional (default: None)
            Additional arameters for the NEST recorder.
        '''
        from shapely.geometry import Point
        assert spatial_network.nest_gid is not None, \
            "Use :func:`~nngt.Network.to_nest` on the network beforehand."
        keep = []
        for i, pos in enumerate(spatial_network.get_positions()):
            p = Point(*pos)
            if shape.contains(p):
                keep.append(i)
        if average:
            return cls.averaged(
                keep, record_from, params=params, network=spatial_network)
        else:
            return cls(
                keep, record_from, params=params, network=spatial_network)

    @classmethod
    def averaged(cls, neurons, record_from, params=None, network=None):
        '''
        Average the signal over the recorded neurons.
        
        Parameters
        ----------
        neurons : list
            Ids of the neurons. If `network` is None, should be the NEST gids,
            otherwise it must be the NNGT network ids.
        record_from : str or list
            Variables that should be recorded, among "spikes", "V_m", or any
            other state parameter allowed by the NEST neuron model.
        spatial_network : :class:`nngt.SpatialNetwork`
            Network containing the neurons and their positions.
        params : dict, optional (default: None)
            Additional arameters for the NEST recorder.
        '''
        average_rec = cls([], record_from, params=params, network=network)
        w = 1. / len(neurons)
        average_rec._recorders.append(
            _AccumulatorRecorder(neurons, record_from, params, network=network,
                                 syn_params={'weight': w}))
        return average_rec
        

    def __init__(self, neurons, record_from, params=None, network=None):
        '''
        Create a new recorder instance to monitor neurons.
        
        Parameters
        ----------
        neurons : list
            Ids of the neurons. If `network` is None, should be the NEST gids,
            otherwise it must be the NNGT network ids.
        record_from : str or list
            Variables that should be recorded, among "spikes", "V_m", or any
            other state parameter allowed by the NEST neuron model.
        params : dict, optional (default: None)
            Additional arameters for the NEST recorder.
        network : :class:`nngt.Network`, optional (default: None)
            Network containing the neurons.
        '''
        self.neurons = neurons
        if isinstance(record_from, str):
            self.record_from = [record_from]
        else:
            self.record_from = [rec for rec in record_from]
        self.network = None if network is None else weakref.proxy(network)
        params = {} if params is None else params
        # set the recorders
        self.recorders = []
        for n in neurons:
            self.recorders.append(
                _SingleNeuronRecorder(n, record_from, params, network=network))

    def get_recording(self, smooth=False, kernel_std=30., resolution=None,
                      cut_gaussian=5., causal=0.):
        '''
        Return the recorded values.

        Parameters
        ----------
        smooth : bool, optional (default: False)
            Whether the results should be smoothed over time.
        kernel_std : float, optional (default: 30.)
            Standard deviation of the Gaussian kernel in ms.
        resolution : float, optional (default: `0.1*kernel_std`)
            The resolution at which the firing rate values will be computed.
            Choosing a value smaller than `kernel_std` is strongly advised.
        causal : float, optional (default: 0.)
            If nonzero, the smoothed signal starts after the original signal
            since it is caused by it: it is shifted in time by `causal`.

        Returns
        -------
        dict of dict of arrays (first key is the ID of the object which is
        recorded, second the name of the recorded variable).
        '''
        recordings = {}
        for recorder in self.recorders:
            recordings.update(
                recorder.get_recording(smooth, kernel_size, std, causal))
        return recordings

    def get_groups(self):
        '''
        Returns the list :class:`nngt.NeuralGroup` objects containing the
        neurons associated to a common recorder.
        '''
        groups = []
        for recorder in self.recorders:
            groups.extend(recorder.get_groups())
        return groups


# ------------------ #
# Subclass recorders #
# ------------------ #

class _SingleNeuronRecorder:

    def __init__(self, neuron, record_from, params, network=None,
                 conn_params=None, syn_params=None):
        '''
        Create a new recorder instance to monitor neurons.
        
        Parameters
        ----------
        neuron : int
            Id of the neuron. If `network` is None, should be the NEST gid,
            otherwise it must be the NNGT network id.
        record_from : list
            Variables that should be recorded, among "spikes", "V_m", or any
            other state parameter allowed by the NEST neuron model.
        params : dict
            Additional arameters for the NEST recorder.
        network : :class:`nngt.Network`, optional (default: None)
            Network containing the neuron.
        conn_params : str or dict, optional
            Specifies connectivity rule (see the `conn_spec` parameters of the
            :func:`nest.Connect` function).
        syn_params : dict, optional (default: None)
            Additional parameters describing the connection (see the `syn_spec`
            parameters of the :func:`nest.Connect` function).
        '''
        self._id = neuron
        if network is None:
            self.network = None
            self._gid = (neuron,)
        else:
             self.network = weakref.proxy(network)
             self._gid = (network.nest_gid[neuron],)
        self._recorders = {}
        self._variables = {}
        self._groups = []
        if with_nngt:
            g = nngt.core.NeuralGroup([self._id])
            if network is not None and network.nest_gid is not None:
                g._nest_gids = [network.nest_gid[neuron]]
            self._groups.append(g)
        _init_recorders(
            self._recorders, self._variables, record_from, params, False)
        _connect_recorders(self._recorders, self._gid, conn_params, syn_params)

    def get_recording(self, smooth, std, resolution, cut, causal=0.):
        recordings = {}
        for name, gid in self._recorders.items():
            data_to_get = self._variables[name]
            for d in data_to_get:
                _data_to_recording(self._id, gid, d, recordings, smooth, std,
                                   resolution, cut, causal)
        return recordings

    def get_groups(self):
        if with_nngt:
            return self._groups
        else:
            raise NotImplementedError("This function requires NNGT.")


class _AccumulatorRecorder:

    def __init__(self, neurons, record_from, params, network=None,
                 syn_params=None, rid=None):
        '''
        Create a new recorder instance to accumulate signal over several
        neurons.
        
        Parameters
        ----------
        neurons : list
            Ids of the neurons. If `network` is None, should be the NEST gids,
            otherwise it must be the NNGT network ids.
        record_from : list
            Variables that should be recorded, among "spikes", "V_m", or any
            other state parameter allowed by the NEST neuron model.
        params : dict
            Additional parameters for the NEST recorder.
        network : :class:`nngt.Network`, optional (default: None)
            Network containing the neuron.
        syn_params : dict, optional (default: None)
            Additional parameters describing the connectivity.
        rid : tuple, optional (default: None)
            ID of the recorder. If None, defaults to `neurons`.
        '''
        self._id = tuple(neurons)
        if network is None:
            self.network = None
            self._gids = neurons
        else:
             self.network = weakref.proxy(network)
             self._gids = [network.nest_gid[n] for n in neurons]
        self._recorders = {}
        self._variables = {}
        if with_nngt:
            g = nngt.core.NeuralGroup(neurons)
            if network is not None and network.nest_gid is not None:
                g._nest_gids = network.nest_gid[neurons]
            self._groups.append(g)
        _init_recorders(
            self._recorders, self._variables, record_from, params, True)
        _connect_recorders(
            self._recorders, self._gids, conn_params, syn_params)

    def get_recording(self, smooth, std, resolution, cut, causal=0.):
        recordings = {}
        for name, gid in self._recorders.items():
            data_to_get = self._variables[name]
            for d in data_to_get:
                data_to_recording(self._id, gid, d, recordings, smooth, std,
                                  resolution, cut, causal)
        return recordings

    def get_groups(self):
        if with_nngt:
            return self._groups
        else:
            raise NotImplementedError("This function requires NNGT.")


# ----- #
# Tools #
# ----- #

def _init_recorders(recorders, variables, record_from, params, to_accumulator):
    if len(record_from) == 1:
        if record_from[0] == "spikes":
            recorders["spike_detector"] = nest.Create(
                "spike_detector", params=params)
            variables["spike_detector"] = ("times",)
        elif record_from[0] == "V_m":
            vm_params = params.copy()
            vm_params["to_accumulator"] = to_accumulator
            recorders["voltmeter"] = nest.Create("voltmeter", params=params)
            variables["voltmeter"] = ("V_m",)
        else:
            mm_params = params.copy()
            mm_params["to_accumulator"] = to_accumulator
            mm_params["record_from"] = record_from
            recorders["multimeter"] = nest.Create(
                "multimeter", params=self.params)
            variables["multimeter"] = record_from
    else:
        if "spikes" in record_from:
            recorders["spike_detector"] = nest.Create(
                "spike_detector", params=params)
            variables["spike_detector"] = ("times",)
        mm_record = [r for r in record_from if r != "spikes"]
        mm_params = params.copy()
        mm_params["to_accumulator"] = to_accumulator
        mm_params["record_from"] = mm_record
        variables["multimeter"] = mm_record
        recorders["multimeter"] = nest.Create("multimeter", params=mm_params)


def _connect_recorders(recorders, neurons, conn_params, syn_params):
    for name, rec in recorders:
        if name == "spike_detector":
            nest.Connect(
                neurons, rec, conn_spec=conn_params, syn_spec=syn_params)
        else:
            nest.Connect(
                rec, neurons, conn_spec=conn_params, syn_spec=syn_params)


def _data_to_recording(rid, gid, d, rec, smooth, std, resolution, cut, causal):
    '''
    Store the data into the recording.

    Parameters
    ----------
    rid : recorder id
    gid : nest gid(s)
    d : name of the data
    rec : recordings

    see `get_recording` for the other parameters.
    '''
    data = nest.GetStatus(gid, "events")[0][d]
    times = nest.GetStatus(gid, "events")[0]["times"]
    if name == "spike_detector" and smooth:
        data, times = firing_rate(
            data, kernel_center=causal, kernel_std=std,
            resolution=resolution, cut_gaussian=cut)
        recordings[self._id]["times"] = times
    elif smooth:
        bin_std = std / float(resolution)
        kernel_size = 2. * cut_gaussian * bin_std
        data = _smooth(data, kernel_size, bin_std)
        if causal:
            times += causal
    rec[rid][d] = data
    rec[rid]["times"] = times

def _smooth_spikes(spikes, kernel_size, std):
    resol = nest.GetKernelStatus("resolution")
    times = np.arange(0., np.max(spikes), resol)
    rate = np.zeros(len(times))
    rate[find_idx_nearest(times, spikes)] += \
        1. / (std*np.sqrt(np.pi))
    return _smooth(rate, kernel_size, std)
