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

import weakref

import numpy as np

import nest


__all__ = ["Recorder"]


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
    def coarse_grained(cls, record_from, spatial_network, square, params=None):
        '''
        Creates a grid-based recording where the recorders will average the
        signal over an area described by `square`.
        
        Parameters
        ----------
        record_from : str or list
            Variables that should be recorded, among "spikes", "V_m", or any
            other state parameter allowed by the NEST neuron model.
        spatial_network : :class:`nngt.SpatialNetwork`
            Network containing the neurons and their positions.
        square : :class:`PyNCulture.Shape` object (or in ``nngt.geometry``)
            Area that will be used to pave space, and over which the recording
            will be averaged.
        params : dict, optional (default: None)
            Additional arameters for the NEST recorder.
        '''
        pass

    @classmethod
    def localized(cls, record_from, spatial_network, shape, params=None):
        '''
        Creates a grid-based recording where the recorders will average the
        signal over an area described by `square`.
        
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
        pass

    @classmethod
    def averaged(cls, neurons, record_from, spatial_network, shape,
                 params=None):
        '''
        Creates a grid-based recording where the recorders will average the
        signal over an area described by `square`.
        
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
        shape : :class:`PyNCulture.Shape` object (or in ``nngt.geometry``)
            Area over which the recording will be averaged. Only the average
            activity of the neurons in that region is recorded, not the
            individual behaviours of the neurons.
        params : dict, optional (default: None)
            Additional arameters for the NEST recorder.
        '''
        pass

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
                _SingleNeuronRecorder(n, record_from, params, network=network)

    def get_recording(self):
        recordings = {}
        for recorder in self.recorders:
            recordings.update(recorder.get_recording())
        return recordings


# --------------------------------- #
# Subclass directly mapping neurons #
# --------------------------------- #

class _SingleNeuronRecorder:

    def __init__(self, neuron, record_from, params, network=None):
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
        if len(record_from) == 1:
            if record_from[0] == "spikes":
                self._recorders["spike_detector"] = \
                    nest.Create("spike_detector", params=params))
                self._variables["spike_detector"] = ("times",)
            elif record_from[0] == "V_m" and :
                self._recorders["voltmeter"] = \
                    nest.Create("voltmeter", params=params))
                self._variables["voltmeter"] = ("V_m",)
            else:
                mm_params = params.copy()
                mm_params["record_from"] = record_from
                self._recorders["multimeter"] = \
                    nest.Create("multimeter", params=self.params))
                self._variables["multimeter"] = record_from
        else:
            if "spikes" in record_from:
                self._recorders["spike_detector"] = \
                    nest.Create("spike_detector", params=params))
                self._variables["spike_detector"] = ("times",)
            mm_record = [r for r in record_from if r != "spikes"]
            mm_params = params.copy()
            mm_params["record_from"] = mm_record
            self._variables["multimeter"] = mm_record
            self._recorders["multimeter"] = \
                nest.Create("mutlimeter", params=mm_params))

    def get_recording(self):
        recordings = {}
        for name, gid in self._recorders.items():
            data_to_get = self._variables[name]
            for d in data_to_get:
                data = nest.GetStatus(gid, "events")[0][d]
                recordings[self._id][d] = data
            if name != "spike_detector":
                data = nest.GetStatus(gid, "events")[0]["times"]
                recordings[self._id]["times"] = data
        return recordings


class _AverageRecorder:

    def __init__(self, neurons, record_from, params, network=None):
        '''
        Create a new recorder instance to monitor the average signal over
        several neurons.
        
        Parameters
        ----------
        neurons : list
            Ids of the neurons. If `network` is None, should be the NEST gids,
            otherwise it must be the NNGT network ids.
        record_from : list
            Variables that should be recorded, among "spikes", "V_m", or any
            other state parameter allowed by the NEST neuron model.
        params : dict
            Additional arameters for the NEST recorder.
        network : :class:`nngt.Network`, optional (default: None)
            Network containing the neuron.
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
        if len(record_from) == 1:
            if record_from[0] == "spikes":
                self._recorders["spike_detector"] = \
                    nest.Create("spike_detector", params=params))
                self._variables["spike_detector"] = ("times",)
            elif record_from[0] == "V_m" and :
                self._recorders["voltmeter"] = \
                    nest.Create("voltmeter", params=params))
                self._variables["voltmeter"] = ("V_m",)
            else:
                mm_params = params.copy()
                mm_params["record_from"] = record_from
                self._recorders["multimeter"] = \
                    nest.Create("multimeter", params=self.params))
                self._variables["multimeter"] = record_from
        else:
            if "spikes" in record_from:
                self._recorders["spike_detector"] = \
                    nest.Create("spike_detector", params=params))
                self._variables["spike_detector"] = ("times",)
            mm_record = [r for r in record_from if r != "spikes"]
            mm_params = params.copy()
            mm_params["record_from"] = mm_record
            self._variables["multimeter"] = mm_record
            self._recorders["multimeter"] = \
                nest.Create("mutlimeter", params=mm_params))

    def get_recording(self):
        recordings = {}
        for name, gid in self._recorders.items():
            data_to_get = self._variables[name]
            for d in data_to_get:
                data = nest.GetStatus(gid, "events")[0][d]
                recordings[self._id][d] = data
            if name != "spike_detector":
                data = nest.GetStatus(gid, "events")[0]["times"]
                recordings[self._id]["times"] = data
        return recordings
