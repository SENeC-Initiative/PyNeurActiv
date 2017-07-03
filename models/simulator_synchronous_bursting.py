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

""" Calculate the properties from a simulation """

from nngt.simulation import (make_nest_network, monitor_nodes, plot_activity,
                             activity_types)

from collections import namedtuple
from six import add_metaclass
from weakref import proxy
import warnings

import numpy as np
import matplotlib.pyplot as plt

import nest

from .. import lib as _plib
from ..analysis import (activity_types, find_idx_nearest, data_from_nest,
                        interburst_properties, spiking_properties)

# ------------------ #
# Check NEST version #
# ------------------ #

nest_version = nest.version()
while not nest_version[0].isdigit():
    nest_version = nest_version[1:]

dot_pos = nest_version.find('.')
dot_pos2 = nest_version[dot_pos + 1:].find('.')

version_major = int(nest_version[:dot_pos])
version_minor = int(nest_version[dot_pos + 1:dot_pos + 1 + dot_pos2])

useGIDCollection = False
if version_major > 2:
    useGIDCollection = True
elif version_minor <= 11:
    raise ImportError("NEST >= 2.12.0 required!")


# --------------- #
# Simulator class #
# --------------- #

@add_metaclass(_plib.ResNames)
class Simulator_SynchroBurst:

    ''' Class computing the theoretical values '''

    _res_names = ('burst_duration', 'IBI', 'ISI', 'std_ISI', 'num_spikes',
                  'std_num_spikes')

    @classmethod
    def from_nngt_network(cls, network, resolution=None, monitor_rate=None,
                          mean_field=True, omp=1, ignore_errors=False):
        '''
        Create a simulator instance from a :class:`nngt.Network` object.

        Parameters
        ----------
        network : :class:`nngt.Network`
            Network to copy.

        All the other parameters of the ``__init__`` method except for
        `num_neurons`.

        See also
        --------
        :func:`~PyNeurActiv.Simulator2017_SynchroBurst.__init__`,
        :func:`~PyNeurActiv.Simulator2017_SynchroBurst.from_nest_network`.
        '''
        # get the neurons
        num_neurons = network.node_nb()
        # check whether the network has been converted to NEST
        gids = network.nest_gid
        if gids is None:
            # else reset kernel, set resolution and omp, then create network
            nest.ResetKernel()
            nest.SetKernelStatus({"local_num_threads": omp})
            np.random.seed()
            msd = int(100000*np.random.random())
            nest.SetKernelStatus({'grng_seed': msd})
            nest.SetKernelStatus(
                {'rng_seeds': range(msd + 1, msd + omp + 1)})
            if resolution is not None:
                nest.SetKernelStatus({'resolution': resolution})
            gids = network.to_nest()
        di_param = nest.GetStatus((gids[0],))[0]
        di_param["weight"] = np.average(network.get_weights())
        di_param["delay"] = np.average(network.get_delays())
        sim = cls(num_neurons, di_param, resolution=resolution,
                  monitor_rate=monitor_rate, mean_field=mean_field, omp=omp,
                  create_network=False, gids=gids)
        sim.network = proxy(network)
        return sim

    @classmethod
    def from_nest_network(cls, gids, resolution=None, monitor_rate=None,
                          mean_field=True, omp=1, ignore_errors=False):
        '''
        Create a Simulator instance from the GIDs of a NEST network and the
        network parameters.

        Parameters
        ----------
        gids : tuple
            NEST GIDs.

        All the other parameters of the ``__init__`` method except for
        `num_neurons`.

        See also
        --------
        :func:`~PyNeurActiv.Simulator2017_SynchroBurst.__init__`,
        :func:`~PyNeurActiv.Simulator2017_SynchroBurst.from_nngt_network`.
        '''
        # get the neurons
        neurons = nest.GetNodes([0], properties={'element_type': 'neuron'})[0]
        num_neurons = len(neurons)
        di_param = nest.GetStatus((neurons[0],))[0]
        edges_info = nest.GetStatus(nest.GetConnections(source=neurons))
        weight = 0.
        delay = 0.
        num_edges = 0.
        for di in edges_info:
            weight += di["weight"]
            delay += di["delay"]
            num_edges += 1.
        if not num_edges:
            raise RuntimeError("No edges in the network.")
        di_param["weight"] = weight / num_edges
        di_param["delay"] = delay / num_edges
        sim = cls(num_neurons, di_param, resolution=resolution,
                  monitor_rate=monitor_rate, mean_field=mean_field, omp=omp,
                  create_network=False, gids=gids)
        return sim

    def __init__(self, num_neurons, parameters, resolution=None,
                 monitor_rate=None, mean_field=True, omp=1,
                 ignore_errors=False, **kwargs):
        '''
        Initialize the class:
            - create the network in NEST
            - prepare the dict to contain the results
            - get the state variables that can be recorded
            - create the devices to monitor the spikes and variables

        Parameters
        ----------
        num_neurons : :obj:`tuple`
            Number of neurons in the network.
        parameters : :obj:`dict`
            Parameters of the neurons, same as in `Fardet2017_SynchroBurst`.
        resolution : float, optional (default: ``None``)
            Timestep of the simulation in NEST.
        monitor_rate : float, optional (default: ``None``)
            Sampling frequency for the monitoring device (should be greater or
            equal to `resolution`).
        mean_field : bool, optional (default: True)
            Whether the signals from all neurons should be averaged or kept
            separated (it is highly recommanded to set it to False to compute
            phase synchronicity).
        omp : int, optional (default: 1)
            Number of OpenMP processes to use during the simulation.

        See also
        --------
        :class:`~PyNeurActiv.models.Fardet2017_SynchroBurst`
        '''
        self.num_neurons = num_neurons
        self._params = parameters.copy()
        self.network = None
        self.simtime = None
        self.simulated = False
        self.ignore_errors = ignore_errors
        # set NEST and network
        if kwargs.get("create_network", True):
            nest.ResetKernel()
        if nest.GetKernelStatus('network_size') == 1:
            if nest.GetKernelStatus('local_num_threads') != omp:
                nest.SetKernelStatus({"local_num_threads": omp})
                np.random.seed()
                msd = int(100000*np.random.random())
                nest.SetKernelStatus({'grng_seed': msd})
                nest.SetKernelStatus(
                    {'rng_seeds': range(msd + 1, msd + omp + 1)})
            if resolution is not None:
                nest.SetKernelStatus({'resolution': resolution})
                self.resolution = resolution
            else:
                self.resolution = nest.GetKernelStatus('resolution')
        else:
            print("Could not change OMP number and resolution: existing nodes "
                  "in NEST.")
            self.resolution = nest.GetKernelStatus('resolution')
        if kwargs.get("create_network", True):
            self._make_network()
        else:
            assert "gids" in kwargs, "`gids` required if `create_network` " \
                                     + "is False."
            self.gids = kwargs["gids"]
        # monitoring
        self.mf = mean_field
        recordables = self._monitor(monitor_rate)
        # get parameters
        self._adim_params = _plib.adim_dict(self._params)
        # results
        self._res = {res: np.NaN for res in self.__class__.result_names}
        for varname in recordables:
            varname = "V" if varname == "V_m" else varname
            self._res[varname + "_min"] = np.NaN
            self._res[varname + "_max"] = np.NaN
        self.SimResults = namedtuple("SimResults", list(self._res.keys()))

    @property
    def results(self):
        return self.SimResults(**self._res)

    #--------------------------------------------------------------------------
    # Simulation and computations

    def simulate(self, simtime, mbis=None, show=False, sort=None):
        '''
        Simulate the activity.

        Parameters
        ----------
        simtime : double
            Duration of the simulation in ms.
        mbis : double, optional (default: None)
            Maximal interval between 2 consecutive spikes to consider them as
            belonging to the same burst. If not specified, it will be set as
            `1.5 * delay`.
        show : bool, optional (default: False)
            Whether the resulting activity should be displayed (available only
            if the NNGT package is present).
        sort : :obj:`str`, optional (default: None)
            Sort neurons depending on a given property (available only
            if the NNGT package is present and simulator was created using
            the ``from_nngt_network`` method).

        See also
        --------
        :func:`nngt.simulation.plot_activity`.
        '''
        mbis = 1.5 * self._params['delay'] if mbis is None else mbis
        self.simtime = simtime
        # check that the neurons are out of equilibrium
        out_of_eq = _plib.out_of_eq_aeif(self._adim_params)
        if out_of_eq:
            try:
                nest.Simulate(simtime)
                if show:
                    from nngt.simulation import plot_activity
                    self.fignums = plot_activity(
                        self.recorders, self.record, show=False,
                        network=self.network, gids=self.gids, hist=False,
                        sort=sort, limits=(0, simtime))
                else:
                    self.fignums = None
                self.activity = activity_types(
                    spike_detector=self.recorders[0], limits=(0.,simtime),
                    network=self.network, mbis=mbis, fignums=self.fignums,
                    show=show)
                self.phases = self.activity.phases
                self.simulated = True
            except Exception as e:
                if self.ignore_errors:
                    print(e)
                else:
                    raise
        else:
            raise RuntimeError("Parameters lead to stable equilibrium.")
        if not self.simulated:
            self.phases = {"network_burst": [], "mixed": [], "quiescent": [],
                           "local_burst": []}

    def compute_properties(self, simtime=None, resimulate=False, mbis=10.,
                           steady_state=0, adim=False):
        if resimulate or simtime is not None:
            simu_length = self.simtime if simtime is None else simtime
            self.simulate(simu_length, mbis=mbis)
        else:
            assert hasattr(self, 'phases'), ("No simulation run, please " +
                                             "enter a value for `simtime`.")
        # get data
        spike_times, senders, time_var, data_var = data_from_nest(self.recorders)
        avg_var = data_var if self.mf else {key: np.average(val, axis=0)
                                            for key, val in data_var.items()}
        # get bursts
        lst_bursts = self.phases["network_burst"]
        num_bursts = len(lst_bursts[steady_state:])
        # compute
        if num_bursts:
            # check that the last period is not a trucated burst, else skip it
            last_burst_end = lst_bursts[-1][-1]
            ends_with_burst = True
            for key, val in self.phases.items():
                if key != "network_burst" and val:
                    ends_with_burst *= (last_burst_end > val[-1][-1])
            if ends_with_burst:
                lst_bursts.pop()
                num_bursts -= 1
        if num_bursts:
            for res in self._res:
                self._res[res] = 0.
            stimes = []
            b_got_ib_prop = False
            for i, burst in enumerate(lst_bursts[steady_state:]):
                # T_burst
                self._res["burst_duration"] += burst[1] - burst[0]
                # IBI and extremal values of the recordables
                if i > 0:
                    interburst_properties(
                        lst_bursts, i, steady_state, time_var, avg_var,
                        self.resolution, self._res)
                    b_got_ib_prop = True
                # ISI and num_spikes
                spiking_properties(burst, spike_times, senders, self._res)
            if not b_got_ib_prop:
                warnings.warn(
                   "Simulation too short to get interburst properties " +
                   "after steady_state is reached; either decrease " +
                   "steady_state or increase simulation time.")
            ignore = ('ISI', 'IBI', 'std_ISI', 'num_spikes', 'std_num_spikes',
                      'burst_duration')
            avgb = ('num_spikes', 'std_num_spikes', 'ISI', 'std_ISI',
                    'burst_duration')
            for key in iter(self._res.keys()):
                if key in avgb:
                    self._res[key] /= float(num_bursts)
                else:
                    if num_bursts-1:
                        self._res[key] /= float(num_bursts-1)
                if key not in ignore:
                    self._res[key] /= float(self.num_neurons)
                self._res[key] = np.around(self._res[key], 2)
            if self._res["num_spikes"] <= 1:
                for res in self._res:
                    self._res[res] = np.NaN
        else:
            warnings.warn("No detected bursts")
        return self.SimResults(**self._res)

    def time_evolution(self, simtime, start_time=None, start_burst=0,
                       num_bursts=None, mbis=10., show=False,
                       show_raster=False):
        '''
        Return the time evolution of the average neuron.

        Parameters
        ----------
        simtime : double
            Duration of the simulation.
        start_time : double, optional (default: None)
            Start time for the recording.
        start_burst : int, optional (default: None)
            Index of the starting burst (same as Python lists: starts at 0).
        num_bursts : int, optional (defaault: None)
            Number of burst that should be plotted.
        show : bool, optional (default: False)
            Whether the time evolution should be plotted.

        Returns
        -------
        ts, Vs, ws : numpy arrays containing the time, voltage, and
            adaptation variables.
        '''
        if simtime != self.simtime or not self.simulated:
            self.simulate(simtime, mbis=mbis, show=show_raster)
        (_, _, ts, data_var), avg_var = data_from_nest(self.recorders), {}
        for key, val in data_var.items():
            avg_var[key] = (val[0] / self.num_neurons if self.mf
                            else np.average(val, axis=0))
        Vs, ws = avg_var["V_m"], avg_var["w"]
        lst_bursts = self.phases["network_burst"]
        if start_burst is not None:
            if len(lst_bursts) >= start_burst:
                start_time = (lst_bursts[start_burst][-1]
                              if start_burst > 0 else 0.)
                if start_time >= ts[-1]:
                    raise AttributeError("`start_burst` too high for simtime.")
            else:
                raise AttributeError("`start_burst` is too high for simtime.")
        if start_time is not None:
            idx = find_idx_nearest(ts, start_time) + 1
            if idx >= len(Vs):
                raise AttributeError("`start_time` is too high for simtime.")
            ts, Vs, ws = ts[idx:], Vs[idx:], ws[idx:]
        if num_bursts is not None:
            if len(lst_bursts) > start_burst + num_bursts:
                end_time = lst_bursts[start_burst + num_bursts][-1]
                idx = find_idx_nearest(ts, end_time)
                ts, Vs, ws = ts[:idx], Vs[:idx], ws[:idx]
            else:
                raise AttributeError("`num_bursts` is too high for simtime.")
        ts -= ts[0]
        if show:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            ax1.plot(ts, Vs)
            ax1.set_ylabel("V (mV)")
            ax2.plot(ts, ws)
            ax2.set_ylabel("w (pA)")
            ax2.set_xlabel("Time (ms)")
        return ts, Vs, ws

    #--------------------------------------------------------------------------
    # Tools

    def randomize(self, parameter, values):
        '''
        Randomize one or several neuronal parameters.

        Parameters
        ----------
        parameter : string or list of strings
            Name of the parameter that should be randomized.
        values : array-like or list of arrays
            Values (one per neuron) to randomize the parameter. If `parameter`
            is a list, `values` should be a list of the same size.
        '''
        if isinstance(parameter, str):
            nest.SetStatus(self.gids, parameter, values)
        else:
            for param, val in zip(parameter, values):
                nest.SetStatus(self.gids, param, val)

    def _make_network(self):
        ''' Create network in NEST '''
        default_params = nest.GetDefaults("aeif_psc_alpha")
        n_param = {
            k: v for k, v in self._params.items() if k in default_params}
        self.gids = nest.Create(
            "aeif_psc_alpha", self.num_neurons, params=n_param)
        conn_spec = {'rule': 'fixed_indegree',
                     'indegree': self._params['avg_deg']}
        syn_spec={'weight': self._params['weight'],
                  'delay': self._params['delay']}
        nest.Connect(self.gids, self.gids, conn_spec=conn_spec,
                     syn_spec=syn_spec)

    def _monitor(self, monitor_rate=None):
        ''' Monitor the nodes (record V, w and spikes) '''
        recordables = ('V_m', 'w', 'I_syn_ex')
        mm_param = {
             'record_from': recordables,
             'interval': self.resolution,
             'to_accumulator': self.mf
        }
        if monitor_rate is not None:
            mm_param['interval'] = monitor_rate
        else:
            monitor_rate = self.resolution
        self.monitor_rate = monitor_rate
        self.recorders = []
        sd = nest.Create("spike_detector")
        nest.Connect(self.gids, sd)
        mm = nest.Create("multimeter", params=mm_param)
        nest.Connect(mm, self.gids)
        self.recorders = (sd, mm)
        self.record = ['spikes'] + list(recordables)
        return recordables
