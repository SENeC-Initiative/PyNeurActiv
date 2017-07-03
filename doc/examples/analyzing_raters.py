#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Using a custom recorder """

from pprint import pprint

import numpy as np
import nest

import nngt
from nngt.simulation import monitor_nodes
import PyNeurActiv as pna

import matplotlib.pyplot as plt


num_omp = 5
nest.SetKernelStatus({'local_num_threads': num_omp, 'overwrite_files': True})


''' Make a neuronal population '''

di_param = {
    'V_reset': -58.,
    'V_peak': 0.0,
    'V_th': -50.,
    'I_e': 300.,
    'g_L': 9.,
    'tau_w': 300.,
    'E_L': -70.,
    'Delta_T': 2.,
    'a': 2.,
    'b': 60.,
    'C_m': 200.,
    'V_m': -60.,
    'w': 0.,
    'tau_syn_ex': 0.2
}

pop = nngt.NeuralPop.uniform(
    1000, neuron_model='aeif_psc_alpha', neuron_param=di_param)

net = nngt.generation.erdos_renyi(
    avg_deg=100, nodes=1000, population=pop, weights=43.)
gids = net.to_nest()


''' Record from it '''

rec_param = [{'to_file': True}]

recorder, recorded = monitor_nodes(gids, params=rec_param, network=net)
nest.Simulate(2000.)

activity = pna.analysis.raster_analysis(recorder[0], limits=[300., np.inf])


''' Plot the spikes with sorted neurons '''

pprint(activity.properties())
pna.plot.raster_plot(activity, sort='spikes')

plt.show()
