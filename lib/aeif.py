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

import numpy as np
from scipy.special import lambertw


def out_of_eq_aeif(params):
    '''
    Check whether aEIF neuronal parameters lead to a stable equilibrium.
    
    Args:
        params (dict): Dictionnary of the neuron's parameters with the keys
            formatted as returned by ``nest.GetStatus(aeif_neuron)``. It must
            contain at least "E_L", "I_e", "a", and "tau_w".
    
    Returns: 
        out_of_eq (bool): ``True`` if the neuron is out of equilibrium;
            ``False`` if a fixed point exists.
    '''
    A = params['E_L'] + params['I_e'] / (1 + params['a'])
    aPlusOne = 1 + params['a']
    has_FP = aPlusOne - np.exp(1 + A) > 0.
    V0minus = np.real(A - lambertw(- np.exp(A) / aPlusOne, 0))
    stable_FP = np.exp(V0minus) - 1. < min(aPlusOne, 1./ params['tau_w'])
    out_of_eq = not (has_FP and stable_FP)
    return out_of_eq


def Vprime_aeif(V, w, EL, Ie, Is):
    '''
    Returns the derivative of the membrane potential for an aEIF neuron.
    
    Args:
        V (double): membrane potential.
        w (double): adaptation variable.
        EL (double): resting potential.
        Ie (double): constant stimulation current.
        Is (double): synaptic current.
    
    Returns:
        -(V-EL) + np.exp(V) + Ie + Is - w
    '''
    return -(V-EL) + np.exp(V) + Ie + Is - w


def inv_Vprime_aeif(V, w, EL, Ie, Is):
    '''
    Inverse of Vprime_aeif.
    
    See Also:
        :func:`Vprime_aeif`
    '''
    return 1. / Vprime_aeif(V, w, EL, Ie, Is)
