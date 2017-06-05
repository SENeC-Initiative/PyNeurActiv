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


# ---------------------- #
# Single value functions #
# ---------------------- #

def adim(var, value, VT, DT, gL, tm):
    if var[0] in ("E", "V", "D"):
        return np.divide(value-VT, DT)
    elif var[0] in ("w", "I", "b"):
        return np.divide(value, gL*DT)
    elif var[0] in ("t", "d"):
        return np.divide(value, tm)
    elif var[0] == "a":
        return np.divide(value, gL)
    else:
        return value


def redim(var, value):
    if var[0] in ("E", "V", "D"):
        return (value*DT+VT)
    elif var[0] in ("w", "I", "b"):
        return value*(gL*DT)
    elif var[0] == "t":
        return value*tm
    elif var[0] == "a":
        return value*gL
    else:
        raise ArgumentError("Invalid value in redim")


# ------------------------------------------ #
# Changing dimensions for a whole dictionary #
# ------------------------------------------ #

def adim_dict(di):
    '''
    Return a dimensionless dictionary from a dictionary of dimensioned neuronal
    parameters for the AEIF model with post-synaptic currents.
    '''
    VT = di["V_th"]
    DT = di["Delta_T"]
    gL = di["g_L"]
    tm = di.get("tau_m", -1)
    if tm == -1:
        tm = di["C_m"]/gL
    di_new = di.copy()
    for key, val in iter(di.items()):
        if key == "Delta_T":
            di_new[key] = (val-VT)/DT
        elif key == "weight":
            di_new[key] /= (gL*DT)
        elif key == "delay":
            di_new[key] /= tm
        elif key == "a":
            di_new[key] /= gL
        elif key == "b":
            di_new[key] /= (gL*DT)
        elif key[0] in ("V", "E"):
            di_new[key] = (val-VT)/DT
        elif key[0] in ("w", "I"):
            di_new[key] /= (gL*DT)
        elif key[0] == "t":
            di_new[key] /= tm
        elif key[0] == "g":
            di_new[key] /= gL
    return di_new

def redim_dict(di, di_dim):
    '''
    Return a dimensionned dictionary from a dictionary of dimensionless neuronal
    parameters and from the original dimensionned dict for the AEIF model.
    '''
    VT = di_dim["V_th"]
    DT = di_dim["Delta_T"]
    gL = di_dim["g_L"]
    tm = di_dim.get("tau_m", -1)
    if tm == -1:
        tm = di_dim["C_m"]/gL
    di_new = di.copy()
    for key, val in iter(di.items()):
        if key == "Delta_T":
            di_new[key] = val*DT+VT
        elif key == "weight":
            di_new[key] *= (gL*DT)
        elif key in ("delay", "IBI", "ISI", "burst_duration"):
            di_new[key] *= tm
        elif key == "a":
            di_new[key] *= gL
        elif key == "b":
            di_new[key] *= (gL*DT)
        elif key[0].lower() in ("v", "e"):
            di_new[key] = val*DT+VT
        elif key[0].lower() in ("w", "i"):
            di_new[key] *= (gL*DT)
        elif key[0].lower() == "t":
            di_new[key] *= tm
        elif key[0].lower() == "g":
            di_new[key] *= gL
    return di_new
