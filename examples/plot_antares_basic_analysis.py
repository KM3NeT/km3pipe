#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
======================
ANTARES Basic Analysis Example
==============================

"""
from __future__ import absolute_import, print_function, division

# Author: Nicole Geisselbrecht <ngeisselbrecht@km3net.de>
# License: BSD-3
# Date: 2017-10-10
# Status: Under construction...

#####################################################
# Preparation
# -----------
# Importing libraries:

import matplotlib.pyplot as plt    # our plotting module
import pandas as pd    # the main HDF5 reader
import numpy as np    # must have
import km3pipe as kp    # some KM3NeT related helper functions
import km3pipe.style
km3pipe.style.use("km3pipe")

#####################################################
# General stuff
# --------------------------
# Example plots for ANTARES HDF5 Files which are generated by the
# I3Pump.

#####################################################
# Creating some Graphs for Neutrinos
# --------------------------

#####################################################
#
filepath_neutrino = "data/MC_054405_anue_a_CC_reco.h5"

nu = pd.read_hdf(filepath_neutrino, '/nu')

nu.energy.hist(bins=100, log=True)
plt.xlabel('energy [GeV]')
plt.ylabel('number of events')
plt.title('Energy Distribution')
plt.tight_layout()

#####################################################
#
zeniths = kp.math.zenith(nu.filter(regex='^dir_.?$'))
nu['zenith'] = zeniths

plt.hist(np.cos(nu.zenith), bins=21, histtype='step', linewidth=2)
plt.xlabel(r'cos($\theta$)')
plt.ylabel('number of events')
plt.title('Zenith Distribution')
plt.tight_layout()

#####################################################
#
azimuths = kp.math.azimuth(nu.filter(regex='^dir_.?$'))
nu['azimuth'] = azimuths

plt.hist(nu.azimuth, bins=21, histtype='step', linewidth=2)
plt.xlabel('azimuth')
plt.ylabel('number of events')
plt.title('Azimuth Distribution')
plt.tight_layout()

#####################################################
#
# Starting positions of primaries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.hist2d(nu.pos_x, nu.pos_y, bins=100, cmap='viridis')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('2D Plane')
plt.colorbar()
plt.tight_layout()

#####################################################
# Creating some Graphs for Muons
# --------------------------

#####################################################
#

filepath_muon = "data/MC_054403_mupage_reco.h5"

muon = pd.read_hdf(filepath_muon, '/muon')

muon.energy.hist(bins=100, log=True)
plt.xlabel('energy [GeV]')
plt.ylabel('number of events')
plt.title('Energy Distribution')
plt.tight_layout()

#####################################################
#
zeniths = kp.math.zenith(muon.filter(regex='^dir_.?$'))
muon['zenith'] = zeniths

plt.hist(np.cos(muon.zenith), bins=21, histtype='step', linewidth=2)
plt.xlabel(r'cos($\theta$)')
plt.ylabel('number of events')
plt.title('Zenith Distribution')
plt.tight_layout()

#####################################################
#
azimuths = kp.math.azimuth(muon.filter(regex='^dir_.?$'))
muon['azimuth'] = azimuths

plt.hist(muon.azimuth, bins=21, histtype='step', linewidth=2)
plt.xlabel('azimuth')
plt.ylabel('number of events')
plt.title('Azimuth Distribution')
plt.tight_layout()

#####################################################
#
# Starting positions of primaries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.hist2d(muon.pos_x, muon.pos_y, bins=100, cmap='viridis')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('2D Plane')
plt.colorbar()
plt.tight_layout()




