#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================
Basic Analysis Example (ROOT)
=============================

"""

# Authors: Tamás Gál <tgal@km3net.de>
# License: BSD-3
# Date: 2021-02-11
#
# sphinx_gallery_thumbnail_number = 1

#####################################################
# Introduction
# ------------
# This document shows how a very basic exploratory analysis of an
# offline-ROOT file can be done using km3pipe.

#####################################################
# Preparation
# ~~~~~~~~~~~
# The very first thing we do is importing our libraries and setting up
# the Jupyter Notebook environment.

import matplotlib.pyplot as plt
import km3pipe as kp
import km3io
from km3net_testdata import data_path

#####################################################
# this is just to make our plots a bit "nicer", you can skip it

kp.style.use("km3pipe")

#####################################################
# Note for Lyon Users
# ~~~~~~~~~~~~~~~~~~~
# If you are working on the Lyon cluster, you just need to load the
# Python module with ``module load python`` and you are all set.

#####################################################
# First Look at the Data
# ----------------------
# Let's load a sample file which is coming from a gSeaGen CC production and
# was reconstructed with the JChain (JGandalf track reconstruction in Jpp) and
# written in the offline-ROOT format:

filename = data_path("offline/mcv5.11r2.gsg_muonCChigherE-CC_50-5000GeV.km3_AAv1.jterbr00004695.jchain.aanet.498.root")

#####################################################
# There are two main libraries to access the actual data, one is ``km3io``
# and the other is ``km3pipe``, while ``km3io`` is a low level library and is
# nice to explore the data and ``km3pipe`` is a full-fledge framework with
# tools to build modular pipelines.
#
# Let's explore the file with ``km3io`` first:

f = km3io.OfflineReader(filename)
print(f)

#####################################################
# As you can see, it automatically read the ``E/Evt`` tree of the ROOT file and
# recognised a number of entries which is represented in brackets. You can treat
# it as a vector of events and index/slice it, while the number in the printed
# square brackets shows the number of affected events
# (not the ID of the selected event):

print(f[0])     # selects the first event
print(f[23])    # selects the 24th event
print(f[5:23])  # selects the events from ID 5 to 23

#####################################################
# The ``.keys()`` function gives you all the possible sub-branches available:

print(f.keys())


#####################################################
# The reconstructed tracks are sitting on the ``.tracks`` sub-branch. If you
# access all of them, you will recieve a vector with nested track-vectors
# inside:

print(f.tracks)


#####################################################
# This means that you can select the tracks for a specific event by:

print(f[5].tracks)

#####################################################
# This tracks vector is now a sub-branch and you can descover its fields with
# the ``.fields`` property:

print(f[5].tracks.fields)

#####################################################
# The z-components of the track directions from the sixth event are therefore:

print(f[5].tracks.dir_z)


#####################################################
# And to access the z-component of the direction of all tracks from all events,
# you can use:

print(f.tracks.dir_z)

