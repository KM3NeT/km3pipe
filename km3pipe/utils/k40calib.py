#!/usr/bin/env python
# coding=utf-8
# Filename: k40calib.py
# vim: ts=4 sw=4 et
"""
=========================
K40 Intra-DOM Calibration
=========================

The following script calculates the PMT time offsets using K40 coincidences


Usage:
    k40calib FILE DET_ID TMAX CTMIN
    k40calib (-h | --help)
    k40calib --version

Options:
    FILE         Input file (ROOT). 
    DET_ID       Detector ID (e.g. 29).
    TMAX         Coincidence time window [default: 10].
    CTMIN        Minimum cos(angle) between PMTs for L2 [default: -1.0].
    -h --help    Show this screen.
"""
# Author: Jonas Reubelt <jreubelt@km3net.de> and Tamas Gal <tgal@km3net.de>
# License: MIT

from __future__ import division, absolute_import, print_function

from km3pipe import version
import km3pipe as kp
from km3modules import k40
from km3modules.common import StatusBar, MemoryObserver

__author__ = "Tamas Gal and Jonas Reubelt"
__copyright__ = "Copyright 2016, KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Jonas Reubelt and Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def k40calib(filename, tmax, ctmin, det_id):
    pipe = kp.Pipeline(timeit=True)
    pipe.attach(kp.io.jpp.TimeslicePump, filename=filename)
    pipe.attach(StatusBar, every=5000)
    pipe.attach(MemoryObserver, every=10000)
    pipe.attach(k40.SummaryMedianPMTRateService, filename=filename)
    pipe.attach(k40.TwofoldCounter, tmax=10)
    pipe.attach(k40.K40BackgroundSubtractor, mode='offline')
    pipe.attach(k40.IntraDOMCalibrator, ctmin=0., mode='offline', det_id=29)
    pipe.drain()


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)
    print(args)

    k40calib(args['FILE'],
             int(args['TMAX']),
             float(args['CTMIN']),
             int(args['DET_ID']))
