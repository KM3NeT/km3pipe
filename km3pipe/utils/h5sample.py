#!/usr/bin/env python3
# Filename: h5sample.py
"""
Extract a specific number of groups (usually events) and data from a KM3NeT
HDF5 file.

Usage:
    h5sample [options] FILENAME
    h5sample (-h | --help)
    h5sample --version

Options:
    -o OUTFILE                  Output file.
    -n N_GROUPS                 Number of groups (events) to extract.
    -h --help                   Show this screen.
    --version                   Show the version.

"""
from thepipe import Provenance
import km3pipe as kp
import km3modules as km


def main():
    from docopt import docopt

    args = docopt(__doc__, version=kp.version)


    outfile = args["-o"]
    if outfile is None:
        outfile = args["FILENAME"] + ".h5"

    provfile = args["--provenance-file"]
    if provfile is None:
        provfile = outfile + ".prov.json"

    Provenance().outfile = provfile

    pipe = kp.Pipeline(timeit=args["--timeit"])
    pipe.attach(kp.io.OfflinePump, filename=args["FILENAME"], step_size=step_size)
    pipe.attach(km.StatusBar, every=100)
    pipe.attach(km.common.MemoryObserver, every=500)
    if args["--offline-header"]:
        pipe.attach(km.io.OfflineHeaderTabulator)
    if args["--event-info"]:
        pipe.attach(km.io.EventInfoTabulator)
    if args["--offline-hits"]:
        pipe.attach(km.io.HitsTabulator, name="Offline", kind="offline")
    if args["--online-hits"]:
        pipe.attach(km.io.HitsTabulator, name="Online", kind="online")
    if args["--mc-hits"]:
        pipe.attach(km.io.HitsTabulator, name="MC", kind="mc")
    if args["--mc-tracks"]:
        pipe.attach(km.io.MCTracksTabulator, read_usr_data=args["--mc-tracks-usr-data"])
    if args["--reco-tracks"]:
        pipe.attach(km.io.RecoTracksTabulator)
    pipe.attach(kp.io.HDF5Sink, filename=outfile)
    pipe.drain()
