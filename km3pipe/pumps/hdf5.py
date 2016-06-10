# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import os.path

import numpy as np
import tables

from km3pipe import Pump, Module
from km3pipe.dataclasses import HitSeries, TrackSeries, EventInfo
from km3pipe.logger import logging
from km3pipe.reco_dtypes import recname_to_dtype

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'

POS_ATOM = tables.FloatAtom(shape=3)


class EventInfoDesc(tables.IsDescription):
    det_id = tables.IntCol()
    event_id = tables.UIntCol()
    frame_index = tables.UIntCol()
    mc_id = tables.IntCol()
    mc_t = tables.Float64Col()
    overlays = tables.UInt8Col()
    run_id = tables.UIntCol()
    # timestamp = tables.Float64Col()
    trigger_counter = tables.UInt64Col()
    trigger_mask = tables.UInt64Col()


class Hit(tables.IsDescription):
    channel_id = tables.UInt8Col()
    dom_id = tables.UIntCol()
    event_id = tables.UIntCol()
    id = tables.UIntCol()
    pmt_id = tables.UIntCol()
    run_id = tables.UIntCol()
    time = tables.IntCol()
    tot = tables.UInt8Col()
    triggered = tables.BoolCol()


class Track(tables.IsDescription):
    dir = tables.FloatCol(shape=(3,))
    energy = tables.FloatCol()
    event_id = tables.UIntCol()
    id = tables.UIntCol()
    pos = tables.FloatCol(shape=(3,))
    run_id = tables.UIntCol()
    time = tables.IntCol()
    type = tables.IntCol()


class HDF5Sink(Module):
    def __init__(self, **context):
        """A Module to convert (KM3NeT) ROOT files to HDF5."""
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.index = 1
        self.h5file = tables.open_file(self.filename, mode="w", title="KM3NeT")
        self.filters = tables.Filters(complevel=5)
        self.hits = self.h5file.create_table('/', 'hits',
                                             Hit, "Hits",
                                             filters=self.filters)
        self.mc_hits = self.h5file.create_table('/', 'mc_hits',
                                                Hit, "MC Hits",
                                                filters=self.filters)
        self.mc_tracks = self.h5file.create_table('/', 'mc_tracks',
                                                  Track, "MC Tracks",
                                                  filters=self.filters)
        self.event_info = self.h5file.create_table('/', 'event_info',
                                                   EventInfoDesc, "Event Info",
                                                   filters=self.filters)
        self._reco_tables = {}

    def _write_hits(self, hits, hit_row):
        """Iterate through the hits and write them to the HDF5 table.

        Parameters
        ----------
        hits : HitSeries
        hit_row : HDF5 TableRow

        """
        event_id = hits.event_id
        if event_id is None:
            log.error("Event ID is `None`")
        for hit in hits:
            hit_row['channel_id'] = hit.channel_id
            hit_row['dom_id'] = hit.dom_id
            hit_row['event_id'] = event_id
            hit_row['id'] = hit.id
            hit_row['pmt_id'] = hit.pmt_id
            # hit_row['run_id'] = hit.run_id
            hit_row['time'] = hit.time
            hit_row['tot'] = hit.tot
            hit_row['triggered'] = hit.triggered
            hit_row.append()

    def _write_tracks(self, tracks, track_row):
        for track in tracks:
            track_row['dir'] = track.dir
            track_row['energy'] = track.energy
            track_row['event_id'] = tracks.event_id
            track_row['id'] = track.id
            track_row['pos'] = track.pos
            # track_row['run_id'] = track.run_id
            track_row['time'] = track.time
            track_row['type'] = track.type
            track_row.append()

    def _write_event_info(self, info, info_row):
        info_row['det_id'] = info.det_id
        try:  # dealing with aanet naming conventions
            info_row['event_id'] = info.id
        except AttributeError:
            info_row['event_id'] = info.event_id
        info_row['frame_index'] = info.frame_index
        info_row['mc_id'] = info.mc_id
        info_row['mc_t'] = info.mc_t
        info_row['overlays'] = info.overlays
        info_row['run_id'] = info.run_id
        # info_row['timestamp'] = info.timestamp
        info_row['trigger_counter'] = info.trigger_counter
        info_row['trigger_mask'] = info.trigger_mask
        info_row.append()

    def _write_reco(self, track, reco_row):
        for colname, val in track.items():
            reco_row[colname] = val
        reco_row.append()

    def _write_minidst(self, minidst):
        for recname, track in minidst.items():
            if recname == 'event_id':
                continue
            if recname not in self._reco_tables:
                reco_table = self.h5file.create_table(
                    '/reco', recname.lower(),
                    np.dtype(recname_to_dtype[recname]),
                    recname, createparents=True, filters=self.filters
                )
                self._reco_tables[recname] = reco_table
            reco_table = self._reco_tables[recname]
            if not track:
                track = np.zeros(1, recname_to_dtype[recname])
                for key in track.dtype.names:
                    reco_table.row[key] = track[key]
                reco_table.row.append()
                continue
            self._write_reco(track, reco_table.row)

    def process(self, blob):
        if 'Hits' in blob:
            hits = blob['Hits']
            self._write_hits(hits, self.hits.row)
        if 'MCHits' in blob:
            self._write_hits(blob['MCHits'], self.mc_hits.row)
        if 'MCTracks' in blob:
            self._write_tracks(blob['MCTracks'], self.mc_tracks.row)
        if 'Evt' in blob and 'EventInfo' not in blob:  # skip in emergency
            self._write_event_info(blob['Evt'], self.event_info.row)
        if 'EventInfo' in blob:  # TODO: decide how to deal with that class
            self._write_event_info(blob['EventInfo'], self.event_info.row)
        if 'MiniDST' in blob:
            self._write_minidst(blob['MiniDST'])

        if not self.index % 1000:
            self.hits.flush()
            self.mc_hits.flush()
            self.mc_tracks.flush()
            self.event_info.flush()
            for tab in self._reco_tables.values():
                tab.flush()

        self.index += 1
        return blob

    def finish(self):
        self.hits.flush()
        self.event_info.flush()
        self.mc_hits.flush()
        self.mc_tracks.flush()
        for tab in self._reco_tables.values():
            tab.flush()
        self.hits.cols.event_id.create_index()
        self.event_info.cols.event_id.create_index()
        self.mc_hits.cols.event_id.create_index()
        self.mc_tracks.cols.event_id.create_index()
        # TODO: maybe index reco tables? idk
        self.h5file.close()


class HDF5Pump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, filename, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = filename
        if os.path.isfile(self.filename):
            self.h5_file = tables.File(self.filename)
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.index = None
        self._reset_index()

        try:
            event_info = self.h5_file.get_node('/', 'event_info')
            self.event_ids = event_info.cols.event_id[:]
        except tables.NoSuchNodeError:
            log.critical("No /event_info table found.")
            raise SystemExit

        self._n_events = len(self.event_ids)

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self._reset_index()
            raise StopIteration
        self.index += 1
        return blob

    def _get_hits(self, event_id, table_name='hits', where='/'):
        table = self.h5_file.get_node(where, table_name)
        rows = table.read_where('event_id == %d' % event_id)
        return HitSeries.from_table(rows, event_id)

    def _get_tracks(self, event_id, table_name='tracks', where='/'):
        table = self.h5_file.get_node(where, table_name)
        rows = table.read_where('event_id == %d' % event_id)
        return TrackSeries.from_table(rows, event_id)

    def _get_event_info(self, event_id, table_name='event_info', where='/'):
        table = self.h5_file.get_node(where, table_name)
        return EventInfo.from_table(table[event_id])

    def get_blob(self, index):
        event_id = self.event_ids[index]
        blob = {}
        blob['Hits'] = self._get_hits(event_id, table_name='hits')
        blob['MCHits'] = self._get_hits(event_id, table_name='mc_hits')
        blob['MCTracks'] = self._get_tracks(event_id, table_name='mc_tracks')
        blob['EventInfo'] = self._get_event_info(event_id,
                                                 table_name='event_info')
        return blob

    def finish(self):
        """Clean everything up"""
        self.h5_file.close()

    def _reset_index(self):
        """Reset index to default value"""
        self.index = 0

    def __len__(self):
        return self._n_events

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self.index >= self._n_events:
            self._reset_index()
            raise StopIteration
        blob = self.get_blob(self.index)
        self.index += 1
        return blob

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_blob(index)
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self.get_blob(i)
