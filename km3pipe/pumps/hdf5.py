# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import os.path

import tables
import numpy as np

from km3pipe import Pump, Module
from km3pipe.dataclasses import HitSeries, TrackSeries
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'

POS_ATOM = tables.FloatAtom(shape=3)


class Hit(tables.IsDescription):
    channel_id = tables.UInt8Col()
    dom_id = tables.UIntCol()
    event_id = tables.UIntCol()
    id = tables.UIntCol()
    pmt_id = tables.UIntCol()
    time = tables.IntCol()
    tot = tables.UInt8Col()
    triggered = tables.BoolCol()


class Track(tables.IsDescription):
    dir = tables.FloatCol(shape=(3,))
    energy = tables.FloatCol()
    event_id = tables.UIntCol()
    id = tables.UIntCol()
    pos = tables.FloatCol(shape=(3,))
    time = tables.IntCol()
    type = tables.IntCol()


class EventInfo(tables.IsDescription):
    id = tables.IntCol()
    det_id = tables.IntCol()
    event_id = tables.UIntCol()
    frame_index = tables.UIntCol()
    mc_id = tables.IntCol()
    mc_t = tables.Float64Col()
    overlays = tables.UInt8Col()
    run_id = tables.UIntCol()
    #timestamp = tables.Float64Col()
    trigger_counter = tables.UInt64Col()
    trigger_mask = tables.UInt64Col()


class HDF5Sink(Module):
    def __init__(self, **context):
        """A Module to convert (KM3NeT) ROOT files to HDF5."""
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.index = 1
        self.h5file = tables.open_file(self.filename, mode="w", title="Test file")
        self.filters = tables.Filters(complevel=5)
        self.hits = self.h5file.create_table('/', 'hits', Hit, "Hits", filters=self.filters)
        self.mc_hits = self.h5file.create_table('/', 'mc_hits', Hit, "MC Hits", filters=self.filters)
        self.mc_tracks = self.h5file.create_table('/', 'mc_tracks', Track, "MC Tracks", filters=self.filters)
        self.event_info = self.h5file.create_table('/', 'event_info', EventInfo, "Event Info", filters=self.filters)

    def _write_hits(self, hits, hit_row):
        for hit in hits:
            hit_row['channel_id'] = hit.channel_id
            hit_row['dom_id'] = hit.dom_id
            hit_row['event_id'] = self.index
            hit_row['id'] = hit.id
            hit_row['pmt_id'] = hit.pmt_id
            hit_row['time'] = hit.time
            hit_row['tot'] = hit.tot
            hit_row['triggered'] = hit.triggered
            hit_row.append()

    def _write_tracks(self, tracks, track_row):
        for track in tracks:
            track_row['dir'] = track.dir
            track_row['energy'] = track.energy
            track_row['event_id'] = self.index
            track_row['id'] = track.id
            track_row['pos'] = track.pos
            track_row['time'] = track.time
            track_row['type'] = track.type
            track_row.append()

    def _write_event_info(self, info, info_row):
        info_row['det_id'] = info.det_id
        info_row['event_id'] = self.index
        info_row['frame_index'] = info.frame_index
        info_row['id'] = info.id
        info_row['mc_id'] = info.mc_id
        info_row['mc_t'] = info.mc_t
        info_row['overlays'] = info.overlays
        info_row['run_id'] = info.run_id
        #info_row['timestamp'] = info.timestamp
        info_row['trigger_counter'] = info.trigger_counter
        info_row['trigger_mask'] = info.trigger_mask
        info_row.append()

    def process(self, blob):
        hits = blob['Hits']
        self._write_hits(hits, self.hits.row)
        if 'MCHits' in blob:
            self._write_hits(blob['MCHits'], self.mc_hits.row)
        if 'MCTracks' in blob:
            self._write_tracks(blob['MCTracks'], self.mc_tracks.row)
        if 'Evt' in blob:
            self._write_event_info(blob['Evt'], self.event_info.row)

        if not self.index % 1000:
            self.hits.flush()
            self.mc_hits.flush()
            self.mc_tracks.flush()
            self.event_info.flush()

        self.index += 1
        return blob

    def finish(self):
        self.hits.cols.event_id.create_index()
        self.event_info.cols.event_id.create_index()
        self.mc_hits.cols.event_id.create_index()
        self.mc_tracks.cols.event_id.create_index()
        self.hits.flush()
        self.event_info.flush()
        self.mc_hits.flush()
        self.mc_tracks.flush()
        self.h5file.close()


class HDF5Pump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        if os.path.isfile(self.filename):
            self.h5_file = tables.File(self.filename)
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.index = None
        self._reset_index()

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self._reset_index()
            raise StopIteration
        self.index += 1
        return blob

    def _get_hits(self, index, table_name='hits', where='/'):
        table = self.h5_file.get_node(where, table_name)
        _channel_id = table.channel_id[index]
        _dom_id = table.dom_id[index]
        _id = table.id[index]
        _pmt_id = table.pmt_id[index]
        _time = table.time[index]
        _tot = table.tot[index]
        _triggered = table.triggered[index]
        return HitSeries.from_arrays(_channel_id, _dom_id, _id, _pmt_id,
                                     _time, _tot, _triggered)

    def _get_tracks(self, index, table_name='tracks', where='/'):
        table = self.h5_file.get_node(where, table_name)
        _dir = table.dir[index]
        _energy = table.energy[index]
        _id = table.id[index]
        _pos = table.pos[index]
        _time = table.time[index]
        _type = table.type[index]
        return TrackSeries.from_arrays(_dir, _energy, _id, _pos,
                                       _time, _type)

    def _get_event_info(self, index, table_name='info', where='/'):
        table = self.h5_file.get_node(where, table_name)
        info = {}
        info['id'] = table.id[index]
        info['det_id'] = table.det_id[index]
        info['mc_id'] = table.mc_id[index]
        info['run_id'] = table.run_id[index]
        info['trigger_mask'] = table.trigger_mask[index]
        info['trigger_counter'] = table.trigger_counter[index]
        info['overlays'] = table.overlays[index]
        #info['timestamp'] = table.timestamp[index]
        info['mc_t'] = table.mc_t[index]
        return info

    def get_blob(self, index):
        blob = {}
        blob['Hits'] = self._get_hits(index, table_name='hits')
        blob['MCHits'] = self._get_hits(index, table_name='mc_hits')
        blob['MCTracks'] = self._get_tracks(index, table_name='mc_tracks')
        blob['EventInfo'] = self._get_event_info(index, table_name='info')
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
