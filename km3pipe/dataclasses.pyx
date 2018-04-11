# coding=utf-8
# Filename: dataclasses.py
# cython: embedsignature=True
# cython: linetrace=True
# cython: profile=True
# distutils: define_macros=CYTHON_TRACE=1
# pylint: disable=W0232,C0103,C0111
"""
Dataclasses for internal use. Heavily based on Numpy arrays.
"""
from __future__ import division, absolute_import, print_function

from collections import namedtuple, defaultdict
import ctypes
from libcpp cimport bool as c_bool  # noqa
from six import with_metaclass
from struct import Struct, calcsize

import numpy as np
from numpy.lib import recfunctions as rfn
cimport numpy as np
cimport cython

np.import_array()

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"
__all__ = (
    'Table',)

DEFAULT_H5LOC = '/misc'

TEMPLATE_H5LOCS = {
    'EventInfo': '/event_info',
    'TimesliceHits': '/time_slice_hits',
    'Hits': '/hits',
    'CalibHits': '/hits',
    'McHits': '/mc_hits',
    'CalibMcHits': '/mc_hits',
    'Tracks': '/tracks',
    'McTracks': '/mc_tracks',
    'SummaryFrameSeries': '/summary_frame_series',
    'SummaryFrameInfo': '/summary_slice_info',
}

TEMPLATE_DTYPES = {
    'EventInfo': np.dtype([
        ('det_id', '<i4'),
        ('frame_index', '<u4'),
        ('livetime_sec', '<u8'),
        ('mc_id', '<i4'),
        ('mc_t', '<f8'),
        ('n_events_gen', '<u8'),
        ('n_files_gen', '<u8'),
        ('overlays', '<u4'),
        ('trigger_counter', '<u8'),
        ('trigger_mask', '<u8'),
        ('utc_nanoseconds', '<u8'),
        ('utc_seconds', '<u8'),
        ('weight_w1', '<f8'),
        ('weight_w2', '<f8'),
        ('weight_w3', '<f8'),
        ('run_id', '<u8'),
        ('event_id', '<u8'),
        ('group_id', '<u4'),
    ]),
    'TimesliceHits': np.dtype([
        ('channel_id', 'u1'),
        ('dom_id', '<u4'),
        ('time', '<i4'),
        ('tot', 'u1'),
        ('group_id', '<u4')
    ]),
    'Hits': np.dtype([
        ('channel_id', 'u1'),
        ('dom_id', '<u4'),
        ('time', '<f8'),
        ('tot', 'u1'),
        ('triggered', 'u1'),
        ('group_id', '<u4')
    ]),
    'CalibHits': np.dtype([
        ('channel_id', 'u1'),
        ('dir_x', '<f4'),
        ('dir_y', '<f4'),
        ('dir_z', '<f4'),
        ('dom_id', '<u4'),
        ('du', 'u1'),
        ('floor', 'u1'),
        ('pos_x', '<f4'),
        ('pos_y', '<f4'),
        ('pos_z', '<f4'),
        ('t0', '<f4'),
        ('time', '<f8'),
        ('tot', 'u1'),
        ('triggered', 'u1'),
        ('group_id', '<u4'),
    ]),
    'McHits': np.dtype([
        ('a', 'f4'),
        ('origin', '<u4'),
        ('pmt_id', '<u4'),
        ('time', 'f8'),
        ('group_id', '<u4'),
    ]),
    'CalibMcHits': np.dtype([
        ('a', 'f4'),
        ('dir_x', '<f4'),
        ('dir_y', '<f4'),
        ('dir_z', '<f4'),
        ('origin', '<u4'),
        ('pmt_id', '<u4'),
        ('pos_x', '<f4'),
        ('pos_y', '<f4'),
        ('pos_z', '<f4'),
        ('time', 'f8'),
        ('group_id', '<u4'),
    ]),
    'Tracks': np.dtype([
        ('bjorkeny', '<f8'),
        ('dir_x', '<f8'),
        ('dir_y', '<f8'),
        ('dir_z', '<f8'),
        ('energy', '<f8'),
        ('id', '<u4'),
        ('interaction_channel', '<u4'),
        ('is_cc', '<u4'),
        ('length', '<f8'),
        ('pos_x', '<f8'),
        ('pos_y', '<f8'),
        ('pos_z', '<f8'),
        ('time', '<i4'),
        ('type', '<i4'),
        ('group_id', '<u4'),
    ]),
    'McTracks': np.dtype([
        ('bjorkeny', '<f8'),
        ('dir_x', '<f8'),
        ('dir_y', '<f8'),
        ('dir_z', '<f8'),
        ('energy', '<f8'),
        ('id', '<u4'),
        ('interaction_channel', '<u4'),
        ('is_cc', '<u4'),
        ('length', '<f8'),
        ('pos_x', '<f8'),
        ('pos_y', '<f8'),
        ('pos_z', '<f8'),
        ('time', '<i4'),
        ('type', '<i4'),
        ('group_id', '<u4'),
    ]),
    'SummaryFrameInfo': np.dtype([
        ('dom_id', '<u4'),
        ('fifo_status', '<u4'),
        ('frame_id', '<u4'),
        ('frame_index', '<u4'),
        ('has_udp_trailer', '<u4'),
        ('high_rate_veto', '<u4'),
        ('max_sequence_number', '<u4'),
        ('n_packets', '<u4'),
        ('slice_id', '<u4'),
        ('utc_nanoseconds', '<u4'),
        ('utc_seconds', '<u4'),
        ('white_rabbit_status', '<u4'),
    ]),
    'SummaryframeSeries': np.dtype([
        ('dom_id', '<u4'),
        ('max_sequence_number', '<u4'),
        ('n_received_packets', '<u4'),
        ('group_id', '<u4'),
    ]),
}


def has_structured_dt(arr):
    """Check if the array representation has a structured dtype."""
    arr = np.asanyarray(arr)
    return is_structured(arr.dtype)


def is_structured(dt):
    """Check if the dtype is structured."""
    if not hasattr(dt, 'fields'):
        return False
    return not (dt.fields is None)


def inflate_dtype(arr, names):
    """Create structured dtype from a 2d ndarray with unstructured dtype."""
    arr = np.asanyarray(arr)
    if has_structured_dt(arr):
        return arr.dtype
    s_dt = arr.dtype
    dt = [(n, s_dt) for n in names]
    dt = np.dtype(dt)
    return dt


class Table(np.recarray):
    """2D generic Table with grouping index.

    This class adds the following to ``np.recarray``:

    Parameters
    ----------
    data: array-like or dict(array-like)
        numpy array with structured/flat dtype, or dict of arrays.
    h5loc: str
    Location in HDF5 file where to store the data. [default: '{h5l}'
    dtype: numpy dtype
        Datatype over array. If not specified and data is an unstructured
        array, ``names`` needs to be specified. [default: None]
    colnames: list(str)
        Column names to use when generating a dtype from unstructured arrays.
        [default: None]

    Attributes
    ----------
    h5loc: str
        HDF5 group where to write into. (default='{h5l}')

    Methods
    -------
    from_dict(arr_dict, dtype=None, **kwargs)
        Create an Table from a dict of arrays (similar to pandas).
    from_template(data, template, **kwargs)
        Create an array from a dict of arrays with a predefined dtype.
    sorted(by)
        Sort the table by one of its columns.
    append_columns(colnames, values)
        Append new columns to the table.
    """.format(h5l=DEFAULT_H5LOC)

    def __new__(cls, data, h5loc=DEFAULT_H5LOC, dtype=None,
                colnames=None, **kwargs):
        if isinstance(data, dict):
            return cls.from_dict(data, h5loc=h5loc, dtype=dtype, **kwargs)
        if not has_structured_dt(data):
            # flat (nonstructured) dtypes fail miserably!
            # default to `|V8` whyever
            if dtype is None or not is_structured(dtype):
                # infer structured dtype from array data + column names
                if colnames is None:
                    raise ValueError(
                        "Need to either specify column names or a "
                        "structured dtype when passing unstructured arrays!"
                        )
                dtype = inflate_dtype(data, colnames)
            # this *should* have been checked above, but do this
            # just to be sure in case I screwed up the logic above;
            # users will never see this, this should only show in tests
            assert is_structured(dtype)
            data = np.asanyarray(data).view(dtype)
            dtype = data.dtype

        obj = np.asanyarray(data, dtype=dtype).view(cls)
        obj.h5loc = h5loc
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            # called from explicit contructor
            return obj
        # views or slices
        self.h5loc = getattr(obj, 'h5loc', DEFAULT_H5LOC)
        # attribute access returns void instances on slicing/iteration
        # kudos to https://github.com/numpy/numpy/issues/3581#issuecomment-108957200
        if obj is not None and type(obj) is not type(self):
            self.dtype = np.dtype((np.record, obj.dtype))

    @staticmethod
    def _expand_scalars(arr_dict):
        scalars = []
        maxlen = 1      # have at least 1-elem arrays
        for k, v in arr_dict.items():
            if np.isscalar(v):
                scalars.append(k)
            elif len(v) > maxlen:
                maxlen = len(v)
        for s in scalars:
            arr_dict[s] = np.full(maxlen, arr_dict[s])
        return arr_dict

    @classmethod
    def from_dict(cls, arr_dict, dtype=None, **kwargs):
        """Generate a table from a dictionary of arrays.
        """
        # i hope order of keys == order or values
        if dtype is None:
            names = list(arr_dict.keys())
        else:
            dtype = np.dtype(dtype)
            dt_names = [f for f in dtype.names]
            dict_names = [k for k in arr_dict.keys()]
            if not set(dt_names) == set(dict_names):
                raise KeyError('Dictionary keys and dtype fields do not match!')
            names = list(dtype.names)

        arr_dict = cls._expand_scalars(arr_dict)
        return cls(np.rec.fromarrays(arr_dict.values(), names=names,
                                     dtype=dtype), **kwargs)

    @property
    def templates_avail(self):
        return sorted(list(TEMPLATE_DTYPES.keys()))

    @classmethod
    def from_template(cls, data, template):
        """Create a table from a predefined datatype.

        See the ``templates_avail`` property for available names.

        Parameters
        ----------
        data
            Data in a format that the ``__init__`` understands.
        template: str
            Name of the dtype template to use.
        """
        dt = TEMPLATE_DTYPES[template]
        loc = TEMPLATE_H5LOCS[template]
        return cls(data, h5loc=loc, dtype=dt)

    def append_columns(self, colnames, values, **kwargs):
        """Append new columns to the table.

        See the docs for ``numpy.lib.recfunctions.append_fields`` for an
        explanation of the options.
        """
        new_arr = rfn.append_fields(self, colnames, values,
                                    usemask=False, **kwargs)
        return self.__class__(new_arr, h5loc=self.h5loc)

    def sorted(self, by, **kwargs):
        """Sort array by a column.

        Parameters
        ==========
        by: str
            Name of the columns to sort by(e.g. 'time').
        """
        sort_idc = np.argsort(self[by], **kwargs)
        return self.__class__(self[sort_idc], h5loc=self.h5loc)
