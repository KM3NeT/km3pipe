# coding=utf-8
# Filename: __init__.py
# pylint: disable=locally-disabled
"""
A collection of commonly used modules.

"""
from __future__ import division, absolute_import, print_function

import resource

from km3pipe import Module


class HitCounter(Module):
    """Prints the number of hits"""
    def process(self, blob):
        try:
            print("Number of hits: {0}".format(len(blob['Hit'])))
        except KeyError:
            pass
        return blob


class BlobIndexer(Module):
    """Puts an incremented index in each blob for the key 'blob_index'"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.blob_index = 0

    def process(self, blob):
        blob['blob_index'] = self.blob_index
        self.blob_index += 1
        return blob


class StatusBar(Module):
    """Displays the current blob number"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.blob_index = 0

    def process(self, blob):
        print('-'*33 + "[Blob {0:>7}]".format(self.blob_index) + '-'*33)
        self.blob_index += 1
        return blob


class MemoryObserver(Module):
    """Shows the maximum memory usage"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

    def process(self, blob):
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory = memory / 1024 / 1024  # convert to MB
        print("Memory peak usage: {0:.3f} MB".format(memory))
