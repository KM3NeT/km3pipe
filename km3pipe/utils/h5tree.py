# coding=utf-8
# Filename: h5tree.py
"""
Print the HDF5 file structure.

Usage:
    h5tree FILE
    h5tree (-h | --help)
    h5tree --version

Options:
    FILE       Input file.
    -h --help  Show this screen.

"""
from __future__ import division, absolute_import, print_function

import tables


def h5tree(h5name):
    with tables.open_file(h5name) as h5:
        for node in h5.walk_nodes():
            print(node)


def main():
    from docopt import docopt
    arguments = docopt(__doc__)

    h5tree(arguments['FILE'])
