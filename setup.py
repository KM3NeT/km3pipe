#!/usr/bin/env python
# Filename: setup.py
"""
KM3Pipe setup script.

"""

from setuptools import setup
import sys

import builtins
# so we can detect in __init__.py that it's called from setup.py
builtins.__KM3PIPE_SETUP__ = True

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

try:
    with open("README.rst") as fh:
        long_description = fh.read()
except UnicodeDecodeError:
    long_description = "KM3Pipe"

setup(
    name='km3pipe',
    url='http://git.km3net.de/km3py/km3pipe',
    description='An analysis framework for KM3NeT',
    long_description=long_description,
    author='Tamas Gal and Moritz Lotze',
    author_email='tgal@km3net.de',
    packages=[
        'km3pipe', 'km3pipe.io', 'km3pipe.utils', 'km3modules', 'pipeinspector'
    ],
    include_package_data=True,
    platforms='any',
    setup_requires=['numpy>=1.12', 'setuptools_scm'],
    use_scm_version=True,
    install_requires=requirements,
    python_requires='>=2.7',
    entry_points={
        'console_scripts': [
            'km3pipe=km3pipe.cmd:main',
            'pipeinspector=pipeinspector.app:main',
            'rtree=km3pipe.utils.rtree:main',
            'h5info=km3pipe.utils.h5info:main',
            'h5tree=km3pipe.utils.h5tree:main',
            'h5header=km3pipe.utils.h5header:main',
            'ptconcat=km3pipe.utils.ptconcat:main',
            'meantots=km3pipe.utils.meantots:main',
            'pushover=km3pipe.utils.pushover:main',
            'ztplot=km3pipe.utils.ztplot:main',
            'k40calib=km3pipe.utils.k40calib:main',
            'totmonitor=km3pipe.utils.totmonitor:main',
            'streamds=km3pipe.utils.streamds:main',
            'triggermap=km3pipe.utils.triggermap:main',
            'nb2sphx=km3pipe.utils.nb2sphx:main',
            'triggersetup=km3pipe.utils.triggersetup:main',
            'ligiermirror=km3pipe.utils.ligiermirror:main',
            'runtable=km3pipe.utils.runtable:main',
            'runinfo=km3pipe.utils.runinfo:main',
            'qrunprocessor=km3pipe.utils.qrunprocessor:main',
            'wtd=km3pipe.utils.wtd:main',
            'qrunqaqc=km3pipe.utils.qrunqaqc:main',
            'daqsample=km3pipe.utils.daqsample:main',
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
)

__author__ = 'Tamas Gal and Moritz Lotze'
