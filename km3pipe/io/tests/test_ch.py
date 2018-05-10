# Filename: test_ch.py
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212
from km3pipe.testing import TestCase, patch, Mock
from km3pipe.io.ch import CHPump

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2018, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestCHPump(TestCase):
    def test_init(self):
        pass
