# Filename: test_style.py
# pylint: disable=C0111,E1003,R0904,C0103,R0201,C0102

from km3pipe.testing import TestCase, patch
from km3pipe.style import get_style_path, ColourCycler, use

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestStyle(TestCase):

    def test_get_style_path(self):
        gsp = get_style_path
        self.assertTrue(gsp('km3pipe').endswith(
            "kp-data/stylelib/km3pipe.mplstyle"))
        self.assertTrue(gsp('foo').endswith("/stylelib/foo.mplstyle"))
        self.assertTrue(gsp('bar').endswith("/stylelib/bar.mplstyle"))


class TestColourCycler(TestCase):
    def test_available(self):
        cc = ColourCycler()
        self.assertTrue('km3pipe' in cc.available)
        self.assertTrue('classic' in cc.available)

    def test_next(self):
        cc = ColourCycler()
        first = cc.next
        second = cc.next
        self.assertTrue(first != second)

    def test_next_a_few_times(self):
        cc = ColourCycler()
        last_colour = ""
        for i in range(100):
            current_colour = cc.next
            self.assertTrue(last_colour != current_colour)
            last_colour = current_colour

    def test_raise_keyerror_if_style_not_available(self):
        with self.assertRaises(KeyError):
            cc = ColourCycler("foo")  # noqa


class TestStyles(TestCase):
    @patch('matplotlib.pyplot')
    def test_non_existent_style(self, plt_mock):
       use('non-existent')
       assert not plt_mock.style.use.called

    @patch('matplotlib.pyplot')
    def test_km3pipe(self, plt_mock):
       use('km3pipe')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('km3pipe.mplstyle')

    @patch('matplotlib.pyplot')
    def test_noargs_load_km3pipe_style(self, plt_mock):
       use()
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('km3pipe.mplstyle')

    @patch('matplotlib.pyplot')
    def test_poster_style(self, plt_mock):
       use('poster')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('km3pipe-poster.mplstyle')

    @patch('matplotlib.pyplot')
    def test_notebook_style(self, plt_mock):
       use('notebook')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('km3pipe-notebook.mplstyle')

    @patch('matplotlib.pyplot')
    def test_talk_style(self, plt_mock):
       use('talk')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('km3pipe-talk.mplstyle')

    @patch('matplotlib.pyplot')
    def test_alba_style(self, plt_mock):
       use('alba')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('alba.mplstyle')

    @patch('matplotlib.pyplot')
    def test_jonas_style(self, plt_mock):
       use('jonas-phd')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('jonas-phd.mplstyle')

    @patch('matplotlib.pyplot')
    def test_jvs_style(self, plt_mock):
       use('jvs')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('jvs.mplstyle')

    @patch('matplotlib.pyplot')
    def test_moritz_style(self, plt_mock):
       use('moritz')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('moritz.mplstyle')

    @patch('matplotlib.pyplot')
    def test_serifs_style(self, plt_mock):
       use('serifs')
       args, kwargs = plt_mock.style.use.call_args_list[0]
       assert args[0].endswith('serifs.mplstyle')
