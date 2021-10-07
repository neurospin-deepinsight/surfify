# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
from logging import warning
import sys
import warnings
import unittest
from io import StringIO
from surfify.utils.io import HidePrints


class TestUtilsSampling(unittest.TestCase):
    """ Test spherical sampling.
    """
    def setUp(self):
        """ Setup test.
        """
        pass

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_hideprints(self):
        """ Test HidePrints class.
        """
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        with HidePrints():
            print("hey")
        sys.stdout = old_stdout
        self.assertTrue(mystdout.getvalue() == "")

        old_stderr = sys.stderr
        sys.stderr = mystderr = StringIO()

        with HidePrints():
            print("hey", file=sys.stderr)
        sys.stderr = old_stderr
        self.assertTrue(mystderr.getvalue() == "hey\n")

        old_stderr = sys.stderr
        sys.stderr = mystderr = StringIO()

        with HidePrints(hide_err=True):
            print("hey", file=sys.stderr)
        sys.stderr = old_stderr
        self.assertTrue(mystderr.getvalue() == "")


if __name__ == "__main__":
    from surfify.utils import setup_logging

    setup_logging(level="debug")
    unittest.main()
