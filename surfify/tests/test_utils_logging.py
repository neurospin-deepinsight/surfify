# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import unittest
import os
import tempfile
import torch
from surfify.utils import get_logger, setup_logging, debug_msg


class TestUtilsLogging(unittest.TestCase):
    """ Test logging.
    """
    def setUp(self):
        """ Setup test.
        """
        self.tensor = torch.zeros((10, 10))

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_level(self):
        """ Test wrong logging level.
        """
        self.assertRaises(ValueError, setup_logging, level="bad")

    def test_logging(self):
        """ Test logging functions.
        """
        setup_logging(level="info")
        logger = get_logger()
        logger.info(debug_msg("test", self.tensor))
        setup_logging(level="debug")

    def test_filelogging(self):
        """ Test logging functions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logfile = os.path.join(tmpdir, "log.txt")
            setup_logging(level="debug", logfile=logfile)
        setup_logging(level="debug")


if __name__ == "__main__":

    from surfify.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
