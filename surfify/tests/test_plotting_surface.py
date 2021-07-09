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
from surfify.utils import icosahedron
from surfify.plotting import plot_trisurf


class TestPlottingSurface(unittest.TestCase):
    """ Test surface plotting.
    """
    def setUp(self):
        """ Setup test.
        """
        self.vertices, self.triangles = icosahedron(order=2)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_plot_trisurf(self):
        """ Test plot_trisurf function.
        """
        plot_trisurf(self.vertices, self.triangles)
        plot_trisurf(self.vertices, self.triangles, is_label=True)


if __name__ == "__main__":

    from surfify.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
