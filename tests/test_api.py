""" Tests API

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-03-12
:Copyright: 2018, Karr Lab
:License: MIT
"""

import wc_sim
import types
import unittest


class ApiTestCase(unittest.TestCase):
    def test(self):
        self.assertIsInstance(wc_sim, types.ModuleType)
        self.assertIsInstance(wc_sim.log, types.ModuleType)
        self.assertIsInstance(wc_sim.config, types.ModuleType)
        self.assertIsInstance(wc_sim.distributed_properties, types.ModuleType)
        self.assertIsInstance(wc_sim.config.get_config, types.FunctionType)
