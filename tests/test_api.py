""" Tests API

:Author: Jonathan Karr <jonrkarr@gmail.com>
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
        self.assertIsInstance(wc_sim.core, types.ModuleType)
        self.assertIsInstance(wc_sim.core.config, types.ModuleType)
        self.assertIsInstance(wc_sim.core.config.get_config, types.FunctionType)
        self.assertIsInstance(wc_sim.log, types.ModuleType)
        self.assertIsInstance(wc_sim.log.checkpoint, types.ModuleType)
        self.assertIsInstance(wc_sim.multialgorithm, types.ModuleType)
        self.assertIsInstance(wc_sim.multialgorithm.distributed_properties, types.ModuleType)
        self.assertIsInstance(wc_sim.multialgorithm.config, types.ModuleType)
        self.assertIsInstance(wc_sim.multialgorithm.config.get_config, types.FunctionType)