"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-02-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import warnings

from argparse import Namespace
from examples.random_state_variable import RunRandomStateVariableSimulation
from wc_sim.core.debug_logs import config


class TestRandomStateVariableSimulation(unittest.TestCase):

    def setUp(self):
        # turn off console logging
        self.console_level = config['debug_logs']['loggers']['wc.debug.console']['level']
        config['debug_logs']['loggers']['wc.debug.console']['level'] = 'ERROR'
        warnings.simplefilter("ignore")

    def tearDown(self):
        # restore console logging
        config['debug_logs']['loggers']['wc.debug.console']['level'] = self.console_level

    def test_random_state_variable_simulation(self):
        args = Namespace(initial_state=3, end_time=10, output=False)
        self.assertTrue(0<RunRandomStateVariableSimulation.main(args))
