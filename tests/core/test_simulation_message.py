'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-06-01
:Copyright: 2017-2018, Karr Lab
:License: MIT
'''

import unittest
import six
from wc_sim.core.simulation_message import SimulationMsgUtils, SimulationMessage

class TestSimulationMessage(unittest.TestCase):

    def test_utils(self):
        attrs = {'__slots__':['arg_1','arg_2']}
        SimMsgType = type('test', (SimulationMessage,), attrs)
        with self.assertRaises(ValueError) as context:
            SimMsgType()
        six.assertRegex(self, str(context.exception), "Constructor .*'test' expects 2 arg.*but 0 provided")
        t = SimMsgType(1, 'a')
        self.assertIn('1', str(t))
        self.assertIn("'a'", str(t))
        delattr(t, 'arg_2')
        self.assertIn("'undef'", str(t))

        with self.assertRaises(ValueError) as context:
            SimulationMsgUtils.create('T', '')
        self.assertIn('SimulationMessage docstring cannot be empty', str(context.exception))
