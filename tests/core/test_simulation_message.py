'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-06-01
:Copyright: 2017-2018, Karr Lab
:License: MIT
'''

import unittest
import six
from wc_sim.core.simulation_message import SimulationMessageFactory, SimulationMessage
from wc_sim.core.errors import SimulatorError


class TestSimulationMessage(unittest.TestCase):

    def test_utils(self):
        attributes = ['arg_1','arg_2']
        attrs = {'__slots__':attributes}
        SimMsgType = type('test', (SimulationMessage,), attrs)
        with self.assertRaises(SimulatorError) as context:
            SimMsgType()
        six.assertRegex(self, str(context.exception),
            "Constructor .*'test' expects 2 arg.*but 0 provided")
        vals = [1, 'a']
        t = SimMsgType(*vals)
        self.assertIn(str(vals[0]), str(t))
        self.assertIn(vals[1], str(t))
        for attr in attributes:
            self.assertIn(attr, t.attrs())
            self.assertIn(attr, t.header())
        self.assertEqual('\t'.join([str(v) for v in vals]), t.delimited())
        delattr(t, 'arg_2')
        self.assertIn(str(None), str(t))


class TestSimulationMessageFactory(unittest.TestCase):

    def test_simulation_message_factory(self):
        attrs = ['attr1', 'attr2']
        ds = 'docstring'
        TestMsg = SimulationMessageFactory.create('TestMsg', ds, attrs)
        test_msg = TestMsg('att1_val', 'att2_val')
        self.assertEqual(test_msg.__doc__, ds)
        for attr in attrs:
            self.assertTrue(hasattr(test_msg, attr))

        TestMsg2 = SimulationMessageFactory.create('TestMsg2', ds)
        test_msg2 = TestMsg2()
        for attr in attrs:
            self.assertFalse(hasattr(test_msg2, attr))

        with self.assertRaises(SimulatorError) as context:
            SimulationMessageFactory.create('', '')
        self.assertIn('SimulationMessage name cannot be empty', str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            SimulationMessageFactory.create('T', '')
        self.assertIn('SimulationMessage docstring cannot be empty', str(context.exception))
