"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-06-01
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

import unittest
import six
import warnings

from wc_sim.core.simulation_message import SimulationMessage, SimulationMessageInterface
from wc_sim.core.errors import SimulatorError
from wc_utils.util.list import elements_to_str


class TestSimulationMessageInterface(unittest.TestCase):

    def test_utils(self):
        attributes = ['arg_1','arg_2']
        attrs = {'__slots__':attributes}
        SimMsgType = type('test', (SimulationMessageInterface,), attrs)
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
            self.assertIn(attr, t.header(as_list=True))
            self.assertIn(attr, t.values(annotated=True))
            self.assertIn(attr, t.values(annotated=True, separator=','))
        self.assertEqual(elements_to_str(vals), t.values(as_list=True))
        self.assertEqual('\t'.join(elements_to_str(vals)), t.values())
        delattr(t, 'arg_2')
        self.assertIn(str(None), str(t))


class ExampleSimulationMessage1(SimulationMessage):
    ' My docstring '
    attributes = ['attr1', 'attr2']


class ExampleSimulationMessage2(SimulationMessage):
    " docstring "
    pass


class TestSimulationMessageMeta(unittest.TestCase):

    def test_simulation_message_meta(self):
        self.assertTrue(issubclass(ExampleSimulationMessage1, SimulationMessage))
        with warnings.catch_warnings(record=True) as w:
            class BadSimulationMessage2(SimulationMessage):
                attributes = ['x']
            self.assertIn("definition does not contain a docstring", str(w[-1].message))
        warnings.simplefilter("ignore")

        self.assertEqual(ExampleSimulationMessage1.__doc__, 'My docstring')
        attr_vals = ('att1_val', 'att2_val')
        example_simulation_message = ExampleSimulationMessage1(*attr_vals)
        attrs = ['attr1', 'attr2']
        for attr,val in zip(attrs, attr_vals):
            self.assertEqual(getattr(example_simulation_message, attr), val)

        example_simulation_message2 = ExampleSimulationMessage2()
        self.assertEqual(example_simulation_message2.attrs(), [])
        self.assertEqual(example_simulation_message2.header(), None)

        with self.assertRaises(SimulatorError) as context:
            class BadSimulationMessage1(SimulationMessage):
                attributes = [2.5]
        self.assertIn('must be a list of strings', str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            class BadSimulationMessage2(SimulationMessage):
                attributes = ['x', 'y', 'x']
        self.assertIn('contains duplicates', str(context.exception))
