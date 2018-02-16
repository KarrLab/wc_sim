"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-01-22
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import six

from wc_sim.core.event import Event
from wc_sim.core.simulation_message import SimulationMessageFactory, SimulationMessage
from tests.core.example_simulation_objects import (ALL_MESSAGE_TYPES, TEST_SIM_OBJ_STATE,
    ExampleSimulationObject)
from wc_utils.util.misc import most_qual_cls_name


class TestEvent(unittest.TestCase):

    def test_event_inequalities(self):
        e1 = Event(0, 1, object(), object(), object())
        e2 = Event(0, 2, object(), object(), object())
        self.assertTrue(e1 < e2)
        self.assertFalse(e1 > e2)
        self.assertTrue(e1 <= e1)
        self.assertTrue(e1 <= e2)
        self.assertTrue(e2 > e1)
        self.assertFalse(e2 < e1)
        self.assertTrue(e1 >= e1)
        self.assertTrue(e2 >= e1)

    def test_event_w_message(self):
        ds = 'docstring'
        attrs = ['attr1', 'attr2']
        TestMsg = SimulationMessageFactory.create('TestMsg', ds, attrs)
        vals = ['att1_val', 'att2_val']
        test_msg = TestMsg(*vals)
        times = [0, 1]
        ev = Event(*times, ExampleSimulationObject('sender'), ExampleSimulationObject('receiver'),
            test_msg)
        self.assertIn('\t'.join(Event.BASE_HEADERS), Event.header())
        self.assertIn('\t'.join(Event.BASE_HEADERS), ev.custom_header())
        self.assertIn('\t'.join(attrs), ev.custom_header())
        self.assertIn('\t'.join(vals), str(ev))
        self.assertIn('TestMsg', str(ev))

        NoBodyMessage = SimulationMessageFactory.create('NoBodyMessage',
            """A message with no attributes""")
        ev2 = Event(0, 1, ExampleSimulationObject('sender'), ExampleSimulationObject('receiver'),
            NoBodyMessage())
        self.assertIn('\t'.join(Event.BASE_HEADERS), ev2.custom_header())
        self.assertIn('\t'.join([str(t) for t in times]), str(ev2))
