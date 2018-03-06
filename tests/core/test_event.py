"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-01-22
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest

from wc_sim.core.event import Event
from wc_sim.core.simulation_message import SimulationMessage
from wc_utils.util.misc import most_qual_cls_name, round_direct
from wc_utils.util.list import elements_to_str
from tests.core.example_simulation_objects import (ALL_MESSAGE_TYPES, TEST_SIM_OBJ_STATE,
    ExampleSimulationObject)
from tests.core.some_message_types import InitMsg


class TestEvent(unittest.TestCase):

    def test_event_inequalities(self):
        example_sim_obj_a = ExampleSimulationObject('a')
        example_sim_obj_b = ExampleSimulationObject('b')

        # test Events with different event times
        e1 = Event(0, 1, example_sim_obj_a, example_sim_obj_a, InitMsg())
        e2 = Event(0, 2, example_sim_obj_a, example_sim_obj_b, InitMsg())
        self.assertTrue(e1 < e2)
        self.assertFalse(e1 > e2)
        self.assertTrue(e1 <= e1)
        self.assertTrue(e1 <= e2)
        self.assertTrue(e2 > e1)
        self.assertFalse(e2 < e1)
        self.assertTrue(e1 >= e1)
        self.assertTrue(e2 >= e1)

        # test Events with equal event times and different recipients
        e3 = Event(0, 1, example_sim_obj_a, example_sim_obj_b, InitMsg())
        self.assertTrue(e1 < e3)
        self.assertFalse(e1 > e3)
        self.assertTrue(e1 <= e1)
        self.assertTrue(e1 <= e3)
        self.assertTrue(e3 > e1)
        self.assertFalse(e3 < e1)
        self.assertTrue(e1 >= e1)
        self.assertTrue(e3 >= e1)

    def test_event_w_message(self):
        ds = 'docstring'
        attrs = ['attr1', 'attr2']
        class TestMsg(SimulationMessage):
            'docstring'
            attributes = ['attr1', 'attr2']
        vals = ['att1_val', 'att2_val']
        test_msg = TestMsg(*vals)
        times = (0, 1)
        SENDER = 'sender'
        RECEIVER = 'receiver'
        ev = Event(*(times + (ExampleSimulationObject(SENDER), ExampleSimulationObject(RECEIVER),
            test_msg)))

        # test headers
        self.assertEquals(Event.BASE_HEADERS, Event.header(as_list=True)[:-1])
        self.assertIn('\t'.join(Event.BASE_HEADERS), Event.header())
        self.assertEquals(Event.BASE_HEADERS, ev.custom_header(as_list=True)[:-len(attrs)])
        self.assertIn('\t'.join(Event.BASE_HEADERS), ev.custom_header())
        self.assertIn('\t'.join(attrs), ev.custom_header())
        data = list(times) + [SENDER, RECEIVER, TestMsg.__name__]

        # test data
        self.assertIn('\t'.join(elements_to_str(data)), ev.render())
        self.assertIn('\t'.join(elements_to_str(vals)), ev.render())
        self.assertEqual(data+vals, ev.render(as_list=True))
        self.assertIn('\t'.join(elements_to_str(data)), ev.render(annotated=True))
        self.assertEqual(data, ev.render(annotated=True, as_list=True)[:len(data)])
        self.assertIn('\t'.join(elements_to_str(vals)), str(ev))
        self.assertIn('TestMsg', str(ev))
        offset_times = (0.000001, 0.999999)
        ev_offset = Event(*(offset_times + (ExampleSimulationObject(SENDER), ExampleSimulationObject(RECEIVER),
            test_msg)))
        for t in offset_times:
            self.assertIn(round_direct(t), ev_offset.render(round_w_direction=True, as_list=True))
            self.assertIn(str(round_direct(t)), ev_offset.render(round_w_direction=True))
        self.assertEqual(data+vals, ev.render(as_list=True))

        class NoBodyMessage(SimulationMessage):
            """A message with no attributes"""
        ev2 = Event(0, 1, ExampleSimulationObject('sender'), ExampleSimulationObject('receiver'),
            NoBodyMessage())
        self.assertIn('\t'.join(Event.BASE_HEADERS), ev2.custom_header())
        self.assertIn('\t'.join([str(t) for t in times]), str(ev2))
