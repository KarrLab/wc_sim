from __future__ import print_function
import unittest
import random

from wc_sim.core.simulation_object import EventQueue, SimulationObject
from wc_sim.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from tests.some_message_types import InitMsg, Test1
from wc_utils.util.misc import most_qual_cls_name

class ExampleSimulationObject(SimulationObject):
    pass

ALL_MESSAGE_TYPES = [InitMsg, Test1]
MessageTypesRegistry.set_sent_message_types( ExampleSimulationObject, ALL_MESSAGE_TYPES )
MessageTypesRegistry.set_receiver_priorities( ExampleSimulationObject, ALL_MESSAGE_TYPES )


class TestSimulationObject(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()
        self.o1 = ExampleSimulationObject('o1')
        self.o2 = ExampleSimulationObject('o2')

    def test_event(self):
        times=[1.0, 2.0, 0.5]
        for t in times:
            self.o1.send_event( t, self.o2, Test1 )

        tmp = sorted(times)
        while self.o2.event_queue.next_event_time() < float('inf'):
            self.assertEqual( self.o2.event_queue.next_event_time(), tmp.pop(0) )
            self.o2.event_queue.next_events(self.o2)
        self.assertEqual( self.o2.event_queue.next_events(self.o2), [] )

    def test_event_ties(self):
        self.o1.send_event(0, self.o2, Test1 )
        self.o1.send_event(2, self.o2, InitMsg )

        num=10
        self.o1.send_event(1, self.o2, InitMsg )
        for i in range(num):
            if random.choice([True, False]):
                self.o1.send_event(1, self.o2, Test1 )
            else:
                self.o1.send_event(1, self.o2, InitMsg )

        self.assertEqual( self.o2.event_queue.next_event_time(), 0)
        event_list = self.o2.event_queue.next_events(self.o2)
        self.assertEqual(event_list[0].event_time, 0)

        self.assertEqual( self.o2.event_queue.next_event_time(), 1)
        event_list = self.o2.event_queue.next_events(self.o2)
        # all InitMsg messages come before any Test1 message,
        # and at least 1 InitMsg message exists
        expected_type = InitMsg
        switched = False
        for event in event_list:
            if not switched and event.event_type == most_qual_cls_name(Test1):
                expected_type = Test1
            self.assertEqual(event.event_type, most_qual_cls_name(expected_type))

        self.assertEqual( self.o2.event_queue.next_event_time(), 2)

    def test_exceptions(self):
        delay = -1.0
        with self.assertRaises(ValueError) as context:
            self.o1.send_event( delay, self.o2, Test1 )
        self.assertEqual( str(context.exception),
            "delay < 0 in send_event(): {}".format( str( delay ) ) )

        eq = EventQueue()
        with self.assertRaises(ValueError) as context:
            eq.schedule_event( 2, 1, None, None, '' )
        self.assertEqual( str(context.exception),
            "receive_time < send_time in schedule_event(): {} < {}".format( 1, 2 ) )
