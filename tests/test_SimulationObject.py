from __future__ import print_function
import unittest

from wc_sim.core.simulation_object import (EventQueue, SimulationObject)
from wc_sim.core.simulation_engine import (SimulationEngine, MessageTypesRegistry)
from tests.some_message_types import init_msg, test1

class ExampleSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = [init_msg, test1]
    MessageTypesRegistry.set_sent_message_types( 'ExampleSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypesRegistry.set_receiver_priorities( 'ExampleSimulationObject', ALL_MESSAGE_TYPES )


class TestSimulationObject(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()
        self.o1 = ExampleSimulationObject('o1')
        self.o2 = ExampleSimulationObject('o2')

    def test_event(self):
        times=[1.0, 2.0, 0.5]
        for t in times:
            self.o1.send_event( t, self.o2, 'test1' )
        
        tmp = sorted(times)
        while self.o2.event_queue.next_event_time() < float('inf'):
            self.assertEqual( self.o2.event_queue.next_event_time(), tmp.pop(0) )
            el = self.o2.event_queue.next_events()
        self.assertEqual( self.o2.event_queue.next_events(), [] )
    
    def test_event_ties(self):
        times=[1.0, 2.0, 1.0]
        for t in times:
            self.o1.send_event( t, self.o2, 'test1' )
        
        tmp = sorted(times[0:2])
        while self.o2.event_queue.next_event_time() < float('inf'):
            self.assertEqual( self.o2.event_queue.next_event_time(), tmp.pop(0) )
            el = self.o2.event_queue.next_events()
        self.assertEqual( self.o2.event_queue.next_events(), [] )

    def test_exceptions(self):
        delay = -1.0
        with self.assertRaises(ValueError) as context:
            self.o1.send_event( delay, self.o2, 'test1' )
        self.assertEqual( str(context.exception),
            "delay < 0 in send_event(): {}".format( str( delay ) ) )
        
        eq = EventQueue()
        with self.assertRaises(ValueError) as context:
            eq.schedule_event( 2, 1, None, None, '' )
        self.assertEqual( str(context.exception),
            "receive_time < send_time in schedule_event(): {} < {}".format( 1, 2 ) )
