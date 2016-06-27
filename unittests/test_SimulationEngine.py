#!/usr/bin/env python

from __future__ import print_function
import unittest

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.Event import Event
from SequentialSimulator.core.SimulationEngine import SimulationEngine
from SequentialSimulator.multialgorithm.MessageTypes import MessageTypes

class ExampleSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = 'init_msg'.split()
    MessageTypes.set_sent_message_types( 'ExampleSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypes.set_receiver_priorities( 'ExampleSimulationObject', ALL_MESSAGE_TYPES )

    def __init__( self, name, debug=False):
        self.debug = debug
        super(ExampleSimulationObject, self).__init__( name )

    def handle_event( self, event_list ):
        # print events
        if self.debug:
            self.print_event_queue( )
        # schedule event
        self.send_event( 2.0, self, 'init_msg' )

class InteractingSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = 'init_msg'.split()
    MessageTypes.set_sent_message_types( 'InteractingSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypes.set_receiver_priorities( 'InteractingSimulationObject', ALL_MESSAGE_TYPES )

    def __init__( self, name, debug=False):
        self.debug = debug
        super(InteractingSimulationObject, self).__init__( name )

    def handle_event( self, event_list ):
        if self.debug:
            self.print_event_queue( )
        # schedule events
        # send an event to each InteractingSimulationObject
        for obj in SimulationEngine.simulation_objects.values():
            self.send_event( 2.0, obj, 'init_msg'.format( self.name, obj.name, self.time ) )


NAME_PREFIX = 'sim_obj'

def obj_name( i ):
    return '{}_{}'.format( NAME_PREFIX, i )

class TestSimulationEngine(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    # @unittest.skip("demonstrating skipping")
    def test_one_object_simulation(self):
    
        # initial event
        obj = ExampleSimulationObject(  obj_name( 1 ) )
        obj.send_event( 1.0, obj, 'init_msg' )
        obj.send_event( 2.0, obj, 'init_msg' )
        self.assertEqual( SimulationEngine.simulate( 5.0 ), 5 )

    def test_simulation_engine_exception(self):
    
        # initial event
        obj = ExampleSimulationObject(  obj_name( 1 ) )
        event_queue = obj.event_queue
        event_queue.schedule_event( -1, -1, obj, obj, 'init_msg' )
        with self.assertRaises(AssertionError) as context:
            SimulationEngine.simulate( 5.0 )
        self.assertIn( 'find object time', context.exception.message )
        self.assertIn( '> event time', context.exception.message )

    def test_multi_object_simulation(self):
    
        # initial events
        for i in range( 1, 4):
            obj = ExampleSimulationObject(  obj_name( i ) )
            obj.send_event( 1.0, obj, 'init_msg' )
            obj.send_event( 2.0, obj, 'init_msg' )
        self.assertEqual( SimulationEngine.simulate( 5.0, debug=True ), 15 )

    def test_multi_interacting_object_simulation(self):
    
        # initial events
        for i in range( 1, 4):
            obj = InteractingSimulationObject(  obj_name( i ) )
            obj.send_event( 1.0, obj, 'init_msg' )
            obj.send_event( 2.0, obj, 'init_msg' )
        self.assertEqual( SimulationEngine.simulate( 5.0 ), 15 )


if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass
