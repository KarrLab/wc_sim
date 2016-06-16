#!/usr/bin/env python

from __future__ import print_function
import unittest

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.Event import Event
from SequentialSimulator.core.SimulationEngine import SimulationEngine

class ExampleSimulationObject(SimulationObject):

    def __init__( self, name, debug=False):
        self.debug = debug
        super(ExampleSimulationObject, self).__init__( name )

    def handle_event( self, event_list ):
        # print events
        if self.debug:
            self.print_event_queue( )
        # schedule event
        self.send_event( 2.0, self, 'test event sent by {} at {:6.3f}'.format( self.name, self.time ) )


class InteractingSimulationObject(SimulationObject):

    def __init__( self, name, debug=False):
        self.debug = debug
        super(InteractingSimulationObject, self).__init__( name )

    def handle_event( self, event_list ):
        if self.debug:
            self.print_event_queue( )
        # schedule events
        # send an event to each InteractingSimulationObject
        for obj in SimulationEngine.simulation_objects.values():
            self.send_event( 2.0, obj, 'test event sent by {} to {} at {:6.3f}'.format( self.name, obj.name, self.time ) )


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

    def test_multi_object_simulation(self):
    
        # initial events
        for i in range( 1, 4):
            obj = ExampleSimulationObject(  obj_name( i ) )
            obj.send_event( 1.0, obj, 'init_msg' )
            obj.send_event( 2.0, obj, 'init_msg' )
        self.assertEqual( SimulationEngine.simulate( 5.0 ), 15 )

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
