from __future__ import print_function
import unittest

from wc_sim.core.simulation_object import (EventQueue, SimulationObject)
from wc_sim.core.event import Event
from wc_sim.core.simulation_engine import (SimulationEngine, MessageTypesRegistry)
from tests.some_message_types import InitMsg

class ExampleSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = [InitMsg]
    MessageTypesRegistry.set_sent_message_types( 'ExampleSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypesRegistry.set_receiver_priorities( 'ExampleSimulationObject', ALL_MESSAGE_TYPES )

    def __init__( self, name, debug=False):
        self.debug = debug
        super(ExampleSimulationObject, self).__init__( name )

    def handle_event( self, event_list ):
        # schedule event
        self.send_event( 2.0, self, 'InitMsg' )

class InteractingSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = [InitMsg]
    MessageTypesRegistry.set_sent_message_types( 'InteractingSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypesRegistry.set_receiver_priorities( 'InteractingSimulationObject', ALL_MESSAGE_TYPES )

    def __init__( self, name, debug=False):
        self.debug = debug
        super(InteractingSimulationObject, self).__init__( name )

    def handle_event( self, event_list ):
        # schedule events
        # send an event to each InteractingSimulationObject
        for obj in SimulationEngine.simulation_objects.values():
            self.send_event( 2.0, obj, 'InitMsg'.format( self.name, obj.name, self.time ) )


NAME_PREFIX = 'sim_obj'

def obj_name( i ):
    return '{}_{}'.format( NAME_PREFIX, i )

class TestSimulationEngine(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    def test_one_object_simulation(self):
    
        # initial event
        obj = ExampleSimulationObject(  obj_name( 1 ) )
        obj.send_event( 1.0, obj, 'InitMsg' )
        obj.send_event( 2.0, obj, 'InitMsg' )
        self.assertEqual( SimulationEngine.simulate( 5.0 ), 5 )

    def test_simulation_engine_exception(self):
    
        # initial event
        obj = ExampleSimulationObject(  obj_name( 1 ) )
        event_queue = obj.event_queue
        event_queue.schedule_event( -1, -1, obj, obj, 'InitMsg' )
        with self.assertRaises(AssertionError) as context:
            SimulationEngine.simulate( 5.0 )
        self.assertIn( 'find object time', str(context.exception) )
        self.assertIn( '> event time', str(context.exception) )

    def test_multi_object_simulation(self):
    
        # initial events
        for i in range( 1, 4):
            obj = ExampleSimulationObject(  obj_name( i ) )
            obj.send_event( 1.0, obj, 'InitMsg' )
            obj.send_event( 2.0, obj, 'InitMsg' )
        self.assertEqual( SimulationEngine.simulate( 5.0 ), 15 )

    def test_multi_interacting_object_simulation(self):
    
        # initial events
        for i in range( 1, 4):
            obj = InteractingSimulationObject(  obj_name( i ) )
            obj.send_event( 1.0, obj, 'InitMsg' )
            obj.send_event( 2.0, obj, 'InitMsg' )
        self.assertEqual( SimulationEngine.simulate( 5.0 ), 15 )
