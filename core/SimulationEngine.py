from __future__ import print_function

import warnings
import logging
logger = logging.getLogger(__name__)
# control logging level with: logger.setLevel()
# this enables debug output: logging.basicConfig( level=logging.DEBUG )

from Event import Event
from Sequential_WC_Simulator.core.config import SimulatorConfig

"""
General-purpose simulation mechanisms, including the event queue for each simulation object and
the simulation scheduler.

Stores event list, and provides send and next events methods. Architected for an OO simulation that
is prepared to be parallelized. 

Created 2016/05/31
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

class MessageTypesRegistry( object ):
    """A registry of message types, which is used to check that objects are sending and receiving
    the right types of messages.
    """

    senders = {}
    receiver_priorities = {}
    
    @staticmethod
    def set_sent_message_types( sim_obj_name, message_types ):
        """Register message types that sim_obj_name will send.
        
        Store the types as strings, because they are also used in event messages, which 
            should not contain references.
        """
        MessageTypesRegistry.senders[ sim_obj_name ] = map( lambda x: x.__name__, message_types )
    
    @staticmethod
    def set_receiver_priorities( sim_obj_name, message_priorities ):
        MessageTypesRegistry.receiver_priorities[ sim_obj_name ] = \
            map( lambda x: x.__name__, message_priorities )

class SimulationEngine(object):
    """A simulation's static engine.
    
    A static, singleton class that contains and manipulates global simulation data.
    SimulationEngine registers all simulation objects; runs the simulation, scheduling objects to execute events
        in non-decreasing time order; and generates debugging output.
    
    Attributes:
        These are global attributes, since the SimulationEngine is static.

        time: A float containing the simulations's current time.
        simulation_objects: A dict of all the simulation objects, indexed by name
        # TODO(Arthur): when the number of objects in a simulation grows large, a list representation of the event_queues 
            will preform sub-optimally; use a priority queue instead, reordering the queue as event messages are sent and
            received
    """

    time=0.0
    simulation_objects={}


    @staticmethod
    def add_object( name, simulation_object ):
        """Add a simulation object to the simulation.
        
        Raises:
            ValueError: if name is in use
        """
        if name in SimulationEngine.simulation_objects:
            raise ValueError( "cannot register '{}', name already in use".format( name ) )
        SimulationEngine.simulation_objects[ name ] = simulation_object 

    @staticmethod
    def remove_object( name ):
        """Remove a simulation object from the simulation.
        """
        del SimulationEngine.simulation_objects[ name ]

    @staticmethod
    def reset( ):
        """Reset the SimulationEngine.
        
        Set time to 0, and remove all objects.
        """
        SimulationEngine.time=0.0
        SimulationEngine.simulation_objects={}

    @staticmethod
    def message_queues( ):
        """Print all message queues in the simulation.
        """
        print( 'Event queues at {:6.3f}'.format( SimulationEngine.time ) )
        for sim_obj in SimulationEngine.simulation_objects.values():
            print( sim_obj.name + ':' )
            if sim_obj.event_queue.event_heap:
                print( Event.header() )
                print( sim_obj.event_queue )
            else:
                print( 'Empty event queue' )
            print(  )

    @staticmethod
    def print_simulation_state( ):

        logger.debug( ' ' + '\t'.join( ['Sender', 'Message types sent'] ) )
        for sender in MessageTypesRegistry.senders.keys():
            logger.debug( ' ' + sender + '\t' + ', '.join( MessageTypesRegistry.senders[sender] ) )

        logger.debug( ' ' + '\t'.join( ['Receiver', 'Message types by priority'] ) )
        for receiver in MessageTypesRegistry.receiver_priorities.keys():
            logger.debug( ' ' + sender + '\t' + ', '.join( MessageTypesRegistry.receiver_priorities[receiver] ) )

    @staticmethod
    def simulate( end_time, debug=False ):
        """Run the simulation; return number of events.
        
        Args:
            end_time: number; the time of the end of the simulation
            
        Return:
            The number of times a simulation object executes handle_event(). This may be larger than the number
            of events sent, because simultaneous events are handled together.
        """
        # TODO(Arthur): IMPORTANT: add optional logical termation condition(s)
        
        if debug:
            SimulationEngine.print_simulation_state()
        
        handle_event_invocations = 0
        logger.debug( ' ' + "\t".join( 'Time Object_type Object_name'.split() ) )
        while SimulationEngine.time <= end_time:
        
            # SimulationEngine.message_queues()
            
            # get the earliest next event in the simulation
            next_time = float('inf')
            for sim_obj in SimulationEngine.simulation_objects.values():
                
                if sim_obj.event_queue.next_event_time() < next_time:
                    next_time = sim_obj.event_queue.next_event_time()
                    next_sim_obj = sim_obj
                    
            if float('inf') == next_time:
                logger.debug( " No events remain" )
                break
                
            if end_time < next_time:
                logger.debug( " End time exceeded" )
                break
                
            handle_event_invocations += 1
            logger.debug( " %f\t%s\t%s" , next_time, next_sim_obj.__class__.__name__, next_sim_obj.name )
            
            # TODO(Arthur): IMPORTANT: test situation when multiple objects have events at minimum simulation time
            # dispatch object that's ready to execute next event
            SimulationEngine.time = next_time
            # This assertion cannot be violoated unless init message sent to negative time or objects decrease their time
            assert next_sim_obj.time <= next_time, ("Dispatching '{}', but find object time "
            "{} > event time {}.".format( next_sim_obj.name, next_sim_obj.time, next_time ) )

            next_sim_obj.time = next_time
            next_sim_obj.handle_event( next_sim_obj.event_queue.next_events() )
            
        return handle_event_invocations
