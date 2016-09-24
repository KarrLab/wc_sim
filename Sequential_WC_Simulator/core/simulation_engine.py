from __future__ import print_function

import datetime

# configure logging
from .config import config_constants
from log.config.config import ConfigAll
debug_log = ConfigAll.setup_logger( config_constants )

from Sequential_WC_Simulator.core.event import Event
from Sequential_WC_Simulator.core.config.config import SimulatorConfig

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
        MessageTypesRegistry.senders[ sim_obj_name ] = tuple( [ x.__name__ for x in message_types ] )
    
    @staticmethod
    def set_receiver_priorities( sim_obj_name, message_priorities ):
        MessageTypesRegistry.receiver_priorities[ sim_obj_name ] = \
            tuple( [ x.__name__ for x in message_priorities ] )

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
    
    debugging_logger = debug_log.get_logger( 'wc.debug.file' )

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
        """Return a string listing all message queues in the simulation.
        """
        data = ['Event queues at {:6.3f}'.format( SimulationEngine.time )]
        for sim_obj in SimulationEngine.simulation_objects.values():
            data.append( sim_obj.name + ':' )
            if sim_obj.event_queue.event_heap:
                data.append( Event.header() )
                data.append( str( sim_obj.event_queue ) )
            else:
                data.append( 'Empty event queue' )
            data.append( '' )
        return '\n'.join(data)

    @staticmethod
    def print_simulation_state( ):

        SimulationEngine.debugging_logger.debug( ' ' + '\t'.join( ['Sender', 'Message types sent'] ) )
        for sender in MessageTypesRegistry.senders:
            SimulationEngine.debugging_logger.debug( ' ' + sender + '\t' + 
                ', '.join( MessageTypesRegistry.senders[sender] ) )

        SimulationEngine.debugging_logger.debug( ' ' + '\t'.join( ['Receiver', 'Message types by priority'] ) )
        for receiver in MessageTypesRegistry.receiver_priorities:
            SimulationEngine.debugging_logger.debug( ' ' + sender + '\t' + 
                ', '.join( MessageTypesRegistry.receiver_priorities[receiver] ) )

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
        
        # write header to a plot log
        # plot logging is controlled by configuration files pointed to by config_constants and by env vars
        plotting_logger = debug_log.get_logger( 'wc.plot.file' )
        plotting_logger.debug( '# {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) )

        handle_event_invocations = 0
        SimulationEngine.debugging_logger.debug( ' ' + "\t".join( 'Time Object_type Object_name'.split() ) )
        while SimulationEngine.time <= end_time:
        
            # get the earliest next event in the simulation
            next_time = float('inf')
            for sim_obj in SimulationEngine.simulation_objects.values():
                
                if sim_obj.event_queue.next_event_time() < next_time:
                    next_time = sim_obj.event_queue.next_event_time()
                    next_sim_obj = sim_obj
                    
            if float('inf') == next_time:
                SimulationEngine.debugging_logger.debug( " No events remain" )
                break
                
            if end_time < next_time:
                SimulationEngine.debugging_logger.debug( " End time exceeded" )
                break
                
            handle_event_invocations += 1
            SimulationEngine.debugging_logger.debug( " {:6.2f}\t{}\t{}".format( next_time, 
                next_sim_obj.__class__.__name__, next_sim_obj.name ) )
            
            # TODO(Arthur): IMPORTANT: test situation when multiple objects have events at minimum simulation time
            # dispatch object that's ready to execute next event
            SimulationEngine.time = next_time
            # This assertion cannot be violoated unless init message sent to negative time or objects decrease their time
            assert next_sim_obj.time <= next_time, ("Dispatching '{}', but find object time "
            "{} > event time {}.".format( next_sim_obj.name, next_sim_obj.time, next_time ) )

            next_sim_obj.time = next_time
            next_sim_obj.handle_event( next_sim_obj.event_queue.next_events() )
            
        return handle_event_invocations
