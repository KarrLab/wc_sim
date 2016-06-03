from __future__ import print_function

from Event import Event
import warnings
import logging
logger = logging.getLogger(__name__)

# control logging level with: logger.setLevel()
# this enables debug output: logging.basicConfig( level=logging.DEBUG )

"""
General-purpose simulation mechanisms, including the event queue for each simulation object and
the simulation scheduler.

Stores event list, and provides send and next events methods. Architected for an OO simulation that
is prepared to be parallelized. 

Created 2016/05/31
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

class SimulationEngine(object):
    """A simulation's static engine.

    Longer class information....
    # TODO(Arthur): more py doc

    Attributes:
        These are global attributes, since the SimulationEngine is static.

        time: A float containing the simulations's current time.
        simulation_objects: A dict of all the simulation objects, indexed by name
        # TODO(Arthur): when the number of objects in a simulation grows large, a list representation of the event_queues 
            will preform sub-optimally; use a priority queue instead, reordering the queue as event messages are sent and
            received
    """

    time=0.0
    # TODO(Arthur): optionally, start a simulation at another time
    simulation_objects={}


    @staticmethod
    def add_object( name, simulation_object ):
        """Add a simulation object to the simulation.
        
        """
        # TODO(Arthur): exception on name collision
        SimulationEngine.simulation_objects[ name ] = simulation_object 

    @staticmethod
    def remove_object( name ):
        """Remove a simulation object from the simulation.
        """
        # TODO(Arthur): also delete the object?
        del SimulationEngine.simulation_objects[ name ]

    @staticmethod
    def reset( ):
        """Reset the SimulationEngine.
        
        """
        SimulationEngine.time=0.0
        SimulationEngine.simulation_objects={}

    @staticmethod
    def message_queues( ):
        """Print the message queues.
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
    def simulate( end_time ):
        """Run the simulation; return number of events.
        
        Args:
            end_time: number; the time of the end of the simulation
            
        """
        events_processed = 0
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
                
            events_processed += 1
            logger.debug( " %f\t%s\t%s" , next_time, next_sim_obj.__class__.__name__, next_sim_obj.name )
            
            # dispatch object that's ready to execute next event
            SimulationEngine.time = next_time
            next_sim_obj.time = next_time
            next_sim_obj.handle_event( next_sim_obj.event_queue.next_events() )
            
        return events_processed
