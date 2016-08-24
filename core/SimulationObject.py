from __future__ import print_function

"""
Base class for simulation objects. 

Created 2016/06/01
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

from copy import deepcopy
import heapq
import warnings
# TODO(Arthur): remove old logging commands & comments; use new loggers
import logging
logger = logging.getLogger(__name__)
# control logging level with: logger.setLevel()
# this enables debug output: logging.basicConfig( level=logging.DEBUG )

from Sequential_WC_Simulator.core.Event import Event
from Sequential_WC_Simulator.core.SimulationEngine import (SimulationEngine, MessageTypesRegistry)
from Sequential_WC_Simulator.core.config import SimulatorConfig

class EventQueue(object):
    """A simulation object's event queue.
    
    Stores a heap of an object's events, with each event in a tuple (time, event). The heap's a 'min heap', with the
        event with the smallest time at the root.

    Attributes:
        event_heap: The object's heap of events, sorted by event time; has O(nlogn) get earliest and insert event.
    """

    def __init__( self ):
        self.event_heap=[]


    def schedule_event( self, send_time, receive_time, sending_object, receiving_object, event_type,
        event_body=None ):
        """Insert an event in this event queue, scheduled to execute at receive_time.
        
        Object X sends an event to object Y by invoking
            Y.event_queue.send_event( send_time, receive_time, X, Y, event_type, <event_body=value> )
        
        Args:
            send_time: number; the simulation time at which the message was generated (sent)
            receive_time: number; the simulation time at which the receiving_object will execute the event
            sending_object: object; the object sending the event
            receiving_object: object; the object that will receive the event
            event_type: class; the type of the message; the class' name is sent with the message,
                as messages should not contain references
            event_body: object; an object containing the body of the event
            
        Raises:
            ValueError: if receive_time < send_time
        """

        # ensure send_time <= receive_time
        # send_time == receive_time can cause loops, but the application programmer is responsible for avoiding them
        if receive_time < send_time:
            raise ValueError( "receive_time < send_time in schedule_event(): {} < {}".format( 
                str( receive_time ), str( send_time ) ) )
            
        event = Event( send_time, receive_time, sending_object, receiving_object, event_type, event_body )
        """
        # TODO(Arthur): optionally log the scheduling of an event
        """
        heapq.heappush( self.event_heap, ( receive_time, event ) )
    
    def next_event_time( self ):
        """Get the time of the next event.
        
        Return:
            The time of the next event. Return infinity if there is no next event.
        """
        if not self.event_heap:
            return float('inf')

        return self.event_heap[0][0]    # time of the 1st event in the heap
        
        
    def next_events( self ):
        """Get the list of next event(s). Events are provided in a list because multiple events may 
        have the same simultion time, and they must be provided to the simulation object as a unit.
        
        Handle 'ties' properly. That is, since an object may receive multiple events 
        with the same event_time (aka receive_time), pass them all to the object in a list.
        (In a Time Warp simulation, the list would need to be ordered by some deterministic
        criteria based on its contents.)

        Return:
            A list of next event(s), which is empty if no events are available.
        """
        # TODO(Arthur): in a Time Warp simulation, order the list by some deterministic criteria
        # TODO(Arthur): based on its contents; see David's lecture on this

        if not self.event_heap:
            return []

        events = []
        (now, an_event) = heapq.heappop( self.event_heap )
        events.append( an_event )
        while self.event_heap and now == self.next_event_time():    # time of the 1st event in the heap
            (ignore, an_event) = heapq.heappop( self.event_heap )
            events.append( an_event )
        return events
        
    @staticmethod
    def event_list_to_string( event_list ):
        '''return event_list members as a table; formatted as a multi-line, tab-separated string'''
        # TODO(Arthur): since event_list is a heap, the printed list is not in event order; sort before printing
        return "\n".join( [event.__str__() for event in event_list] )

    def __str__( self ):
        '''return event queue members as a table'''
        return "\n".join( [event.__str__() for (time, event) in self.event_heap] )


class SimulationObject(object):
    """Base class for simulation objects.
    
    SimulationObject is a base class for all simulations objects. It provides basic functionality:
    the object's name (which must be unique), it's simulation time, a queue of received events, and a send_event() method.

    Attributes:
        name: A string with the simulation object's name.
        time: A float containing the simulation object's current simulation time.
        event_queue: The object's EventQueue.
        plot_output: A boolean, indicating whether to print events, formatted for later plotting
        num_events: int; number of events processed
        # TODO(Arthur): use Python logging for printing the event_queue
        # TODO(Arthur): important: in general, replace print statements with logging
    """

    def __init__( self, name, plot_output=False):
        """Initialize a SimulationObject. 
        
        Create its event queue, initialize its name, and set its start time to 0.
        
        Args:
            name: string; the object's unique name, used as a key in the dict of objects
            plot_output: boolean; if true, print a line for each executed event, suitable for plotting
        """
        self.event_queue = EventQueue()
        self.name = name
        self.plot_output = plot_output
        self.time = 0.0
        self.num_events = 0
        self.register()
        
    def register( self ):
        """Register this simulation object with the simulation.
        
        Each class derived from SimulationObject must register() itself before the simulation starts.
        """
        SimulationEngine.add_object( self.name, self ) 

    def send_event( self, delay, receiving_object, event_type, event_body=None, copy=True ):
        """Send a simulation event message. 
        
        Args:
            delay: number; the simulation delay at which the receiving_object should execute the event
            receiving_object: object; the object that will receive the event
            event_type: string; a string type for the event, used by objects and in debugging output
            event_body: object; an optional object containing the body of the event
            copy: boolean; if True, copy the event_body;
                True by default as a safety measure to avoid unexpected changes to shared objects; set False to optimize
            
        Raises:
            ValueError: if delay < 0
            ValueError: if the sending object type is not registered to send a message type
            ValueError: if the receiving simulation object type is not registered to receive the message type
            
        """
        if delay < 0:
            raise ValueError( "delay < 0 in send_event(): {}".format( str( delay ) ) )
            
        # TODO(Arthur): replace this with event_type_name = event_type_name.__name__; the isinstance()
        # just accommodates some old tests
        event_type_name = event_type
        if not isinstance( event_type_name, str ):
            event_type_name = event_type_name.__name__
        
        # check that the sending object type is registered to send the message type
        if not event_type_name in MessageTypesRegistry.senders[ self.__class__.__name__ ]:
            raise ValueError( "'{}' simulation objects not registered to send '{}' messages".format( 
                self.__class__.__name__, event_type_name ) )

        # check that the receiving simulation object type is registered to receive the message type
        if not event_type_name in MessageTypesRegistry.receiver_priorities[ receiving_object.__class__.__name__ ]:
            raise ValueError( "'{}' simulation objects not registered to receive '{}' messages".format( 
                receiving_object.__class__.__name__, event_type_name ) )
            
        event_body_copy = None
        if event_body and copy:
            event_body_copy = deepcopy( event_body )
        
        receiving_object.event_queue.schedule_event( self.time, self.time + delay, self,
            receiving_object, event_type_name, event_body )
        logger.debug( ": (%s, %f) -> (%s, %f): %s" , self.name, self.time, receiving_object.name, self.time + delay, 
            event_type )
        

    def handle_event( self, event_list ):
        """Handle a simulation event, which may involve multiple event messages.
        
        Each class derived from SimulationObject must implement handle_event( ).

        Attributes:
            event_list: A non-empty list of event messages in the event

        Raises:
            ValueError: if some event message in event_list has an invalid type
        """
        
        self.num_events += 1
        
        # check for messages with invalid types
        # TODO(Arthur): do this checking at send time, probably in SimulationObject.send_event()
        invalid_types = (set( map( lambda x: x.event_type, event_list ) ) - 
            set( MessageTypesRegistry.receiver_priorities[ self.__class__.__name__ ] ))
        if len( invalid_types ):
            raise ValueError( "Error: invalid event event_type(s) '{}' in event_list:\n{}".format( 
                ', '.join( list( invalid_types ) ),
                '\n'.join( [ str( ev_msg ) for ev_msg in event_list ] ) ) )

        # sort event_list by type priority, anticipating non-deterministic arrival order in a parallel implementation
        # this scales for arbitrarily many message types
        # TODO(Arthur): unittest this code
        event_list = sorted( event_list, 
            key=lambda event:
            MessageTypesRegistry.receiver_priorities[ self.__class__.__name__ ].index( event.event_type ) )

        # TODO(Arthur): write this to a plot input log
        # print events for plotting by plotSpaceTimeDiagram.py
        if self.plot_output:
            for event in event_list:
                print( event )

    def print_event_queue( self ):
        print(  )
        print( self.event_queue_to_str() )

    def event_queue_to_str( self ):
        eq = '{} at {:5.3f}\n'.format( self.name, self.time ) 
        if self.event_queue.event_heap:
            eq += Event.header() + '\n' + str(self.event_queue) + '\n' 
        else:
            eq += 'Empty event queue'
        return eq
