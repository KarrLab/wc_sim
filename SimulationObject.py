from __future__ import print_function

from Event import Event
from SimulationEngine import SimulationEngine

import heapq
import warnings
import logging
logger = logging.getLogger(__name__)

# control logging level with: logger.setLevel()
# this enables debug output: logging.basicConfig( level=logging.DEBUG )

"""
Base class for simulation objects. 

Created 2016/06/01
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

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
            Y.event_queue.send_event( send_time, receive_time, X, Y, event_type )
        
        Args:
            send_time: number; the simulation time at which the message was generated (sent)
            receive_time: number; the simulation time at which the receiving_object will execute the event
            sending_object: object; the object sending the event
            receiving_object: object; the object that will receive the event
            event_type: string; a string type for the event, used by objects and in debugging output
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
        # TODO(Arthur): logging support
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
        # TODO(Arthur): order the list by some deterministic criteria based on its contents

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
    """

    # TODO(Arthur): optionally start the simulation at a time other than 0
    def __init__( self, name, start_time=None):
        self.event_queue = EventQueue()

        self.name = name

        self.time = 0.0
        if start_time:
            self.time = start_time
            
        self.register()
        
    def register( self ):
        """Register this simulation object with the simulation.
        
        Each class derived from SimulationObject must register() itself before the simulation starts.
        """
        SimulationEngine.add_object( self.name, self ) 

    def send_event( self, delay, receiving_object, event_type, event_body=None ):
        """Send a simulation event message. 
        
        Args:
            delay: number; the simulation delay at which the receiving_object should execute the event
            receiving_object: object; the object that will receive the event
            event_type: string; a string type for the event, used by objects and in debugging output
            event_body: object; an optional object containing the body of the event
            
        Raises:
            ValueError: if delay < 0
        """
        if delay < 0:
            raise ValueError( "delay < 0 in send_event(): {}".format( str( delay ) ) )
            
        receiving_object.event_queue.schedule_event( self.time, self.time + delay, self,
            receiving_object, event_type, event_body )
        logger.debug( ": (%s, %f) -> (%s, %f): %s" , self.name, self.time, receiving_object.name, self.time + delay, event_type )
        

    def handle_event( self, event_list ):
        """Handle a simulation event, which may involve multiple event messages.
        
        Each class derived from SimulationObject must implement handle_event( ).

        Attributes:
            event_list: A non-empty list of event messages in the event
        """
        pass

    def print_event_queue( self ):
        print(  )
        print( '{} at {:5.3f}'.format( self.name, self.time ) )
        if self.event_queue.event_heap:
            print( Event.header() )
            print( self.event_queue )
        else:
            print( 'Empty event queue' )
    
        
        
