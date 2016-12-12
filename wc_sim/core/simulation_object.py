""" Base class for simulation objects.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-06-01
:Copyright: 2016, Karr Lab
:License: MIT
"""

from copy import deepcopy
import heapq
import warnings
import six

# configure logging
from .debug_logs import logs as debug_logs

from wc_sim.core.event import Event
from wc_sim.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from wc_utils.util.misc import most_qual_cls_name

class EventQueue(object):
    """A simulation object's event queue.

        Stores a heap of an object's events, with each event in a tuple (time, event). The heap is
        a 'min heap', with the event with the smallest time at the root.

    Attributes:
        event_heap: The object's heap of events, sorted by event time; has O(nlogn) get earliest and insert event.
    """

    def __init__(self):
        self.event_heap=[]

    def schedule_event(self, send_time, receive_time, sending_object, receiving_object, event_type,
        event_body=None):
        """Insert an event in this event queue, scheduled to execute at receive_time.

        Object X sends an event to object Y by invoking
            Y.event_queue.send_event(send_time, receive_time, X, Y, event_type, <event_body=value>)

        Args:
            send_time: number; the simulation time at which the message was generated (sent)
            receive_time: number; the simulation time at which the receiving_object will execute the event
            sending_object: object; the object sending the event
            receiving_object: object; the object that will receive the event; when the simulation is
                parallelized `sending_object` and `receiving_object` will need to be global
                identifiers.
            event_type: class; the type of the message; the class' string name is sent with the
                message, as messages should not contain references.
            event_body: object; an object containing the body of the event

        Raises:
            ValueError: if receive_time < send_time
        """

        # ensure send_time <= receive_time
        # send_time == receive_time can cause loops, but the application programmer is responsible for avoiding them
        if receive_time < send_time:
            raise ValueError("receive_time < send_time in schedule_event(): {} < {}".format(
                str(receive_time), str(send_time)))

        event = Event(send_time, receive_time, sending_object, receiving_object, event_type, event_body)
        heapq.heappush(self.event_heap, (receive_time, event))

    def next_event_time(self):
        """Get the time of the next event.

        Returns:
            The time of the next event. Return infinity if there is no next event.
        """
        if not self.event_heap:
            return float('inf')

        return self.event_heap[0][0]    # time of the 1st event in the heap


    def next_events(self, sim_obj):
        """Get the list of next event(s). Events are provided in a list because multiple events may
        have the same simultion time, and they must be provided to the simulation object as a unit.

        Handle 'ties' properly. That is, since an object may receive multiple events
        with the same event_time (aka receive_time), pass them all to the object in a list.
        (In a Time Warp simulation, the list would need to be ordered by some deterministic
        criteria based on its contents.)

        Args:
            sim_obj (SimulationObject): the simulation object that will execute the list of event(s)
                that are returned.

        Returns:
            A list of next event(s), which is empty if no events are available.
        """
        # TODO(Arthur): in a Time Warp simulation, order the list by some deterministic criteria
        # TODO(Arthur): based on its contents; see David's lecture on this

        if not self.event_heap:
            return []

        events = []
        (now, an_event) = heapq.heappop(self.event_heap)
        events.append(an_event)
        while self.event_heap and now == self.next_event_time():    # time of the 1st event in the heap
            (ignore, an_event) = heapq.heappop(self.event_heap)
            events.append(an_event)

        # sort events by message type priority, which is necessary for logical correctness
        # this scales for arbitrarily many message types
        events = sorted(events,
            key=lambda event:
            MessageTypesRegistry.receiver_priorities[most_qual_cls_name(sim_obj)].index(event.event_type))

        return events

    @staticmethod
    def event_list_to_string(event_list):
        '''Return event_list members as a tablesorted by event time.

        The table is formatted as a multi-line, tab-separated string.
        '''
        # todo: prepend header
        return "\n".join([str(event) for event in
            sorted(event_list, key=lambda event: event.event_time)])

    def __str__(self):
         '''return event queue members as a table'''
         return "\n".join([event.__str__() for (time, event) in self.event_heap])

    def nsmallest(self, n=10):
        '''Return event queue members as a table sorted by event time.

        Args:
            n (int, optional): number of events to return; default=10.
        '''
        return "\n".join([str(event) for (time,event) in
            heapq.nsmallest(n, self.event_heap, key=lambda event_heap_entry: event_heap_entry[0])])


class SimulationObject(object):
    """Base class for simulation objects.

    SimulationObject is a base class for all simulations objects. It provides basic functionality:
    the object's name (which must be unique), it's simulation time, a queue of received events, and a send_event() method.

    Attributes:
        name: A string with the simulation object's name.
        time: A float containing the simulation object's current simulation time.
        event_queue: The object's EventQueue.
        num_events: int; number of events processed
    """

    def __init__(self, name):
        """Initialize a SimulationObject.

        Create its event queue, initialize its name, and set its start time to 0.

        Args:
            name: string; the object's unique name, used as a key in the dict of objects
        """
        self.event_queue = EventQueue()
        self.name = name
        self.time = 0.0
        self.num_events = 0
        self.register()

    def register(self):
        """Register this simulation object with the simulation.

        Each class derived from SimulationObject must register() itself before the simulation starts.
        """
        SimulationEngine.add_object(self.name, self)

    def send_event(self, delay, receiving_object, event_type, event_body=None, copy=True):
        """Send a simulation event message.

        Args:
            delay: number; the simulation delay at which the receiving_object should execute the event.
            receiving_object: object; the object that will receive the event
            event_type (class): the class of the event message.
            event_body: object; an optional object containing the body of the event
            copy: boolean; if True, copy the event_body; True by default as a safety measure to
                avoid unexpected changes to shared objects; set False to optimize

        Raises:
            ValueError: if delay < 0
            ValueError: if the sending object type is not registered to send a message type
            ValueError: if the receiving simulation object type is not registered to receive the message type

        """
        if delay < 0:
            raise ValueError("delay < 0 in send_event(): {}".format(str(delay)))

        # Do not put a class reference in a message, as the message might not be received in the
        # same address space.
        # To eliminate the risk of name collisions use the fully qualified classname.
        event_type_name = most_qual_cls_name(event_type)

        # check that the sending object type is registered to send the message type
        if not event_type_name in MessageTypesRegistry.senders[most_qual_cls_name(self)]:
            raise ValueError("'{}' simulation objects not registered to send '{}' messages".format(
                most_qual_cls_name(self), event_type_name))

        # check that the receiving simulation object type is registered to receive the message type
        if not event_type_name in \
            MessageTypesRegistry.receiver_priorities[most_qual_cls_name(receiving_object)]:
            raise ValueError("'{}' simulation objects not registered to receive '{}' messages".format(
                most_qual_cls_name(receiving_object), event_type_name))

        event_body_copy = None
        if event_body and copy:
            event_body_copy = deepcopy(event_body)

        receiving_object.event_queue.schedule_event(self.time, self.time + delay, self,
            receiving_object, event_type_name, event_body)
        self.log_with_time("({}, {:6.2f}) -> ({}, {:6.2f}): {}".format(self.name, self.time,
            receiving_object.name, self.time + delay, event_type))

    def handle_event(self, event_list):
        """Handle a simulation event, which may involve multiple event messages.

        Each class derived from SimulationObject must implement handle_event().

        Attributes:
            event_list: A non-empty list of event messages in the event

        Raises:
            ValueError: if some event message in event_list has an invalid type
        """

        self.num_events += 1

        # check for messages with invalid types
        # TODO(Arthur): do this checking at send time, probably in SimulationObject.send_event()
        invalid_types = (set([x.event_type for x in event_list])  -
            set(MessageTypesRegistry.receiver_priorities[most_qual_cls_name(self)]))
        if len(invalid_types):
            raise ValueError("Error: invalid event event_type(s) '{}' in event_list:\n{}".format(
                ', '.join(list(invalid_types)),
                '\n'.join([str(ev_msg) for ev_msg in event_list])))

        # write events to a plot log, for plotting by plotSpaceTimeDiagram.py
        # plot logging is controlled by configuration files pointed to by config_constants and by env vars
        logger = debug_logs.get_log('wc.plot.file')
        for event in event_list:
            logger.debug(str(event), sim_time=self.time)

    def event_queue_to_str(self):
        eq = '{} at {:5.3f}\n'.format(self.name, self.time)
        if self.event_queue.event_heap:
            eq += Event.header() + '\n' + str(self.event_queue) + '\n'
        else:
            eq += 'Empty event queue'
        return eq

    def log_with_time(self, msg, local_call_depth=1):
        """Write a debug log message with the simulation time.
        """
        debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time,
            local_call_depth=local_call_depth)
