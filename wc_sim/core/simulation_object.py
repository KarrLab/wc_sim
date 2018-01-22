''' Base class for simulation objects.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-06-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

from copy import deepcopy
import heapq
import abc
import six

from wc_sim.core.event import Event
from wc_utils.util.misc import most_qual_cls_name, round_direct

# configure logging
from .debug_logs import logs as debug_logs

class EventQueue(object):
    '''A simulation object's event queue.

    Stores a heap of an object's events, with each event in a tuple (time, event). The heap is
    a 'min heap', with the event with the smallest time at the root.

    Attributes:
        event_heap: The object's heap of events, sorted by event time; has O(n logn) get earliest
        and insert event.
    '''

    def __init__(self):
        self.event_heap = []

    def schedule_event(self, send_time, receive_time, sending_object, receiving_object, event_type,
        event_body=None):
        '''Insert an event in this event queue, scheduled to execute at receive_time.

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
        '''

        # ensure send_time <= receive_time
        # send_time == receive_time can cause loops, but the application programmer is responsible
        # for avoiding them
        if receive_time < send_time:
            raise ValueError("receive_time < send_time in schedule_event(): {} < {}".format(
                str(receive_time), str(send_time)))

        event = Event(send_time, receive_time, sending_object, receiving_object, event_type, event_body)
        heapq.heappush(self.event_heap, (receive_time, event))

    def next_event_time(self):
        '''Get the time of the next event.

        Returns:
            The time of the next event. Return infinity if there is no next event.
        '''
        if not self.event_heap:
            return float('inf')

        return self.event_heap[0][0]    # time of the 1st event in the heap


    def next_events(self, sim_obj):
        '''Get the list of next event(s).

        Events are provided in a list because multiple events may have the same simultion time,
        and they must be provided to the simulation object as a unit.

        Handle 'ties' properly. That is, since an object may receive multiple events
        with the same event_time (aka receive_time), pass them all to the object in a list.
        (In a Time Warp simulation, the list would need to be ordered by some deterministic
        criteria based on its contents.)

        Args:
            sim_obj (SimulationObject): the simulation object that will execute the list of event(s)
                that are returned.

        Returns:
            A list of next event(s), sorted by message type priority. The list will be empty if no
                events are available.
        '''
        # TODO(Arthur): in a Time Warp simulation, order the list by some deterministic criteria
        # based on its contents; see David's lecture on this

        if not self.event_heap:
            return []

        events = []
        (now, an_event) = heapq.heappop(self.event_heap)
        events.append(an_event)
        while self.event_heap and now == self.next_event_time():    # time of the 1st event in the heap
            (_, an_event) = heapq.heappop(self.event_heap)
            events.append(an_event)

        if 1<len(events):
            # sort events by message type priority, so an object handles simultaneous messages
            # in priority order; this costs O(n log(n)) in the number of event messages in events
            receiver_priority_dict = sim_obj.get_receiving_priorities_dict()
            events = sorted(events,
                key=lambda event: receiver_priority_dict[event.event_type])

        for event in events:
            self.log_event(event)

        return events

    def log_event(self, event, local_call_depth=1):
        '''Log an event with its simulation time.
        '''
        debug_logs.get_log('wc.debug.file').debug("Execute: {} {}:{} {} ({})".format(event.event_time,
                type(event.receiving_object).__name__,
                event.receiving_object.name,
                event.event_type.split('.')[-1],
                str(event.event_body)),
                sim_time=event.event_time,
                local_call_depth=local_call_depth)

    @staticmethod
    def event_list_to_string(event_list):
        '''Return event_list members as a tablesorted by event time.

        The table is formatted as a multi-line, tab-separated string.
        '''
        # todo: optionally prepend header
        return "\n".join([str(event) for event in
            sorted(event_list, key=lambda event: event.event_time)])

    def __str__(self):
        '''return event queue members as a table'''
        return "\n".join([event.__str__() for (time, event) in self.event_heap])

    def nsmallest(self, num=10):
        '''Return event queue members as a table sorted by event time.

        Args:
            num (int, optional): number of events to return; default=10.
        '''
        return "\n".join([str(event) for (time, event) in
            heapq.nsmallest(num, self.event_heap, key=lambda event_heap_entry: event_heap_entry[0])])

class SimulationObject(object):
    '''Base class for simulation objects.

    SimulationObject is a base class for all simulations objects. It provides basic functionality:
    the object's name (which must be unique), its simulation time, a queue of received events,
    and a send_event() method.

    Attributes:
        name: A string with the simulation object's name.
        time: A float containing the simulation object's current simulation time.
        event_queue: The object's EventQueue.
        num_events: int; number of events processed
        simulator: The `SimulationEngine` which uses this `SimulationObject`

    Derived class attributes:
        event_handlers: dict: message_type -> event_handler; provides the event handler for each
            message type for a subclass of `SimulationObject`
        event_handler_priorities: `dict`: from message types handled by a `SimulationObject` subclass,
            to message type priority. The highest priority is 0, and priority decreases with
            increasing priority values.
        message_types_sent: set: the types of messages a subclass of `SimulationObject` has
            registered to send

    '''
    def __init__(self, name):
        '''Initialize a SimulationObject.

        Create its event queue, initialize its name, and set its start time to 0.

        Args:
            name: string; the object's unique name, used as a key in the dict of objects
        '''
        self.event_queue = EventQueue()
        self.name = name
        self.time = 0.0
        self.num_events = 0
        self.simulator = None

    def add(self, simulator):
        '''Add this object to a simulation.

        Args:
            simulator: `SimulationEngine`: the simulator that will use this `SimulationObject`

        Raises:
            ValueError: if this `SimulationObject` is already registered with a simulator
        '''
        if self.simulator is None:
            self.simulator = simulator
            return
        raise ValueError("SimulationObject '{}' is already part of a simulator".format(self.name))

    def delete(self):
        '''Delete this object from a simulation.
        '''
        self.simulator = None

    def send_event_absolute(self, event_time, receiving_object, event_type, event_body=None, copy=True):
        '''Send a simulation event message with an absolute event time.

        Args:
            event_time: number; the simulation time at which the receiving_object should execute the event
            receiving_object: object; the object that will receive the event
            event_type (class): the class of the event message
            event_body: object; an optional object containing the body of the event
            copy: boolean; if True, copy the event_body; True by default as a safety measure to
                avoid unexpected changes to shared objects; set False to optimize

        Raises:
            ValueError: if event_time < 0
            ValueError: if the sending object type is not registered to send a message type
            ValueError: if the receiving simulation object type is not registered to receive the message type
        '''
        if event_time < self.time:
            raise ValueError("event_time ({}) < current time ({}) in send_event_absolute()".format(
                round_direct(event_time, precision=3), round_direct(self.time, precision=3)))

        # Do not put a class reference in a message, as the message might not be received in the
        # same address space.
        # To eliminate the risk of name collisions use the fully qualified classname.
        event_type_name = most_qual_cls_name(event_type)

        # check that the sending object type is registered to send the message type
        if (not hasattr(self.__class__, 'message_types_sent') or
            event_type_name not in self.__class__.message_types_sent):
            raise ValueError("'{}' simulation objects not registered to send '{}' messages".format(
                most_qual_cls_name(self), event_type_name))

        # check that the receiving simulation object type is registered to receive the message type
        receiver_priorities = receiving_object.get_receiving_priorities_dict()
        if event_type_name not in receiver_priorities:
            raise ValueError("'{}' simulation objects not registered to receive '{}' messages".format(
                most_qual_cls_name(receiving_object), event_type_name))

        if event_body and copy:
            event_body = deepcopy(event_body)

        receiving_object.event_queue.schedule_event(self.time, event_time, self,
            receiving_object, event_type_name, event_body)
        self.log_with_time("Send: ({}, {:6.2f}) -> ({}, {:6.2f}): {}".format(self.name, self.time,
            receiving_object.name, event_time, event_type.__name__))

    def send_event(self, delay, receiving_object, event_type, event_body=None, copy=True):
        '''Send a simulation event message, specifing the event time as a delay

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
        '''
        if delay < 0:
            raise ValueError("delay < 0 in send_event(): {}".format(str(delay)))
        self.send_event_absolute(delay + self.time, receiving_object, event_type,
            event_body=event_body, copy=copy)

    @staticmethod
    def register_handlers(subclass, handlers):
        '''Register a `SimulationObject`'s event handler methods.

        The priority of message execution in an event containing multiple messages
        is determined by the sequence of tuples in `handlers`.
        Each call to `register_handlers` re-initializes all event handler methods.

        Args:
            subclass: `SimulationObject`: a subclass of `SimulationObject`
            handlers: list: list of (`SimulationMessage`, method) tuples, ordered
                in decreasing priority for handling simulation message types

        Raises:
            ValueError: if a `SimulationMessage` appears repeatedly in `handlers`
            ValueError: if a handler is not callable
        '''
        subclass.event_handlers = {}
        for message_type, handler in handlers:
            if most_qual_cls_name(message_type) in subclass.event_handlers:
                raise ValueError("message type '{}' appears repeatedly".format(
                    most_qual_cls_name(message_type)))
            if not callable(handler):
                raise ValueError("handler '{}' must be callable".format(handler))
            subclass.event_handlers[most_qual_cls_name(message_type)] = handler

        subclass.event_handler_priorities = {}
        for index,(message_type, _) in enumerate(handlers):
            subclass.event_handler_priorities[most_qual_cls_name(message_type)] = index

    @staticmethod
    def register_sent_messages(subclass, sent_messages):
        '''Register the messages sent by a `SimulationObject`.

        Calling `register_sent_messages` re-initializes all registered sent message types.

        Args:
            subclass: `SimulationObject`: a subclass of `SimulationObject`
            sent_messages: list: list of `SimulationMessage`'s which can be sent
            by the calling `SimulationObject`
        '''
        subclass.message_types_sent = set()
        for sent_message_type in sent_messages:
            subclass.message_types_sent.add(most_qual_cls_name(sent_message_type))

    def get_receiving_priorities_dict(self):
        '''Provide dict mapping from message types handled by a `SimulationObject` subclass,
            to message type priority. The highest priority is 0, and priority decreases with
            increasing priority values.
        '''
        if not hasattr(self.__class__, 'event_handler_priorities'):
            raise Exception("SimulationObject type '{}' must call register_handlers()".format(
                self.__class__.__name__))
        return self.__class__.event_handler_priorities

    def _SimulationEngine__handle_event(self, event_list):
        '''Handle a simulation event, which may involve multiple event messages.

        Cannot be overridden, and can only be called from `SimulationEngine`.

        Attributes:
            event_list: A non-empty list of event messages in the event

        Raises:
            ValueError: if some event message in event_list has an invalid type
        '''
        # todo: rationalize naming between simulation message, event, & event_list
        # the PDES field needs this

        self.num_events += 1

        # write events to a plot log, for plotting by plotSpaceTimeDiagram.py
        # plot logging is controlled by configuration files pointed to by config_constants and by env vars
        logger = debug_logs.get_log('wc.plot.file')
        for event in event_list:
            logger.debug(str(event), sim_time=self.time)

        # iterate through event_list, branching to handler
        for event_message in event_list:
            try:
                handler = self.__class__.event_handlers[event_message.event_type]
                handler(self, event_message)
            except KeyError:
                raise ValueError("No handler registered for Simulation message type: '{}'".format(
                    event_message.event_type))

    def event_queue_to_str(self):
        '''Format an event queue as a string.
        '''
        eq_str = '{} at {:5.3f}\n'.format(self.name, self.time)
        if self.event_queue.event_heap:
            eq_str += Event.header() + '\n' + str(self.event_queue)
        else:
            eq_str += 'Empty event queue'
        return eq_str

    def log_with_time(self, msg, local_call_depth=1):
        '''Write a debug log message with the simulation time.
        '''
        debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time,
            local_call_depth=local_call_depth)


# todo: should this inherit from object; is it possible to combine this with SimulationObject into one class?
@six.add_metaclass(abc.ABCMeta)
class SimulationObjectInterface():
    '''Classes derived from `SimulationObject` must implement this interface.
    '''

    @abc.abstractmethod
    def send_initial_events(self, *args):
        '''Send the `SimulationObject`'s initial event messages.

        This method is distinct from initializing the `SimulationObject` with `__init__()`, because
        it requires that communicating `SimulationObject`'s exist. It may send no events.

        Args:
            args: tuple: parameters needed to send the initial event messages
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def register_subclass_handlers(cls):
        '''Register all of the `SimulationObject`'s event handler methods.
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def register_subclass_sent_messages(cls):
        '''Register the messages sent by a `SimulationObject`.
        '''
        pass
