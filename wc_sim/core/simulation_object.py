""" Base class for simulation objects and their event queues

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-06-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from copy import deepcopy
import heapq
import abc
import six

from wc_sim.core.event import Event
from wc_sim.core.errors import SimulatorError
from wc_utils.util.list import elements_to_str
from wc_utils.util.misc import most_qual_cls_name, round_direct
from wc_sim.core.simulation_message import SimulationMessage

# configure logging
from .debug_logs import logs as debug_logs


class EventQueue(object):
    """ A simulation object's event queue

    Stores an object's events in a heap (also known as a priority queue),
    with each event in a tuple (time, event). The heap is
    a 'min heap', which keeps the event with the smallest time at the root in heap[0].

    Attributes:
        event_heap (:obj:`list`): The object's heap of events, sorted by event time;
            the operations get earliest event and insert event cost O(log(n)) in the heap
    """

    def __init__(self):
        self.event_heap = []

    def schedule_event(self, send_time, receive_time, sending_object, receiving_object, message):
        """ Insert an event in this event queue, scheduled to execute at `receive_time`

        Object X sends an event to object Y by invoking
            `Y.event_queue.send_event(send_time, receive_time, X, Y, event_type, <message=value>)`

        Args:
            send_time (:obj:`float`): the simulation time at which the message was generated (sent)
            receive_time (:obj:`float`): the simulation time at which the receiving_object will execute the event
            sending_object: object; the object sending the event
            receiving_object: object; the object that will receive the event; when the simulation is
                parallelized `sending_object` and `receiving_object` will need to be global
                identifiers.
            event_type: class; the type of the message; the class' string name is sent with the
                message, as messages should not contain references.
            message: object; an object containing the body of the event

        Raises:
            SimulatorError: if receive_time < send_time
        """

        # ensure that send_time <= receive_time
        # events with send_time == receive_time can cause loops, but the application programmer
        # is responsible for avoiding them
        if receive_time < send_time:
            raise SimulatorError("receive_time < send_time in schedule_event(): {} < {}".format(
                str(receive_time), str(send_time)))

        if not isinstance(message, SimulationMessage):
            raise SimulatorError("message should be an instance of {} but is a '{}'".format(
                SimulationMessage.__name__, type(message).__name__))

        event = Event(send_time, receive_time, sending_object, receiving_object, message)
        heapq.heappush(self.event_heap, (receive_time, event))

    def next_event_time(self):
        """ Get the time of the next event

        Returns:
            :obj:`float`: the time of the next event; return infinity if no event is scheduled
        """
        if not self.event_heap:
            return float('inf')

        return self.event_heap[0][0]    # time of the 1st event in the heap


    def next_events(self, sim_obj):
        """ Get all events at the smallest event time

        Because multiple events may occur concurrently -- that is, have the same simulation time --
        they must be provided as a collection to the simulation object that executes them.

        # TODO(Arthur): take care of this
        Handle 'ties' properly. That is, since an object may receive multiple events
        with the same event_time (aka receive_time), pass them all to the object in a list.
        (In a Time Warp simulation, the list would need to be ordered by some deterministic
        criteria based on its contents.)

        # TODO(Arthur): stop using an argument; sim_obj is just the receiver of the events in the queue
        Args:
            sim_obj (:obj:`SimulationObject`): the simulation object that will execute the list of event(s)
                that are returned.

        Returns:
            :obj:`list` of `Event`: the earliest event(s), sorted by message type priority. If no
                events are available the list is empty.
        """
        # TODO(Arthur): in a Time Warp simulation, order the list by some deterministic criteria
        # based on its contents; see David Jefferson's lecture on this
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
                key=lambda event: receiver_priority_dict[event.message.__class__])

        for event in events:
            self.log_event(event)

        return events

    def log_event(self, event, local_call_depth=1):
        """ Log an event with its simulation time

        Args:
            event (:obj:`Event`): the Event to log
            local_call_depth (:obj:`int`, optional): the local call depth; default = 1
        """
        debug_logs.get_log('wc.debug.file').debug("Execute: {} {}:{} {} ({})".format(event.event_time,
                type(event.receiving_object).__name__,
                event.receiving_object.name,
                event.message.__class__.__name__,
                str(event.message)),
                sim_time=event.event_time,
                local_call_depth=local_call_depth)

    def __str__(self):
        """return event queue members as a table"""
        return "\n".join([event.__str__() for (time, event) in self.event_heap])

    def render(self, as_list=False, separator='\t'):
        """ Return the content of an `EventQueue`

        Make a human-readable event queue, sorted by non-decreasing event time.
        Provide a header row and a row for each event. If all events have the same type of message,
        the header contains event and message fields. Otherwise, the header has event fields and
        a message field label, and each event labels message fields with their attribute names.

        Args:
            as_list (:obj:`bool`, optional): if set, return the `EventQueue`'s values in a :obj:`list`
            separator (:obj:`str`, optional): the field separator used if the values are returned as
                a string

        Returns:
            :obj:`str`: String representation of the values of an `EventQueue`, or a :obj:`list`
                representation if `as_list` is set
        """
        if not self.event_heap:
            return None

        # Sort the events in non-decreasing event time (receive_time)
        sorted_events = sorted(self.event_heap, key=lambda record: record[0])
        # Does the queue contain multiple message types?
        message_types = set()
        for _,event in self.event_heap:
            message_types.add(event.message.__class__)
            if 1<len(message_types):
                break
        multiple_msg_types = 1<len(message_types)

        rendered_event_queue = []
        if multiple_msg_types:
            # The queue contains multiple message types
            rendered_event_queue.append(Event.header(as_list=True))
            for _,event in sorted_events:
                rendered_event_queue.append(event.render(annotated=True, as_list=True))

        else:
            # The queue contain only one message type
            # message_type = message_types.pop()
            (_,event) = sorted_events[0]
            rendered_event_queue.append(event.custom_header(as_list=True))
            for _,event in sorted_events:
                rendered_event_queue.append(event.render(as_list=True))

        if as_list:
            return rendered_event_queue
        else:
            table = []
            for row in rendered_event_queue:
                table.append(separator.join(elements_to_str(row)))
            return '\n'.join(table)


class SimulationObject(object):
    """Base class for simulation objects.

    SimulationObject is a base class for all simulations objects. It provides basic functionality:
    the object's name (which must be unique), its simulation time, a queue of received events,
    and a send_event() method.

        Args:
            send_time (:obj:`float`): the simulation time at which the message was generated (sent)

    Attributes:
        name (:obj:`str`): this simulation object's name
        time (:obj:`float`): this simulation object's current simulation time
        event_queue (:obj:`EventQueue`): this simulation object's event queue
        num_events (:obj:`int`): number of events processed
        simulator (:obj:`int`): the `SimulationEngine` that uses this `SimulationObject`

    Derived class attributes:
        event_handlers: dict: message_type -> event_handler; provides the event handler for each
            message type for a subclass of `SimulationObject`
        event_handler_priorities: `dict`: from message types handled by a `SimulationObject` subclass,
            to message type priority. The highest priority is 0, and priority decreases with
            increasing priority values.
        message_types_sent: set: the types of messages a subclass of `SimulationObject` has
            registered to send

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
        self.simulator = None

    def add(self, simulator):
        """Add this object to a simulation.

        Args:
            simulator: `SimulationEngine`: the simulator that will use this `SimulationObject`

        Raises:
            SimulatorError: if this `SimulationObject` is already registered with a simulator
        """
        if self.simulator is None:
            self.simulator = simulator
            return
        raise SimulatorError("SimulationObject '{}' is already part of a simulator".format(self.name))

    def delete(self):
        """Delete this object from a simulation.
        """
        self.simulator = None

    def send_event_absolute(self, event_time, receiving_object, message, copy=True):
        """Send a simulation event message with an absolute event time.

        Args:
            event_time: number; the simulation time at which the receiving_object should execute the event
            receiving_object: object; the object that will receive the event
            event_type (class): the class of the event message
            message: object; an optional object containing the body of the event
            copy: boolean; if True, copy the message; True by default as a safety measure to
                avoid unexpected changes to shared objects; set False to optimize

        Raises:
            SimulatorError: if event_time < 0
            SimulatorError: if the sending object type is not registered to send a message type
            SimulatorError: if the receiving simulation object type is not registered to receive the message type
        """
        if event_time < self.time:
            raise SimulatorError("event_time ({}) < current time ({}) in send_event_absolute()".format(
                round_direct(event_time, precision=3), round_direct(self.time, precision=3)))

        # Do not put a class reference in a message, as the message might not be received in the
        # same address space.
        # To eliminate the risk of name collisions use the fully qualified classname.
        # TODO(Arthur): wait until after MVP
        # event_type_name = most_qual_cls_name(event_type)
        event_type_name = message.__class__.__name__

        # check that the sending object type is registered to send the message type
        if (not hasattr(self.__class__, 'message_types_sent') or
            message.__class__ not in self.__class__.message_types_sent):
            raise SimulatorError("'{}' simulation objects not registered to send '{}' messages".format(
                most_qual_cls_name(self), event_type_name))

        # check that the receiving simulation object type is registered to receive the message type
        receiver_priorities = receiving_object.get_receiving_priorities_dict()
        if message.__class__ not in receiver_priorities:
            raise SimulatorError("'{}' simulation objects not registered to receive '{}' messages".format(
                most_qual_cls_name(receiving_object), event_type_name))

        if copy:
            message = deepcopy(message)

        receiving_object.event_queue.schedule_event(self.time, event_time, self,
            receiving_object, message)
        self.log_with_time("Send: ({}, {:6.2f}) -> ({}, {:6.2f}): {}".format(self.name, self.time,
            receiving_object.name, event_time, message.__class__.__name__))

    def send_event(self, delay, receiving_object, message, copy=True):
        """Send a simulation event message, specifing the event time as a delay

        Args:
            delay: number; the simulation delay at which the receiving_object should execute the event.
            receiving_object: object; the object that will receive the event
            event_type (class): the class of the event message.
            message: object; an optional object containing the body of the event
            copy: boolean; if True, copy the message; True by default as a safety measure to
                avoid unexpected changes to shared objects; set False to optimize

        Raises:
            SimulatorError: if delay < 0
            SimulatorError: if the sending object type is not registered to send a message type
            SimulatorError: if the receiving simulation object type is not registered to receive the message type
        """
        if delay < 0:
            raise SimulatorError("delay < 0 in send_event(): {}".format(str(delay)))
        self.send_event_absolute(delay + self.time, receiving_object, message, copy=copy)

    @staticmethod
    def register_handlers(subclass, handlers):
        """Register a `SimulationObject`'s event handler methods.

        The priority of message execution in an event containing multiple messages
        is determined by the sequence of tuples in `handlers`.
        Each call to `register_handlers` re-initializes all event handler methods.

        Args:
            subclass: `SimulationObject`: a subclass of `SimulationObject`
            handlers: list: list of (`SimulationMessage`, method) tuples, ordered
                in decreasing priority for handling simulation message types

        Raises:
            SimulatorError: if a `SimulationMessage` appears repeatedly in `handlers`
            SimulatorError: if a handler is not callable
        """
        subclass.event_handlers = {}
        for message_type, handler in handlers:
            if message_type in subclass.event_handlers:
                raise SimulatorError("message type '{}' appears repeatedly".format(
                    most_qual_cls_name(message_type)))
            if not callable(handler):
                raise SimulatorError("handler '{}' must be callable".format(handler))
            subclass.event_handlers[message_type] = handler

        subclass.event_handler_priorities = {}
        for index,(message_type, _) in enumerate(handlers):
            subclass.event_handler_priorities[message_type] = index

    @staticmethod
    def register_sent_messages(subclass, sent_messages):
        """Register the messages sent by a `SimulationObject`.

        Calling `register_sent_messages` re-initializes all registered sent message types.

        Args:
            subclass: `SimulationObject`: a subclass of `SimulationObject`
            sent_messages: list: list of `SimulationMessage`'s which can be sent
            by the calling `SimulationObject`
        """
        subclass.message_types_sent = set()
        for sent_message_type in sent_messages:
            subclass.message_types_sent.add(sent_message_type)

    def get_receiving_priorities_dict(self):
        """ Get priorities of message types handled by this `SimulationObject`'s type

        Returns:
            :obj:`dict`: mapping from message types handled by this `SimulationObject` to their
                execution priorities. The highest priority is 0, and higher values have lower
                priorities. Execution priorities determine the execution order of concurrent events
                at a `SimulationObject`.

        Raises:
            SimulatorError: if this `SimulationObject` type has not registered its message handlers
        """
        if not hasattr(self.__class__, 'event_handler_priorities'):
            raise SimulatorError("SimulationObject type '{}' must call register_handlers()".format(
                self.__class__.__name__))
        return self.__class__.event_handler_priorities

    def _SimulationEngine__handle_event(self, event_list):
        """ Handle a simulation event, which may involve multiple event messages

        This method's special name ensures that it cannot be overridden, and can only be called
        from `SimulationEngine`.

        Attributes:
            event_list (:obj:`list` of `Event`): the `Event` message(s) in the simulation event

        Raises:
            SimulatorError: if a message in `event_list` has an invalid type
        """
        # TODO(Arthur): rationalize naming between simulation message, event, & event_list.
        # The PDES field needs this clarity.
        self.num_events += 1

        # write events to a plot log
        # plot logging is controlled by configuration files pointed to by config_constants and by env vars
        logger = debug_logs.get_log('wc.plot.file')
        for event in event_list:
            logger.debug(str(event), sim_time=self.time)

        # iterate through event_list, branching to handler
        for event in event_list:
            try:
                handler = self.__class__.event_handlers[event.message.__class__]
                handler(self, event)
            except KeyError: # pragma: no cover     # unreachable because of check that receiving sim
                                                    # obj type is registered to receive the message type
                raise SimulatorError("No handler registered for Simulation message type: '{}'".format(
                    event.message.__class__.__name__))

    def event_queue_to_str(self):
        """Format an event queue as a string.
        """
        eq_str = '{} at {:5.3f}\n'.format(self.name, self.time)
        if self.event_queue.event_heap:
            eq_str += Event.header() + '\n' + str(self.event_queue)
        else:
            eq_str += 'Empty event queue'
        return eq_str

    def log_with_time(self, msg, local_call_depth=1):
        """Write a debug log message with the simulation time.
        """
        debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time,
            local_call_depth=local_call_depth)


@six.add_metaclass(abc.ABCMeta)
class SimulationObjectInterface():  # pragma: no cover
    """Classes derived from `SimulationObject` must implement this interface
    """

    @abc.abstractmethod
    def send_initial_events(self, *args):
        """Send the `SimulationObject`'s initial event messages.

        This method is distinct from initializing the `SimulationObject` with `__init__()`, because
        it requires that communicating `SimulationObject`'s exist. It may send no events.

        Args:
            args: tuple: parameters needed to send the initial event messages
        """
        pass

    @abc.abstractmethod
    def get_state(self):
        """ Obtain the state of a `SimulationObject`

        This is called by the simulation engine to log the simulator state

        Returns:
            :obj:`str`: the state of a `SimulationObject`, in a human-readable string
        """
        pass

    @classmethod
    @abc.abstractmethod
    def register_subclass_handlers(cls):
        """Register all of the `SimulationObject`'s event handler methods.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def register_subclass_sent_messages(cls):
        """Register the messages sent by a `SimulationObject`.
        """
        pass
