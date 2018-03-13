""" Base class for simulation objects and their event queues

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-06-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from copy import deepcopy
import heapq
import abc
from abc import ABCMeta
import warnings
import inspect

from wc_sim.core.event import Event
from wc_sim.core.errors import SimulatorError
from wc_utils.util.list import elements_to_str
from wc_utils.util.misc import most_qual_cls_name, round_direct
from wc_sim.core.simulation_message import SimulationMessage
from wc_sim.core.utilities import ConcreteABCMeta

# configure logging
from .debug_logs import logs as debug_logs


# TODO(Arthur): move to engine
class EventQueue(object):
    """ A simulation's event queue

    Stores a `SimulationEngine`'s events in a heap (also known as a priority queue).
    The heap is a 'min heap', which keeps the event with the smallest
    `(event_time, sending_object.name)` at the root in heap[0].
    This is implemented via comparison operations in `Event`.
    Thus, all entries with equal `(event_time, sending_object.name)` will be popped from the heap
    adjacently. `schedule_event()` costs `O(log(n))`, where `n` is the size of the heap,
    while `next_events()`, which returns all events with the minimum
    `(event_time, sending_object.name)`, costs `O(mlog(n))`, where `m` is the number of events
    returned.

    Attributes:
        event_heap (:obj:`list`): a `SimulationEngine`'s heap of events
    """

    def __init__(self):
        self.event_heap = []

    def reset(self):
        self.event_heap = []

    def schedule_event(self, send_time, receive_time, sending_object, receiving_object, message):
        """ Create an event and insert in this event queue, scheduled to execute at `receive_time`

        Simulation object `X` can sends an event to simulation object `Y` by invoking
            `X.send_event(receive_delay, Y, message)`.

        Args:
            send_time (:obj:`float`): the simulation time at which the event was generated (sent)
            receive_time (:obj:`float`): the simulation time at which the `receiving_object` will
                execute the event
            sending_object (:obj:`SimulationObject`): the object sending the event
            receiving_object (:obj:`SimulationObject`): the object that will receive the event; when
                the simulation is parallelized `sending_object` and `receiving_object` will need
                to be global identifiers.
            message (:obj:`SimulationMessage`): a `SimulationMessage` carried by the event; its type
                provides the simulation application's type for an `Event`; it may also carry a payload
                for the `Event` in its attributes.

        Raises:
            :obj:`SimulatorError`: if `receive_time` < `send_time`
        """

        # Ensure that send_time <= receive_time.
        # Events with send_time == receive_time can cause loops, but the application programmer
        # is responsible for avoiding them.
        if receive_time < send_time:
            raise SimulatorError("receive_time < send_time in schedule_event(): {} < {}".format(
                str(receive_time), str(send_time)))

        if not isinstance(message, SimulationMessage):
            raise SimulatorError("message should be an instance of {} but is a '{}'".format(
                SimulationMessage.__name__, type(message).__name__))

        event = Event(send_time, receive_time, sending_object, receiving_object, message)
        # As per David Jefferson's thinking, the event queue is ordered by data provided by the
        # simulation application, in particular the tuple (event time, receiving object name).
        # See the comparison operators for Event. This will achieve deterministic and reproducible
        # simulations.
        heapq.heappush(self.event_heap, event)

    def empty(self):
        """ Is the event queue empty?

        Returns:
            :obj:`bool`: return `True` if the event queue is empty
        """
        return not self.event_heap

    def next_event_time(self):
        """ Get the time of the next event

        Returns:
            :obj:`float`: the time of the next event; return infinity if no event is scheduled
        """
        if not self.event_heap:
            return float('inf')

        next_event = self.event_heap[0]
        next_event_time = next_event.event_time
        return next_event_time

    def next_event_obj(self):
        """ Get the simulation object that receives the next event

        Returns:
            :obj:`SimulationObject`): the simulation object that will execute the next event, or `None`
                if no event is scheduled
        """
        if not self.event_heap:
            return None

        next_event = self.event_heap[0]
        return next_event.receiving_object

    def next_events(self):
        """ Get all events at the smallest event time destined for the object whose name sorts earliest

        Because multiple events may occur concurrently -- that is, have the same simulation time --
        they must be provided as a collection to the simulation object that executes them.

        Handle 'ties' properly. That is, since an object may receive multiple events
        with the same event_time (aka receive_time), pass them all to the object in a list.

        Returns:
            :obj:`list` of `Event`: the earliest event(s), sorted by message type priority. If no
                events are available the list is empty.
        """
        if not self.event_heap:
            return []

        events = []
        next_event = heapq.heappop(self.event_heap)
        now = next_event.event_time
        receiving_obj = next_event.receiving_object
        events.append(next_event)

        # gather all events with the same event_time and receiving_object
        while (self.event_heap and now == self.next_event_time() and
            receiving_obj == self.next_event_obj()):
            events.append(heapq.heappop(self.event_heap))

        if 1<len(events):
            # sort events by message type priority, and within priority by message content
            # thus, a sim object handles simultaneous messages in priority order;
            # this costs O(n log(n)) in the number of event messages in events
            receiver_priority_dict = receiving_obj.get_receiving_priorities_dict()
            events = sorted(events,
                key=lambda event: (receiver_priority_dict[event.message.__class__], event.message))

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

    def render(self, sim_obj=None, as_list=False, separator='\t'):
        """ Return the content of an `EventQueue`

        Make a human-readable event queue, sorted by non-decreasing event time.
        Provide a header row and a row for each event. If all events have the same type of message,
        the header contains event and message fields. Otherwise, the header has event fields and
        a message field label, and each event labels message fields with their attribute names.

        Args:
            sim_obj (:obj:`SimulationObject`, optional): if provided, return only events to be
                received by `sim_obj`
            `EventQueue`'s values in a :obj:`list`
            as_list (:obj:`bool`, optional): if set, return the `EventQueue`'s values in a :obj:`list`
            separator (:obj:`str`, optional): the field separator used if the values are returned as
                a string

        Returns:
            :obj:`str`: String representation of the values of an `EventQueue`, or a :obj:`list`
                representation if `as_list` is set
        """
        event_heap = self.event_heap
        if sim_obj is not None:
            event_heap = list(filter(lambda event: event.receiving_object==sim_obj, event_heap))

        if not event_heap:
            return None

        # Sort the events in non-decreasing event time (receive_time, receiving_object.name)
        sorted_events = sorted(event_heap)

        # Does the queue contain multiple message types?
        message_types = set()
        for event in event_heap:
            message_types.add(event.message.__class__)
            if 1<len(message_types):
                break
        multiple_msg_types = 1<len(message_types)

        rendered_event_queue = []
        if multiple_msg_types:
            # The queue contains multiple message types
            rendered_event_queue.append(Event.header(as_list=True))
            for event in sorted_events:
                rendered_event_queue.append(event.render(annotated=True, as_list=True))

        else:
            # The queue contain only one message type
            # message_type = message_types.pop()
            event = sorted_events[0]
            rendered_event_queue.append(event.custom_header(as_list=True))
            for event in sorted_events:
                rendered_event_queue.append(event.render(as_list=True))

        if as_list:
            return rendered_event_queue
        else:
            table = []
            for row in rendered_event_queue:
                table.append(separator.join(elements_to_str(row)))
            return '\n'.join(table)

    def __str__(self):
        """ Return event queue members as a table
        """
        rv = self.render()
        if rv is None:
            return ''
        return rv


class SimulationObject(object):
    """Base class for simulation objects.

    SimulationObject is a base class for all simulations objects. It provides basic functionality:
    the object's name (which must be unique), its simulation time, a queue of received events,
    and a send_event() method.

        Args:
            send_time (:obj:`float`): the simulation time at which the message was generated (sent)

    Attributes:
        name (:obj:`str`): this simulation object's name, which is unique across all simulation objects
            handled by a `SimulationEngine`
        time (:obj:`float`): this simulation object's current simulation time
        num_events (:obj:`int`): number of events processed
        simulator (:obj:`int`): the `SimulationEngine` that uses this `SimulationObject`
    """
    def __init__(self, name):
        """Initialize a SimulationObject.

        Create its event queue, initialize its name, and set its start time to 0.

        Args:
            name: string; the object's unique name, used as a key in the dict of objects
        """
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
            # TODO(Arthur): reference to the simulator is problematic because it means simulator can't be GC'ed
            self.simulator = simulator
            return
        raise SimulatorError("SimulationObject '{}' is already part of a simulator".format(self.name))

    def delete(self):
        """Delete this object from a simulation.
        """
        # TODO(Arthur): is this an operation that makes sense to support? if not, remove it; if yes,
        # remove all of this object's state from simulator, and test it properly
        self.simulator = None

    def send_event_absolute(self, event_time, receiving_object, message, copy=True):
        """Send a simulation event message with an absolute event time.

        Args:
            event_time: number; the simulation time at which the receiving_object should execute the event
            receiving_object: object; the object that will receive the event
            message (class): the class of the event message
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
        # event_type_name = most_qual_cls_name(message)
        event_type_name = message.__class__.__name__

        # check that the sending object type is registered to send the message type
        if not isinstance(message, SimulationMessage):
            raise SimulatorError("simulation messages must be instances of type 'SimulationMessage'; "
                "'{}' is not".format(event_type_name))
        if message.__class__ not in self.__class__.metadata.message_types_sent:
            raise SimulatorError("'{}' simulation objects not registered to send '{}' messages".format(
                most_qual_cls_name(self), event_type_name))

        # check that the receiving simulation object type is registered to receive the message type
        receiver_priorities = receiving_object.get_receiving_priorities_dict()
        if message.__class__ not in receiver_priorities:
            raise SimulatorError("'{}' simulation objects not registered to receive '{}' messages".format(
                most_qual_cls_name(receiving_object), event_type_name))

        if copy:
            message = deepcopy(message)

        self.simulator.event_queue.schedule_event(self.time, event_time, self,
            receiving_object, message)
        self.log_with_time("Send: ({}, {:6.2f}) -> ({}, {:6.2f}): {}".format(self.name, self.time,
            receiving_object.name, event_time, message.__class__.__name__))

    def send_event(self, delay, receiving_object, message, copy=True):
        """Send a simulation event message, specifing the event time as a delay

        Args:
            delay: number; the simulation delay at which the receiving_object should execute the event.
            receiving_object: object; the object that will receive the event
            message: object; an object containing the body of the event
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
        for message_type, handler in handlers:
            if message_type in subclass.metadata.event_handlers:
                raise SimulatorError("message type '{}' appears repeatedly".format(
                    most_qual_cls_name(message_type)))
            if not callable(handler):
                raise SimulatorError("handler '{}' must be callable".format(handler))
            subclass.metadata.event_handlers[message_type] = handler

        for index,(message_type, _) in enumerate(handlers):
            subclass.metadata.event_handler_priorities[message_type] = index

    @staticmethod
    def register_sent_messages(subclass, sent_messages):
        """ Register the messages sent by a `SimulationObject` subclass

        Calling `register_sent_messages` re-initializes all registered sent message types.

        Args:
            subclass: `SimulationObject`: a subclass of `SimulationObject`
            sent_messages: list: list of `SimulationMessage`'s which can be sent
            by `SimulationObject`s of type `subclass`
        """
        for sent_message_type in sent_messages:
            subclass.metadata.message_types_sent.add(sent_message_type)

    def get_receiving_priorities_dict(self):
        """ Get priorities of message types handled by this `SimulationObject`'s type

        Returns:
            :obj:`dict`: mapping from message types handled by this `SimulationObject` to their
                execution priorities. The highest priority is 0, and higher values have lower
                priorities. Execution priorities determine the execution order of concurrent events
                at a `SimulationObject`.
        """
        return self.__class__.metadata.event_handler_priorities

    def _SimulationEngine__handle_event_list(self, event_list):
        """ Handle a list of simulation events, which may contain multiple concurrent events

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
                handler = self.__class__.metadata.event_handlers[event.message.__class__]
                handler(self, event)
            except KeyError: # pragma: no cover     # unreachable because of check that receiving sim
                                                    # obj type is registered to receive the message type
                raise SimulatorError("No handler registered for Simulation message type: '{}'".format(
                    event.message.__class__.__name__))

    def render_event_queue(self):
        """ Format an event queue as a string
        """
        return self.simulator.event_queue.render()

    def log_with_time(self, msg, local_call_depth=1):
        """Write a debug log message with the simulation time.
        """
        debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time,
            local_call_depth=local_call_depth)


class ApplicationSimulationObjectInterface(object, metaclass=ABCMeta):  # pragma: no cover

    @abc.abstractmethod
    def send_initial_events(self, *args): pass

    @abc.abstractmethod
    def get_state(self): pass


class ApplicationSimulationObjectMetadata(object):
    """ Meta data for an :class:`ApplicationSimulationObject`

    Attributes:
        event_handlers: dict: message_type -> event_handler; provides the event handler for each
            message type for a subclass of `SimulationObject`
        event_handler_priorities: `dict`: from message types handled by a `SimulationObject` subclass,
            to message type priority. The highest priority is 0, and priority decreases with
            increasing priority values.
        message_types_sent: set: the types of messages a subclass of `SimulationObject` has
            registered to send
    """
    def __init__(self):
        self.event_handlers = {}
        self.event_handler_priorities = {}
        self.message_types_sent = set()


class ApplicationSimulationObjMeta(type):
    # event handler mapping keyword
    EVENT_HANDLERS = 'event_handlers'
    # messages sent list keyword
    MESSAGES_SENT = 'messages_sent'

    def __new__(cls, clsname, superclasses, namespace):
        """
        Args:
            cls (:obj:`class`): an instance of this class
            clsname (:obj:`str`): name of `SimulationObject` subclass being created
            superclasses (:obj: `tuple`): tuple of superclasses
            namespace (:obj:`dict`): namespace of subclass of `ApplicationSimulationObject` being created

        Returns:
            :obj:`SimulationObject`: a new instance of a subclass of `SimulationObject`
        """
        # Short circuit when ApplicationSimulationObject is defined
        if clsname == 'ApplicationSimulationObject':
            return super().__new__(cls, clsname, superclasses, namespace)

        new_application_simulation_obj_subclass = super().__new__(cls, clsname, superclasses, namespace)
        new_application_simulation_obj_subclass.metadata = ApplicationSimulationObjectMetadata()

        # determine whether event_handlers or messages_sent is defined
        event_handlers_defined = cls.EVENT_HANDLERS in namespace
        messages_sent_defined = cls.MESSAGES_SENT in namespace
        for superclass in superclasses:
            for name,value in inspect.getmembers(superclass):
                # TODO(Arthur): check types more carefully
                event_handlers_defined |= name == cls.EVENT_HANDLERS
                messages_sent_defined |= name == cls.MESSAGES_SENT

        if (not event_handlers_defined and not messages_sent_defined):
            raise SimulatorError("ApplicationSimulationObject '{}' definition must provide '{}' or '{}' or both.".format(
                clsname, cls.EVENT_HANDLERS,
                cls.MESSAGES_SENT))
        elif not event_handlers_defined:
            warnings.warn("ApplicationSimulationObject '{}' definition does not provide '{}'.".format(
                clsname, cls.EVENT_HANDLERS))
        elif not messages_sent_defined:
            warnings.warn("ApplicationSimulationObject '{}' definition does not provide '{}'.".format(
                clsname, cls.MESSAGES_SENT))

        # define, and perhaps override, the event_handlers and messages_sent attributes of cls
        if cls.EVENT_HANDLERS in namespace:
            handler_mapping = []
            for msg_type,handler_name in namespace[cls.EVENT_HANDLERS]:
                # TODO(Arthur): check types more carefully: msg_types
                '''
                if not isinstance(msg_type, SimulationMessage):
                    raise SimulatorError("In ApplicationSimulationObject '{}' definition of '{}' "
                        "must be list of (SimulationMessage, method) pairs.".format(clsname, cls.EVENT_HANDLERS))
                '''
                if isinstance(handler_name, str):
                    if handler_name not in namespace:
                        raise SimulatorError("ApplicationSimulationObject '{}' definition must define "
                            "'{}'.".format(clsname, handler_name))
                    handler_mapping.append((msg_type, namespace[handler_name]))
                elif callable(handler_name):
                    handler_mapping.append((msg_type, handler_name))
                else:
                    raise SimulatorError("ApplicationSimulationObject '{}' handler_name '{}' must "
                        "be a string or a callable.".format(clsname, handler_name))
            new_application_simulation_obj_subclass.register_handlers(
                new_application_simulation_obj_subclass, handler_mapping)
        if cls.MESSAGES_SENT in namespace:
                # TODO(Arthur): check types more carefully: msg_types
            new_application_simulation_obj_subclass.register_sent_messages(
                new_application_simulation_obj_subclass,
                namespace[cls.MESSAGES_SENT])

        # return the class to instantiate it
        return new_application_simulation_obj_subclass


class AppSimObjAndABCMeta(ApplicationSimulationObjMeta, ConcreteABCMeta):
    """ A concrete class based on two Meta classes to be used as a metaclass for classes derived from both
    """
    pass


class ApplicationSimulationObject(SimulationObject, ApplicationSimulationObjectInterface,
    metaclass=AppSimObjAndABCMeta):

    def send_initial_events(self, *args): return
    def get_state(self): return ''
