""" Simulation event structure

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-05-31
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from wc_utils.util.misc import round_direct
from wc_sim.core.simulation_message import SimulationMessage


class Event(object):
    """ An object that holds a discrete event simulation (DES) event

    Each DES event is scheduled by creating an `Event` instance and storing it in its
    `receiving_object`'s event queue. To reduce interface errors the `event_type` and
    `message` attributes must be structured as specified in the `message_types` module.

    Attributes:
        creation_time (:obj:`float`): simulation time when the event is created (aka `send_time`)
        event_time (:obj:`float`): simulation time when the event must be executed (aka `receive_time`)
        sending_object (:obj:`SimulationObject`): reference to the object that sends the event
        receiving_object (:obj:`SimulationObject`): reference to the object that receives (aka executes)
            the event
        # TODO(Arthur): FIX
        event_type (:obj:`str`): the event's type; the name of a class declared in `message_types`
        # TODO(Arthur): FIX
        message (:obj:`object`): reference to a `body` subclass of a message type declared in
            `message_types`; the message payload
    """
    # TODO(Arthur): for performance, perhaps pre-allocate and reuse events

    # use __slots__ to save space
    # TODO(Arthur): figure out how to stop Sphinx from documenting these __slots__ as attributes
    __slots__ = "creation_time event_time sending_object receiving_object message".split()
    BASE_HEADERS = ['t(send)', 't(event)', 'Sender', 'Receiver', 'Event type']

    def __init__(self, creation_time, event_time, sending_object, receiving_object, message):
        self.creation_time = creation_time
        self.event_time = event_time
        self.sending_object = sending_object
        self.receiving_object = receiving_object
        self.message = message

    def __lt__(self, other):
        """ Does this `Event` occur earlier than `other`?

        Args:
            other (:obj:`Event`): another `Event`

        Returns:
            :obj:`bool`: `True` if this `Event` occurs earlier than `other`
        """
        return self.event_time < other.event_time

    def __le__(self, other):
        """ Does this `Event` occur earlier or at the same time as `other`?

        Args:
            other (:obj:`Event`): another `Event`

        Returns:
            :obj:`bool`: `True` if this `Event` occurs earlier or at the same time as `other`
        """
        return self.event_time <= other.event_time

    def __gt__(self, other):
        """ Does this `Event` occur later than `other`?

        Args:
            other (:obj:`Event`): another `Event`

        Returns:
            :obj:`bool`: `True` if this `Event` occurs later than `other`
        """
        return self.event_time > other.event_time

    def __ge__(self, other):
        """ Does this `Event` occur later or at the same time as `other`?

        Args:
            other (:obj:`Event`): another `Event`

        Returns:
            :obj:`bool`: `True` if this `Event` occurs later or at the same time as `other`
        """
        return self.event_time >= other.event_time

    def __str__(self):
        """ Return an `Event` as a string

        To generate the returned string, it is assumed that `sending_object` and `receiving_object`
        have name attributes.

        Returns:
            :obj:`str`: String representation of the `Event`'s fields, except `message`,
                delimited by tabs
        """
        # TODO(Arthur): allow formatting of the returned string, e.g. formatting the precision of time values
        vals = [round_direct(self.creation_time), round_direct(self.event_time),
            self.sending_object.name, self.receiving_object.name, self.message.__class__.__name__]
        if self.message.values():
            vals.append(self.message.values())
        return '\t'.join(vals)

    @staticmethod
    def header(as_list=False, separator='\t'):
        """ Return a header for an :obj:`Event` table

        Provide generic header suitable for any type of message in an event.

        Args:
            as_list (:obj:`bool`, optional): if set, return the header fields in a :obj:`list`
            separator (:obj:`str`, optional): the separator used if the header is returned as
                a string

        Returns:
            :obj:`str`: String representation of names of an :obj:`Event`'s fields, or a :obj:`list`
                representation if `as_list` is set
        """
        MESSAGE_FIELDS_HEADER = 'Message fields...'
        list_repr = Event.BASE_HEADERS + [MESSAGE_FIELDS_HEADER]
        if as_list:
            return list_repr
        else:
            return separator.join(list_repr)

    def custom_header(self, as_list=False, separator='\t'):
        """ Return a header for an :obj:`Event` table containing messages of a particular type

        Args:
            as_list (:obj:`bool`, optional): if set, return the header fields in a :obj:`list`
            separator (:obj:`str`, optional): the separator used if the header is returned as
                a string

        Returns:
            :obj:`str`: String representation of names of an `Event`'s fields, or a :obj:`list`
                representation if `as_list` is set
        """
        headers = Event.BASE_HEADERS.copy()
        if self.message.header() is not None:
            headers.extend(self.message.header(as_list=True))
        if as_list:
            return headers
        else:
            return separator.join(headers)
