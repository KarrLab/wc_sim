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
    `event_body` attributes must be structured as specified in the `message_types` module.

    Attributes:
        creation_time (:obj:`float`): simulation time when the event is created (aka `send_time`)
        event_time (:obj:`float`): simulation time when the event must be executed (aka `receive_time`)
        sending_object (:obj:`SimulationObject`): reference to the object that sends the event
        receiving_object (:obj:`SimulationObject`): reference to the object that receives (aka executes)
            the event
        # TODO(Arthur): FIX
        event_type (:obj:`str`): the event's type; the name of a class declared in `message_types`
        # TODO(Arthur): FIX
        event_body (:obj:`object`): reference to a `body` subclass of a message type declared in
            `message_types`; the message payload
    """
    # TODO(Arthur): for performance, perhaps pre-allocate and reuse events

    # use __slots__ to save space
    # TODO(Arthur): figure out how to stop Sphinx from documenting these __slots__ as attributes
    __slots__ = "creation_time event_time sending_object receiving_object event_type event_body".split()
    BASE_HEADERS = ['t(send)', 't(event)', 'Sender', 'Receiver', 'Event type']

    def __init__(self, creation_time, event_time, sending_object, receiving_object, event_type,
        event_body=None):
        self.creation_time = creation_time
        self.event_time = event_time
        self.sending_object = sending_object
        self.receiving_object = receiving_object
        self.event_type = event_type
        self.event_body = event_body

    def __lt__(self, other):
        """ Does this `Event` occur earlier than `other`?

        Attributes:
            other (:obj:`Event`): another `Event`

        Returns:
            :obj:`bool`: `True` if this `Event` occurs earlier than `other`
        """
        return self.event_time < other.event_time

    def __le__(self, other):
        """ Does this `Event` occur earlier or at the same time as `other`?

        Attributes:
            other (:obj:`Event`): another `Event`

        Returns:
            :obj:`bool`: `True` if this `Event` occurs earlier or at the same time as `other`
        """
        return self.event_time <= other.event_time

    def __gt__(self, other):
        """ Does this `Event` occur later than `other`?

        Attributes:
            other (:obj:`Event`): another `Event`

        Returns:
            :obj:`bool`: `True` if this `Event` occurs later than `other`
        """
        return self.event_time > other.event_time

    def __ge__(self, other):
        """ Does this `Event` occur later or at the same time as `other`?

        Attributes:
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
            :obj:`str`: String representation of the `Event`'s fields, except `event_body`,
                delimited by tabs
        """
        # TODO(Arthur): allow formatting of the returned string, e.g. formatting the precision of time values
        vals = [round_direct(self.creation_time), round_direct(self.event_time),
            self.sending_object.name, self.receiving_object.name, self.event_type]
        if self.event_body is not None:
            vals.append(self.event_body.delimited())
        return '\t'.join(vals)

    @staticmethod
    def header():
        """ Return a header for an Event table

        Returns:
            :obj:`str`: String representation of names of an `Event`'s fields, delimited by tabs
        """
        return '\t'.join(Event.BASE_HEADERS + ['Event body fields...'])

    def custom_header(self):
        """ Return a header for an Event table containing messages of a particular type

        Returns:
            :obj:`str`: String representation of names of an `Event`'s fields, delimited by tabs
        """
        headers = Event.BASE_HEADERS.copy()
        if self.event_body is not None and self.event_body.header() is not None:
            headers.append(self.event_body.header())
        return '\t'.join(headers)
