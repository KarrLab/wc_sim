"""Simulation event structure.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-05-31
:Copyright: 2016, Karr Lab
:License: MIT
"""

class Event( object ):
    """An object that holds a discrete event simulation (DES) event.

    Each DES event is scheduled by an Event instance. To reduce interface errors event_type and
    event_body attributes are structured as specified in the MessageTypes module.

    Attributes:
        creation_time (float): simulation time when the event is created (aka send_time)
        event_time (float): simulation time when the event must be executed (aka receive_time)
        sending_object (`SimulationObject`): reference to the object that sends the event
        receiving_object (`SimulationObject`): reference to the object that receives (aka executes)
            the event
        event_type (string): the event's type; the name of a class declared in MessageTypes
        event_body (`obj`): reference to a body subclass of a message type declared in MessageTypes;
            the message payload
    """
    # TODO(Arthur): for performance, perhaps pre-allocate and reuse events
    # use __slots__ to save space
    __slots__ = "creation_time event_time sending_object receiving_object event_type event_body".split()
    def __init__( self, creation_time, event_time, sending_object, receiving_object, event_type,
        event_body=None):
        self.creation_time = creation_time
        self.event_time = event_time
        self.sending_object = sending_object
        self.receiving_object = receiving_object
        self.event_type = event_type
        self.event_body = event_body

    def __lt__(self, other):
        self.event_time < other.event_time

    def __le__(self, other):
        self.event_time <= other.event_time

    def __gt__(self, other):
        self.event_time > other.event_time

    def __ge__(self, other):
        self.event_time >= other.event_time


    def __str__( self ):
        """Return an Event as a string.

        To generate the returned string, it is assumed that sending_object and receiving_object
        contain a name attribute.

        Return:
            String representation of the Event's generic fields.

        """
        # TODO(Arthur): allow formatting of the returned string, e.g. formatting the precision of time values
        # TODO(Arthur): optionally, output the event_body, first checking that it supports str()

        return '\t'.join( [ '{:8.3f}'.format( self.creation_time ), '{:8.3f}'.format( self.event_time ),
            self.sending_object.name, self.receiving_object.name, self.event_type ] )

    @staticmethod
    def header( ):
        """Return a header for an Event table.

        Return:
            String Event table header.

        """
        return '\t'.join( 'creation_time event_time sending_object receiving_object event_type'.split() )

