from __future__ import print_function

"""
Simulation event structure

Created 2016/05/31
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

# each event is represented by an Event record
# Event = recordtype( 'Event', 'creation_time event_time sending_object receiving_object event_type event_body' )

class Event( object ):
    """An object that holds DES events.
    
    the event_body is the payload, opaque content that can only be interpreted by the sending and receiving objects
    """
    # TODO(Arthur): make Event a slots object, to conserve space
    # TODO(Arthur): for performance, perhaps pre-allocate and reuse events
    def __init__( self, creation_time, event_time, sending_object, receiving_object, event_type, 
        event_body=None):
        self.creation_time = creation_time
        self.event_time = event_time
        self.sending_object = sending_object
        self.receiving_object = receiving_object
        self.event_type = event_type
        self.event_body = event_body

    def __str__( self ):
        """Return an Event as a string.
        
        To generate the returned string, it is assumed that sending_object and receiving_object
        contain a name attribute.
        
        Return:
            String representation of the Event's generic fields.
        
        """
        # TODO(Arthur): allow formatting of the returned string, e.g. formatting the precision of time values
        # TODO(Arthur): optionally, output the event_body, by assuming it supports str()
    
        return '\t'.join( [ '{:8.3f}'.format( self.creation_time ), '{:8.3f}'.format( self.event_time ), 
            self.sending_object.name, self.receiving_object.name, self.event_type ] )

    @staticmethod
    def header( ):
        """Return a header for an Event table.
        
        Return:
            String Event table header.
        
        """
        return '\t'.join( 'creation_time event_time sending_object receiving_object event_type'.split() ) 

