from __future__ import print_function

from recordtype import recordtype

"""
Simulation event structure

Created 2016/05/31
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

# each event is represented by an Event record
# the event_body is the payload, opaque content that can only be interpreted by the sending and receiving objects
Event = recordtype( 'Event', 'creation_time event_time sending_object receiving_object event_type event_body' )

def Event_str( Event, header=False ):
    """Return an Event as a string, or an Event table header.
    
    To generate the returned string, it is assumed that sending_object and receiving_object
    support a name() method that returns their name. 
    
    Args:
        header: boolean; if True, return an Event table header
        
    Return:
        String representation of the Event's generic fields.
    
    """
    # TODO(Arthur): allow formatting of the returned string, e.g. formatting the precision of time values
    # TODO(Arthur): output the event_body, by assuming it supports str()

    if header:
        return '\t'.join( 'creation_time event_time sending_object receiving_object event_type'.split() ) 
    return '\t'.join( [ '{:8.3f}'.format( Event.creation_time ), '{:8.3f}'.format( Event.event_time ), 
        Event.sending_object.name(), Event.receiving_object.name(), Event.event_type ] )
