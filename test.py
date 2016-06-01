#!/usr/bin/env python

'''
Created 2016/05/31
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu

'''
from Event import Event, Event_str

class t(object):
    def __init__( self, name):
        self.my_name = name
    def name(self):
        return self.my_name
t1 = t( 'test1' )
t2 = t( 'test2' )
e = Event( 1.5, 3.7, t1, t2, 'test_type', None )
print Event_str( None, header=True )
print Event_str( e )
