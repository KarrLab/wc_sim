#!/usr/bin/env python

class BaseClass(object):
    # an abstract class
    def semiAbstractMethod( args ):
        # do some stuff here
        pass
    
class Engine(object):
    # Engine is a scheduler
    @staticmethod
    def schedule( args ):
        # execute methods in classes derived from BaseClass, like semiAbstractMethod
        pass