# example message types:
class init_msg( object ):
    """A init_msg message.
    """

    class body(object):
        __slots__ = ["reaction_index"]
        def __init__( self, reaction_index ):
            self.reaction_index = reaction_index

class test1( object ):
    pass
