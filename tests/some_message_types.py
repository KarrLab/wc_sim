# example message types:
class InitMsg( object ):
    """A InitMsg message.
    """

    class Body(object):
        __slots__ = ["reaction_index"]
        def __init__( self, reaction_index ):
            self.reaction_index = reaction_index

class Test1( object ):
    pass
