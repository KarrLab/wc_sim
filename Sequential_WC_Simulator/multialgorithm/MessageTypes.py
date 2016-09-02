from collections import namedtuple

"""
A simulation application's static set of message types and their content.

Declare
    2. For each message type which has an event_body, a class that represents the body

Attributes:
    These are global attributes, since MessageTypes is static.
    senders: dict: SimulationObject type -> list of messages it sends
    receiver_priorities: dict: SimulationObject type -> list of messages it receives, in priority order

Event message types, bodies and reply message:
    ADJUST_POPULATION_BY_DISCRETE_MODEL: a discrete (stochastic) model increases or decreases some species copy
        numbers: data: dict: species_name -> population_change; no reply message
    ADJUST_POPULATION_BY_CONTINUOUS_MODEL: a continuous model integrated by a time-step simulation increases or
        decreases some species copy numbers: 
        data: dict: species_name -> (population_change, population_flux); no reply message
    GET_POPULATION: list of species whose population is needed; data: set: species_name(s)
    GIVE_POPULATION: response to GET_POPULATION; dict: species_name -> population
    
    For sequential simulator, store message bodies as a copy of or reference to sender's data structure
    # TODO(Arthur): for parallel simulation, use Pickle to serialize and deserialize message bodies

Created 2016/06/10
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""


'''
Define a class that stores the body of each message type. This avoids confusing the structure of a message body.
These classes should be used by all message senders and receivers.
It is enforced by checking class names against message body types.

# TODO(Arthur): these class names should be changed to standard class name CamelCase format
'''
class ADJUST_POPULATION_BY_DISCRETE_MODEL( object ):
    """An ADJUST_POPULATION_BY_DISCRETE_MODEL message.
    
        Attributes:
            population_change: dict: species_name -> population_change; the increase or decrease 
            in some species copy numbers
    """
    
    class body(object):
            
        __slots__ = ["population_change"]
        def __init__( self, population_change  ):
            self.population_change = population_change
    
        def add( self, specie, adjustment  ):
            self.population_change[ specie ] = adjustment
    
        def __str__( self  ):
            '''Return string representation of an ADJUST_POPULATION_BY_DISCRETE_MODEL message body.
                specie:change: name1:change1, name2:change2, ...
            '''
            l = [ "{}:{:.1f}".format( x, self.population_change[x] ) for x in \
                sorted( self.population_change.keys() ) ]
            return "specie:change: {}".format( ', '.join( l ) )

Continuous_change_namedtuple = namedtuple( 'Continuous_change_namedtuple', 'change, flux' )
class Continuous_change(Continuous_change_namedtuple):
    def type_check(self):
        """Check that the fields in Continuous_change are numbers.
        
        Raises:
            ValueError: if one of the fields is non-numeric.
        """
        # https://docs.python.org/2.7/library/collections.html#collections.namedtuple documents namedtuple
        # and this approach for extending its functionality
        for f in self._fields:
            v = getattr(self,f)
            if not ( isinstance( v, int ) or isinstance( v, float ) ):
                raise ValueError( "Continuous_change.type_check(): {} is '{}' "
                    "which is not an int or float".format( f, v ) )        
                    
    def __new__( cls, change, flux ):
        """Initialize a Continuous_change.
        
        Raises:
            ValueError: if some fields are not numbers.
        """
        self = super( Continuous_change, cls ).__new__( cls, change, flux )
        self.type_check()
        return self


class ADJUST_POPULATION_BY_CONTINUOUS_MODEL( object ):
    """An ADJUST_POPULATION_BY_CONTINUOUS_MODEL message.
    
        Attributes:
            population_change: dict: species_name -> Continuous_change namedtuple; 
            the increase or decrease in some species copy numbers, and the predicted
            future flux of the species (which may be just the historic flux)
    """

    class body(object):
        __slots__ = ["population_change"]
    
        def __init__( self, population_change  ):
            self.population_change = population_change
    
        def add( self, specie, cont_change  ):
            """
                Arguments:
                    specie: string; a specie name
                    cont_change: Continuous_change = namedtuple; the continuous change for the named tuple
            """
            self.population_change[ specie ] = cont_change
    
        def __str__( self  ):
            '''Return string representation of an ADJUST_POPULATION_BY_CONTINUOUS_MODEL message body.
                specie:(change,flux): name1:(change1,flux1), name2:(change2,flux2), ...
            '''
            l = [ "{}:({:.1f},{:.1f})".format( 
                x, self.population_change[x].change, self.population_change[x].flux ) \
                for x in sorted( self.population_change.keys() ) ]
            return "specie:(change,flux): {}".format( ', '.join( l ) )

class GET_POPULATION( object ):
    """A GET_POPULATION message.

        Attributes:
            species: set of species_names; the species whose populations are requested
    """

    class body(object):
        __slots__ = ["species"]
        def __init__( self, species  ):
            self.species = species
    
        def __str__( self  ):
            '''Return string representation of a GET_POPULATION message body.
                species: name1, name2, ...
            '''
            return "species: {}".format( ', '.join( list( self.species ) ) )

class GIVE_POPULATION( object ):
    """A GIVE_POPULATION message.

        Attributes:
            population: dict: species_name -> population; the copy numbers of some species 
    """

    class body(object):
        __slots__ = ["population"]
        def __init__( self, population ):
            self.population = population
    
        def __str__( self  ):
            '''Return string representation of a GIVE_POPULATION message body.
                specie:population: name1:pop1, name2:pop2, ...
            '''
            l = [ "{}:{:.1f}".format( x, self.population[x] ) \
                for x in sorted( self.population.keys() ) ]
            return "specie:population: {}".format( ', '.join( l ) )

class EXECUTE_SSA_REACTION( object ):
    """A EXECUTE_SSA_REACTION message.

        Attributes:
            reaction_index: integer; the index of the selected reaction in simple_SSA_submodel.reactions
    """

    class body(object):
        __slots__ = ["reaction_index"]
        def __init__( self, reaction_index ):
            self.reaction_index = reaction_index

class SSA_WAIT( object ):
    """A SSA_WAIT message.
    """
    pass

class RUN_FBA( object ):
    """A RUN_FBA message.
    """
    pass

