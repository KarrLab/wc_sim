"""A simulation application's static set of message types and their content.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-06-10
:Copyright: 2016, Karr Lab
:License: MIT

Declare
    1. For each message type which has an event_body, a class that represents the body

Attributes:
    These are global attributes, since MessageTypes is static.
    senders: dict: SimulationObject type -> list of messages it sends
    receiver_priorities: dict: SimulationObject type -> list of messages it receives, in priority order

Event message types, bodies and reply message:
    AdjustPopulationByDiscreteModel: a discrete (stochastic) model increases or decreases some species copy
        numbers: data: dict: species_name -> population_change; no reply message
    AdjustPopulationByContinuousModel: a continuous model integrated by a time-step simulation increases or
        decreases some species copy numbers: 
        data: dict: species_name -> (population_change, population_flux); no reply message
    GetPopulation: list of species whose population is needed; data: set: species_name(s)
    GivePopulation: response to GetPopulation; dict: species_name -> population
    
    For sequential simulator, store message bodies as a copy of or reference to sender's data structure
    # TODO(Arthur): for parallel simulation, use Pickle to serialize and deserialize message bodies

"""

'''
Define a class that stores the body of each message type. This avoids confusing the structure of a message body.
These classes should be used by all message senders and receivers.
It is enforced by checking class names against message body types.

# TODO(Arthur): these class names should be changed to standard class name CamelCase format
'''

from collections import namedtuple

class AdjustPopulationByDiscreteModel( object ):
    """An AdjustPopulationByDiscreteModel message.
    
        Attributes:
            population_change: dict: species_name -> population_change; the increase or decrease 
            in some species copy numbers
    """
    
    class Body(object):
            
        __slots__ = ["population_change"]
        def __init__( self, population_change  ):
            self.population_change = population_change
    
        def add( self, specie, adjustment  ):
            self.population_change[ specie ] = adjustment
    
        def __str__( self  ):
            '''Return string representation of an AdjustPopulationByDiscreteModel message body.
                specie:change: name1:change1, name2:change2, ...
            '''
            l = [ "{}:{:.1f}".format( x, self.population_change[x] ) for x in \
                sorted( self.population_change.keys() ) ]
            return "specie:change: {}".format( ', '.join( l ) )

ContinuousChange_namedtuple = namedtuple( 'ContinuousChange_namedtuple', 'change, flux' )
class ContinuousChange(ContinuousChange_namedtuple):
    def type_check(self):
        """Check that the fields in ContinuousChange are numbers.
        
        Raises:
            ValueError: if one of the fields is non-numeric.
        """
        # https://docs.python.org/2.7/library/collections.html#collections.namedtuple documents namedtuple
        # and this approach for extending its functionality
        for f in self._fields:
            v = getattr(self,f)
            if not ( isinstance( v, int ) or isinstance( v, float ) ):
                raise ValueError( "ContinuousChange.type_check(): {} is '{}' "
                    "which is not an int or float".format( f, v ) )        
                    
    def __new__( cls, change, flux ):
        """Initialize a ContinuousChange.
        
        Raises:
            ValueError: if some fields are not numbers.
        """
        self = super( ContinuousChange, cls ).__new__( cls, change, flux )
        self.type_check()
        return self


class AdjustPopulationByContinuousModel( object ):
    """An AdjustPopulationByContinuousModel message.
    
        Attributes:
            population_change: dict: species_name -> ContinuousChange namedtuple; 
            the increase or decrease in some species copy numbers, and the predicted
            future flux of the species (which may be just the historic flux)
    """

    class Body(object):
        __slots__ = ["population_change"]
    
        def __init__( self, population_change  ):
            self.population_change = population_change
    
        def add( self, specie, cont_change  ):
            """
                Arguments:
                    specie: string; a specie name
                    cont_change: ContinuousChange = namedtuple; the continuous change for the named tuple
            """
            self.population_change[ specie ] = cont_change
    
        def __str__( self  ):
            '''Return string representation of an AdjustPopulationByContinuousModel message body.
                specie:(change,flux): name1:(change1,flux1), name2:(change2,flux2), ...
            '''
            l = [ "{}:({:.1f},{:.1f})".format( 
                x, self.population_change[x].change, self.population_change[x].flux ) \
                for x in sorted( self.population_change.keys() ) ]
            return "specie:(change,flux): {}".format( ', '.join( l ) )

class GetPopulation( object ):
    """A GetPopulation message.

        Attributes:
            species: set of species_names; the species whose populations are requested
    """

    class Body(object):
        __slots__ = ["species"]
        def __init__( self, species  ):
            self.species = species
    
        def __str__( self  ):
            '''Return string representation of a GetPopulation message body.
                species: name1, name2, ...
            '''
            return "species: {}".format( ', '.join( list( self.species ) ) )

class GivePopulation( object ):
    """A GivePopulation message.

        Attributes:
            population: dict: species_name -> population; the copy numbers of some species 
    """

    class Body(object):
        __slots__ = ["population"]
        def __init__( self, population ):
            self.population = population
    
        def __str__( self  ):
            '''Return string representation of a GivePopulation message body.
                specie:population: name1:pop1, name2:pop2, ...
            '''
            l = [ "{}:{:.1f}".format( x, self.population[x] ) \
                for x in sorted( self.population.keys() ) ]
            return "specie:population: {}".format( ', '.join( l ) )

class ExecuteSsaReaction( object ):
    """A ExecuteSsaReaction message.

        Attributes:
            reaction_index: integer; the index of the selected reaction in SsaSubmodel.reactions
    """

    class Body(object):
        __slots__ = ["reaction_index"]
        def __init__( self, reaction_index ):
            self.reaction_index = reaction_index

class SsaWait( object ):
    """A SsaWait message.
    """
    pass

class RunFba( object ):
    """A RunFba message.
    """
    pass


ALL_MESSAGE_TYPES = [
    AdjustPopulationByDiscreteModel,
    AdjustPopulationByContinuousModel,
    GetPopulation,
    GivePopulation,
    ExecuteSsaReaction,
    SsaWait,
]
