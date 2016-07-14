from collections import namedtuple

"""
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

# TODO(Arthur): IMPORTANT: disentangle the simulation-generic from the WC-simulation specific parts MessageTypes
class MessageTypes(object):
    """A simulation application's static set of message types and their content.
    
    Declare
        1. Event message types as string constants
        2. For each message type which has an event_body, a class that represents the body
    
    Attributes:
        These are global attributes, since MessageTypes is static.
        senders: dict: SimulationObject type -> list of messages it sends
        receiver_priorities: dict: SimulationObject type -> list of messages it receives, in priority order
    """
    
    ADJUST_POPULATION_BY_DISCRETE_MODEL = 'ADJUST_POPULATION_BY_DISCRETE_MODEL'
    ADJUST_POPULATION_BY_CONTINUOUS_MODEL = 'ADJUST_POPULATION_BY_CONTINUOUS_MODEL'
    GET_POPULATION = 'GET_POPULATION'
    GIVE_POPULATION = 'GIVE_POPULATION'
    
    senders = {}
    receiver_priorities = {}
    
    @staticmethod
    def set_sent_message_types( sim_obj_name, message_types ):
        MessageTypes.senders[ sim_obj_name ] = message_types
    
    @staticmethod
    def set_receiver_priorities( sim_obj_name, message_priorities ):
        MessageTypes.receiver_priorities[ sim_obj_name ] = message_priorities
    
    

'''
Define a class that stores the body of each message type. This avoids confusing the structure of a message body.
These classes should be used by all message senders and receivers.
# TODO(Arthur): IMPORTANT: enforce this with by type checking message bodies against message types
'''
class ADJUST_POPULATION_BY_DISCRETE_MODEL_body( object ):
    """Body of an ADJUST_POPULATION_BY_DISCRETE_MODEL message.
    
        Attributes:
            population_change: dict: species_name -> population_change; the increase or decrease 
            in some species copy numbers
    """

    # use __slots__ to save space
    __slots__ = ["population_change"]
    def __init__( self, population_change  ):
        self.population_change = population_change

    def add( self, specie, adjustment  ):
        self.population_change[ specie ] = adjustment

    def __str__( self  ):
        '''Return string representation of an ADJUST_POPULATION_BY_DISCRETE_MODEL message body.
            specie:change: name1:change1, name2:change2, ...
        '''
        l = map( lambda x: "{}:{:.1f}".format( x, self.population_change[x] ), 
            self.population_change.keys() )
        return "specie:change: {}".format( ', '.join( l ) )

Continuous_change = namedtuple( 'Continuous_change', 'change, flux' )
class ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body( object ):
    """Body of an ADJUST_POPULATION_BY_CONTINUOUS_MODEL message.
    
        Attributes:
            population_change: dict: species_name -> Continuous_change namedtuple; 
            the increase or decrease in some species copy numbers, and the predicted
            future flux of the species (which may be just the historic flux)
    """

    # use __slots__ to save space
    __slots__ = ["population_change"]

    # TODO(Arthur): IMPORTANT: make sure that change and flux are floats, otherwise __str__ will fail
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
        l = map( lambda x: "{}:({:.1f},{:.1f})".format( 
            x, self.population_change[x].change, self.population_change[x].flux ), 
            self.population_change.keys() )
        return "specie:(change,flux): {}".format( ', '.join( l ) )


# TODO(Arthur): MOVE to unittests
'''
d={
'x':Continuous_change(4,.5) ,
'y':Continuous_change(3,1) 
}
t=ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body( d )
print t
'''

class GET_POPULATION_body( object ):
    """Body of a GET_POPULATION message.

        Attributes:
            species: set of species_names; the species whose populations are requested
    """

    # use __slots__ to save space
    __slots__ = ["species"]
    def __init__( self, species  ):
        self.species = species

    def __str__( self  ):
        '''Return string representation of a GET_POPULATION message body.
            species: name1, name2, ...
        '''
        return "species: {}".format( ', '.join( list( self.species ) ) )

class GIVE_POPULATION_body( object ):
    """Body of a GIVE_POPULATION message.

        Attributes:
            population: dict: species_name -> population; the copy numbers of some species 
    """

    # use __slots__ to save space
    __slots__ = ["population"]
    def __init__( self, population ):
        self.population = population

    def __str__( self  ):
        '''Return string representation of a GIVE_POPULATION message body.
            specie:population: name1:pop1, name2:pop2, ...
        '''
        l = map( lambda x: "{}:{:.1f}".format( x, self.population[x] ), self.population.keys() )
        return "specie:population: {}".format( ', '.join( l ) )

'''
# TODO(Arthur): MOVE to unittests
v={ 'x':5, 'y':222}
t=GIVE_POPULATION_body( v )
print t
'''