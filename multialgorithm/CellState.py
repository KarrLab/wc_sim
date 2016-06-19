    #!/usr/bin/env python

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from MessageTypes import MessageTypes

"""
The cell's state, which represents the state of its species.
"""
# TODO(Arthur): COVERAGE TESTING

class Specie(object):
    """
    CellState tracks the population of a set of species. To enable multi-algorithmic modeling, 
    it supports updates to a specie's population by both discrete and continuous models. E.g., discrete models
    might synthesize proteins while a continuous model degrades them.
    We have these cases. Suppose that a specie's population is updated by:
        DISCRETE model only: estimating the population obtains the last value written
        CONTINUOUS model only: estimating the population obtains the last value written plus an interpolated change
            since the last continuous update
        Both model types: reading the populations obtains the last value written plus an interpolated change based on
            the last continuous update
    Without loss of generality, we assume that all species can be updated by both model types and that at most
    one continuous model updates a specie's population. (Multiple continuous models of a specie - or any 
    model attribute - would be non-sensical, as it implies multiple conflicting, simultaneous rates of change.) 
    We also assume that if a population is updated by a continuous model then the updates occur sufficiently 
    frequently that the last update alway provides a useful estimate of flux. Updates take the following form:
        DISCRETE update: (time, population_change)
        CONTINUOUS update: (time, population_change, flux_estimate_at_time)
    
    Each Specie stores the (time, population) of the most recent update and (time, flux) of the most
    recent continuous update. Naming these R_time, R_pop, C_time, and C_flux, the population p at time t is 
    estimated by:
        interp = 0
        if C_time:
            interp = (t - C_time)*C_flux
        p = R_pop + interp

    This approach is completely general, and can be applied to any simulation value.
    
    Attributes:
        last_population: population after the most recent update
        continuous_flux: flux provided by the most recent update by a continuous model
        continuous_time: time of the most recent update by a continuous model; None if the specie has not
            received a continuous update
    
    # TODO(Arthur): optimization: put Specie functionality into CellState, avoiding overhead of a Class instance
        for each Specie. I'm writing it in a class now while the math and logic are being developed.
    """
    # use __slots__ to save space
    __slots__ = "last_population continuous_time continuous_flux".split()

    def __init__( self, initial_population  ):
        """Initialize a Specie object.
        
        Args:
            initial_population: float; initial population of the specie
        """
        assert 0 <= initial_population, '__init__(): population should be >= 0'
        self.last_population = initial_population
        self.continuous_time = None
      
    def discrete_update( self, population_change ):
        """A discrete model updates the Specie's population.

        Args:
            population_change: number; modeled change in the specie's population
            
        Raises:
            ValueError: if population goes negative
        """
        # TODO(Arthur): optimization: disable test in production, perhaps
        if self.last_population + population_change < 0:
            raise ValueError( "discrete_update(): negative population from "
                "self.last_population + population_change ({:.2f} + {:.2f})\n".format( 
                self.last_population, population_change ) )
        self.last_population += population_change
    
    def continuous_update( self, population_change, time, flux ):
        """A continuous model updates the specie's population.

        Args:
            population_change: number; modeled change in the specie's population
            
        Raises:
            ValueError: if time is <= the time of the most recent continuous update
            ValueError: if population goes negative
        """
        # updates at a time that does not advance the specie's state do not make sense
        if self.continuous_time and time <= self.continuous_time:
            raise ValueError( "continuous_update(): time <= self.continuous_time: {:.2f} < {:.2f}\n".format( 
                time, self.continuous_time ) )
        if self.last_population + population_change < 0:
            raise ValueError( "continuous_update(): negative population from "
                "self.last_population + population_change ({:.2f} + {:.2f})\n".format( 
                self.last_population, population_change ) )
        assert 0 <= time, 'negative time: {:.2f}'.format( time )
        self.continuous_time = time
        assert 0 <= flux, 'negative flux: {:.2f}'.format( flux )
        self.continuous_flux = flux
        self.last_population += population_change
    
    def get_population( self, time=None ):
        """Get the specie's population at time.
        
        Interpolate continuous values as described in the documentation of class Specie.
        
        Args:
            time: number; optional; simulation time of the request; 
                time is required if the specie has had a continuous update
            
        Raises:
            ValueError: if time is required and not provided
            ValueError: time is earlier than a previous continuous update
        """
        if not self.continuous_time:
            return self.last_population
        else:
            if not time:
                raise ValueError( "get_population(): time needed because "
                    "continuous updates received at time {:.2f}.\n".format( self.continuous_time ) )
            if time < self.continuous_time:
                raise ValueError( "get_population(): time < self.continuous_time: {:.2f} < {:.2f}\n".format( 
                    time, self.continuous_time ) )
            interpolation = (time - self.continuous_time) * self.continuous_flux
            return self.last_population + interpolation

    
class CellState( SimulationObject ): 
    """The cell's state, which represents the state of its species.
    
    Attributes:
        type: population type, either Discrete or Continuous; Continuous population values can be interpolated
        population: species and their copy numbers: dict: species_name -> population
        fluxes: the fluxes of species; only used for species represented by Continuous models: 
            flux is measured in population/sec
            dict: species_name -> (time_set, flux)
        debug: whether to print debugging data
    
    Events:

        # at a time instant, CHANGE_POPULATION has priority over GET_POPULATION
        CHANGE_POPULATION:
            apply delta changes in CHANGE_POPULATION to population

        GET_POPULATION:
            # GET_POPULATION.species is requested species
            send POPULATION( now, GET_POPULATION.sender, population[GET_POPULATION.species] )   

    # TODO(Arthur): also previous time step for ODE continuous populations
    # TODO(Arthur): think about whether different types should be subclasses
        
    Contains a species instance with a copy number for each metabolite, and 
    binding site for each macromolecule.

    # TODO(Arthur): optimize to represent just the state of shared species
    # TODO(Arthur): abstract the functionality of CellState so it can be integrated into a submodel
    
    """
    
    # CellState types
    DISCRETE_TYPE = 'DISCRETE'
    CONTINUOUS_TYPE = 'CONTINUOUS'
    TYPES = [ DISCRETE_TYPE, CONTINUOUS_TYPE ]
    
    # at a time instant, CHANGE_POPULATION has priority over GET_POPULATION
    MESSAGE_TYPES_BY_PRIORITY = [ MessageTypes.CHANGE_POPULATION, MessageTypes.GET_POPULATION ]

    def __init__( self, name, initial_population, type, debug=False, write_plot_output=False):
        """Initialize a CellState object.
        
        Args:
            
        Raises:
            ValueError: if type is not in CellState.TYPES

        # TODO(Arthur): complete pydoc
        """
        super( CellState, self ).__init__( name, plot_output=write_plot_output )
        self.population = dict( initial_population )    # create new dict, rather than maintain reference
        # do not construct an illegal CellState
        if type not in CellState.TYPES:
            raise ValueError( "type='{}' not in TYPES {}.\n".format( type, str( CellState.TYPES ) ) )
        self.type = type
        self.debug = debug

    def write_state_variable( self, attribute, value ):
        """write a new value into an attribute.
        
        # TODO(Arthur): probably replace with 'setter' methods for specific attributes; exec is unsafe, and slow
        """
        exec "self.{} = {}".format( attribute, value ) 
        
    def handle_event( self, event_list ):
        """Handle a simulation event."""
        # call handle_event() in class SimulationObject which might produce plotting output or do other things
        super( CellState, self ).handle_event( event_list )
        
        # TODO(Arthur): use logging instead
        if self.debug:
            self.print_event_queue( )
        
        # check for messages with invalid types
        # TODO(Arthur): do this checking at send time, probably in SimulationObject.send_event()
        invalid_types = (set( map( lambda x: x.event_type, event_list ) ) - 
            set( CellState.MESSAGE_TYPES_BY_PRIORITY ))
        if len( invalid_types ):
            raise ValueError( "Error: invalid event event_type(s) '{}' in event_list:\n{}\n".format( 
                ', '.join( list( invalid_types ) ),
                '\n'.join( [ str( ev_msg ) for ev_msg in event_list ] ) ) )
        
        # sort event_list by type priority, anticipating non-deterministic arrival order in a parallel implementation
        # this scales for arbitrarily many message types
        for event_message in sorted( event_list, 
            key=lambda event: CellState.MESSAGE_TYPES_BY_PRIORITY.index( event.event_type ) ):
            # switch/case on event message type
            if event_message.event_type == MessageTypes.CHANGE_POPULATION:

                if self.type == CellState.DISCRETE_TYPE:
                    population_changes = event_message.event_body
                    if self.debug:
                        print( "population_changes: {}".format( str(population_changes) ) )
                    for species_name in population_changes.keys():
                        if species_name in self.population:
                            self.population[ species_name ] += population_changes[ species_name ]
                    
                elif self.type == CellState.CONTINUOUS_TYPE:
                    # TODO(Arthur): do this
                    pass
                else:
                    assert False, "Shouldn't get here - CellState type should be covered in the if statement above"
            
            elif event_message.event_type == MessageTypes.GET_POPULATION:
            
                species = event_message.event_body
                # detect species not stored by this CellState
                invalid_species = set( self.population.keys() ) - set( species )
                if len( invalid_species ):
                    raise ValueError( "Error: {} message requests population of unknown species {} in {}\n".format(
                        MessageTypes.GET_POPULATION,
                        str( list( invalid_species ) ),
                        str( event_message ) ) )

                # give current value for discrete data, interpolate for continuous data
                if self.type == CellState.DISCRETE_TYPE:
                    response_dict = {specie: self.population[specie] for specie in species}
                    self.send_event( 0, event_message.sending_object.name, 
                        MessageTypes.GIVE_POPULATION, event_body=response_dict )

                elif self.type == CellState.CONTINUOUS_TYPE:
                    response_dict = {}  # TODO(Arthur): do this
                    self.send_event( 0, event_message.sending_object.name, 
                        MessageTypes.GIVE_POPULATION, event_body=response_dict )

                else:
                    assert False, "Shouldn't get here - CellState type should be covered in the if statement above"

            else:
                assert False, "Shouldn't get here - event_message.event_type should be covered in the "
                "if statement above"

