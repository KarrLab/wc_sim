#!/usr/bin/env python

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
import MessageTypes

"""
The cell's state, which represents the state of its species.
"""
# TODO(Arthur): COVERAGE TESTING

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

