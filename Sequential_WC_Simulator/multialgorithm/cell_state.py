"""
A species counts cell state encapsulated in a SimulationObject.

Created 2016/07/19
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""
import sys

from wc_utils.util.MiscUtilities import compare_name_with_class, dict_2_key_sorted_str

from Sequential_WC_Simulator.core.simulation_object import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.simulation_engine import MessageTypesRegistry

from Sequential_WC_Simulator.multialgorithm.message_types import *
from Sequential_WC_Simulator.multialgorithm.specie import Specie

    
class CellState( SimulationObject ): 
    """The cell's state, which represents the state of its species.
    
    CellState tracks the population of a set of species. After initialization, population values 
    are incremented or decremented, not written. To enable multi-algorithmic modeling, 
    it supports adjustments to a specie's population by both discrete and continuous models. E.g., discrete models
    might synthesize proteins while a continuous model degrades them.
    
    Attributes:
        population: a set species, represented by Specie objects
    
    Event messages:

        AdjustPopulationByDiscreteModel
        AdjustPopulationByContinuousModel
        GetPopulation
        
        See message descriptions in MessageTypes.
        At any time instant, ADJUST_POPULATION_* has priority over GetPopulation
        
    # TODO(Arthur): extend beyond population, e.g., to represent binding sites for individual macromolecules
    # TODO(Arthur): optimize to represent just the state of shared species
    # TODO(Arthur): abstract the functionality of CellState so it can be integrated into a submodel
    # TODO(Arthur): report error if a Specie instance is updated by multiple continuous sub-models
    """
    
    SENT_MESSAGE_TYPES = [ GivePopulation ]
    # TODO(Arthur): can Python automatically get the object name (e.g. 'CellState') here?
    MessageTypesRegistry.set_sent_message_types( 'CellState', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        AdjustPopulationByDiscreteModel, 
        AdjustPopulationByContinuousModel, 
        GetPopulation ]

    MessageTypesRegistry.set_receiver_priorities( 'CellState', MESSAGE_TYPES_BY_PRIORITY )

    def __init__( self, name, initial_population, initial_fluxes=None ):
        """Initialize a CellState object.
        
        Initialize a CellState object. Establish its initial population, and set debugging booleans.
        
        Args:
            name: string; name of this simulation object
            initial_population: initial population for some species;
                dict: specie_name -> initial_population
                species that are not initialized in this argument are assumed to be initialized with no population
                # TODO(Arthur): evalaute whether this is a good assumption
            initial_fluxes: optional; dict: specie_name -> initial_flux; 
                initial fluxes for all species whose populations are estimated by a continuous model
                fluxes are ignored for species not specified in initial_population

        Raises:
            AssertionError: if the population cannot be initialized.
        """
        super( CellState, self ).__init__( name )

        self.population = {}
        try:
            for specie_name in initial_population:
                initial_flux_given = None
                if initial_fluxes is not None and specie_name in initial_fluxes:
                    initial_flux_given = initial_fluxes[specie_name]
                self.population[specie_name] = Specie( specie_name, initial_population[specie_name], 
                    initial_flux=initial_flux_given )
        except AssertionError as e:
            sys.stderr.write( "Cannot initialize CellState: {}.\n".format( e.message ) )

        # log initialization data
        self.log_with_time( "initial_population: {}".format( dict_2_key_sorted_str(
            initial_population ) ) )
        self.log_with_time( "initial_fluxes: {}".format( dict_2_key_sorted_str( initial_fluxes ) ) )


    def handle_event( self, event_list ):
        """Handle a CellState simulation event.
        
        If an event message adjusts the population of a specie that is not known, initialize the Specie with
        no population. 
        
        # TODO(Arthur): More desc. 
        
        Args:
            event_list: list of event messages to process
            
        Raises:
            ValueError: if a GetPopulation message requests the population of an unknown species
            ValueError: if an AdjustPopulationByContinuousModel event acts on a non-existent species

        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        super( CellState, self ).handle_event( event_list )
        
        for event_message in event_list:
            # switch/case on event message type
            if compare_name_with_class( event_message.event_type, AdjustPopulationByDiscreteModel ):

                # population_changes is an AdjustPopulationByDiscreteModel body attribute
                population_changes = event_message.event_body.population_change

                self.log_with_time(
                    "AdjustPopulationByDiscreteModel: {}".format( str(event_message.event_body) ) ) 
                for specie_name in population_changes:
                    if not specie_name in self.population:
                        self.population[specie_name] = Specie( specie_name, 0 )
                    
                    self.population[specie_name].discrete_adjustment( 
                        population_changes[specie_name] )
                    self.log_event( 'discrete_adjustment', self.population[specie_name] )
                    
            elif compare_name_with_class( event_message.event_type, AdjustPopulationByContinuousModel ):
            
                # population_changes is an AdjustPopulationByContinuousModel body attribute
                population_changes = event_message.event_body.population_change
                self.log_with_time(
                    "AdjustPopulationByContinuousModel: {}".format( str(event_message.event_body) ) ) 

                for specie_name in population_changes:
                    # raise exeception if an AdjustPopulationByContinuousModel event acts on a 
                    # non-existent species because the species has no flux, and we don't want a default flux
                    if not specie_name in self.population:
                        raise ValueError( "Error: {} message requests population of unknown species '{}' in {}".format(
                            AdjustPopulationByContinuousModel.__name__, specie_name, event_message ) )
                    
                    #(population_change, flux) = population_changes[specie_name]
                    self.population[specie_name].continuous_adjustment( 
                        population_changes[specie_name].change, 
                        self.time, 
                        population_changes[specie_name].flux )
                    self.log_event( 'continuous_adjustment', self.population[specie_name] )

            elif compare_name_with_class( event_message.event_type, GetPopulation ):

                # species is a GetPopulation body attribute
                species = event_message.event_body.species
                # detect species requested that are not stored by this CellState
                invalid_species = species - set( self.population.keys() )
                # TODO(Arthur): test this case:                
                if len( invalid_species ):
                    raise ValueError( "Error: {} message requests population of unknown species {} in {}".format(
                        GetPopulation.__name__,
                        str( list( invalid_species ) ),
                        str( event_message ) ) )

                # use Specie() function for determining population
                reported_population = {}
                for specie in species:
                    # always provide the current simulation time to get_population() because it is needed if the 
                    # specie has been updated by a continuous model
                    population = self.population[specie].get_population( time=self.time ) 
                    reported_population[ specie ] = population
                self.send_event( 0, event_message.sending_object, 
                    GivePopulation, event_body=GivePopulation.body(reported_population) )

            else:
                assert False, "Shouldn't get here - event_message.event_type should be covered in the "\
                "if statement above"

    def log_event( self, event_type, specie ):
        """Log simulation events that modify a specie's population.
        
        Log the simulation time, event type, specie population, and current flux for each simulation
        event message that adjusts the population.
        
        Args:
            event_type: string; description of the event's type
            specie: Specie object; the object whose adjustment is being logged
        """
        try:
            flux = specie.continuous_flux
        except AttributeError:
            flux = None
        values = [ event_type, specie.last_population, flux ]
        values = map( lambda x: str(x), values )
        # log Sim_time Adjustment_type New_population New_flux
        self.log_with_time( '\t'.join( values ), local_call_depth=1 )
