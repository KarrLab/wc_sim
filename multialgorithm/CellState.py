#!/usr/bin/env python

import sys

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)
    
"""
The cell's state, which represents the state of its species.
"""

class Specie(object):
    """
    CellState tracks the population of a set of species. After initialization, population values 
    are incremented or decremented, not written. To enable multi-algorithmic modeling, 
    it supports adjustments to a specie's population by both discrete and continuous models. E.g., discrete models
    might synthesize proteins while a continuous model degrades them.
    We have these cases. Suppose that a specie's population is adjusted by:
        DISCRETE model only: estimating the population obtains the last value written
        CONTINUOUS model only: estimating the population obtains the last value written plus an interpolated change
            since the last continuous adjustment
        Both model types: reading the populations obtains the last value written plus an interpolated change based on
            the last continuous adjustment
    Without loss of generality, we assume that all species can be adjusted by both model types and that at most
    one continuous model adjusts a specie's population. (Multiple continuous models of a specie - or any 
    model attribute - would be non-sensical, as it implies multiple conflicting, simultaneous rates of change.) 
    We also assume that if a population is adjusted by a continuous model then the adjustments occur sufficiently 
    frequently that the last adjustment alway provides a useful estimate of flux. Adjustments take the following form:
        DISCRETE adjustment: (time, population_change)
        CONTINUOUS adjustment: (time, population_change, flux_estimate_at_time)
    
    Each Specie stores the (time, population) of the most recent adjustment and (time, flux) of the most
    recent continuous adjustment. Naming these R_time, R_pop, C_time, and C_flux, the population p at time t is 
    estimated by:
        interp = 0
        if C_time:
            interp = (t - C_time)*C_flux
        p = R_pop + interp

    This approach is completely general, and can be applied to any simulation value.
    
    Attributes:
        last_population: population after the most recent adjustment
        continuous_flux: flux provided by the most recent adjustment by a continuous model, 
            if there has been an adjustment by a continuous model; otherwise, uninitializes
        continuous_time: time of the most recent adjustment by a continuous model; None if the specie has not
            received a continuous adjustment
            
        Specie objects do not include the specie's name, as we assume they'll be stored in structures
        which contain the names.
    
    # TODO(Arthur): optimization: put Specie functionality into CellState, avoiding overhead of a Class instance
        for each Specie. I'm writing it in a class now while the math and logic are being developed.
    """
    # use __slots__ to save space
    __slots__ = "last_population continuous_time continuous_flux".split()

    def __init__( self, initial_population  ):
        """Initialize a Specie object.
        
        Args:
            initial_population: number; initial population of the specie
        """
        assert 0 <= initial_population, '__init__(): population should be >= 0'
        self.last_population = initial_population
        self.continuous_time = None
      
    def discrete_adjustment( self, population_change ):
        """A discrete model adjusts the specie's population.

        Args:
            population_change: number; modeled increase or decrease in the specie's population
            
        Raises:
            ValueError: if population goes negative
        """
        # TODO(Arthur): optimization: disable test in production, perhaps
        if self.last_population + population_change < 0:
            raise ValueError( "discrete_adjustment(): negative population from "
                "self.last_population + population_change ({:.2f} + {:.2f})\n".format( 
                self.last_population, population_change ) )
        self.last_population += population_change
    
    def continuous_adjustment( self, population_change, time, flux ):
        """A continuous model adjusts the specie's population.

        Args:
            population_change: number; modeled increase or decrease in the specie's population
            
        Raises:
            ValueError: if time is <= the time of the most recent continuous adjustment
            ValueError: if population goes negative
        """
        # multiple continuous adjustments at a time that does not advance the specie's state do not make sense
        if time <= self.continuous_time:
            raise ValueError( "continuous_adjustment(): time <= self.continuous_time: {:.2f} < {:.2f}\n".format( 
                time, self.continuous_time ) )
        if self.last_population + population_change < 0:
            raise ValueError( "continuous_adjustment(): negative population from "
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
                time is required if the specie has had a continuous adjustment
            
        Raises:
            ValueError: if time is required and not provided
            ValueError: time is earlier than a previous continuous adjustment
        """
        if not self.continuous_time:
            return self.last_population
        else:
            if not time:
                raise ValueError( "get_population(): time needed because "
                    "continuous adjustment received at time {:.2f}.\n".format( self.continuous_time ) )
            if time < self.continuous_time:
                raise ValueError( "get_population(): time < self.continuous_time: {:.2f} < {:.2f}\n".format( 
                    time, self.continuous_time ) )
            interpolation = (time - self.continuous_time) * self.continuous_flux
            return self.last_population + interpolation

    
class CellState( SimulationObject ): 
    """The cell's state, which represents the state of its species.
    
    More desc. 
    
    Attributes:
        population: a set species, represented by Specie objects
        debug: whether to print debugging data
    
    Event messages:

        ADJUST_POPULATION_BY_DISCRETE_MODEL
        ADJUST_POPULATION_BY_CONTINUOUS_MODEL
        GET_POPULATION
        
        See message descriptions in MessageTypes.
        At any time instant, ADJUST_POPULATION_* has priority over GET_POPULATION
        
    # TODO(Arthur): extend beyond population, e.g., to represent binding sites for individual macromolecules
    # TODO(Arthur): optimize to represent just the state of shared species
    # TODO(Arthur): abstract the functionality of CellState so it can be integrated into a submodel
    
    """
    
    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
        MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
        MessageTypes.GET_POPULATION ]

    def __init__( self, name, initial_population, debug=False, write_plot_output=False):
        """Initialize a CellState object.
        
        More desc. 
        
        Args:
            name: 
            initial_population: initial population for some species;
                dict: specie -> population
                species that are not initialized in this argument are assumed to be initialized with no population
                # TODO(Arthur): evalaute whether this is a good assumption
            
        Raises:

        # TODO(Arthur): complete pydoc
        """
        super( CellState, self ).__init__( name, plot_output=write_plot_output )

        self.population = {}
        try:
            for specie_name in initial_population.keys():
                self.population[specie_name] = Specie( initial_population[specie_name] )
        except AssertionError as e:
            sys.stderr.write( "Cannot initialize CellState: {}.\n".format( e.message ) )

        self.debug = debug

    def handle_event( self, event_list ):
        """Handle a simulation event.
        
        If an event message adjusts the population of a specie that is not known, initialize it with no population.
        
        More desc. 
        
        Args:
            event_list: list of event messages to process
            
        Raises:

        # TODO(Arthur): complete pydoc
        """
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
            if event_message.event_type == MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL:

                # population_changes is an ADJUST_POPULATION_BY_DISCRETE_MODEL_body object
                population_changes = event_message.event_body
                if self.debug:
                    print( "ADJUST_POPULATION_BY_DISCRETE_MODEL: {}".format( str(population_changes) ) )
                for specie_name in population_changes.population_change.keys():
                    if not specie_name in self.population:
                        self.population[specie_name] = Specie( 0 )
                    
                    self.population[specie_name].discrete_adjustment( 
                        population_changes.population_change[specie_name] )
                    
            elif event_message.event_type == MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL:
            
                # population_changes is an ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body object
                population_changes = event_message.event_body
                if self.debug:
                    print( "ADJUST_POPULATION_BY_CONTINUOUS_MODEL: {}".format( str(population_changes) ) )
                for specie_name in population_changes.population_change.keys():
                    if not specie_name in self.population:
                        self.population[specie_name] = Specie( 0 )
                    
                    #(population_change, flux) = population_changes[specie_name]
                    self.population[specie_name].continuous_adjustment( 
                        population_changes.population_change[specie_name].change, 
                        self.time, 
                        population_changes.population_change[specie_name].flux )

            elif event_message.event_type == MessageTypes.GET_POPULATION:

                # species is an GET_POPULATION_body object
                species = event_message.event_body
                # detect species requested that are not stored by this CellState
                invalid_species = species.species - set( self.population.keys() )
                if len( invalid_species ):
                    raise ValueError( "Error: {} message requests population of unknown species {} in {}\n".format(
                        MessageTypes.GET_POPULATION,
                        str( list( invalid_species ) ),
                        str( event_message ) ) )

                # use Specie() function for determining population
                reported_population = {}
                for specie in species.species:
                    # always provide the current simulation time to get_population() because it is needed if the 
                    # specie has been updated by a continuous model
                    population = self.population[specie].get_population( time=self.time ) 
                    reported_population[ specie ] = population
                self.send_event( 0, event_message.sending_object, 
                    MessageTypes.GIVE_POPULATION, event_body=GIVE_POPULATION_body(reported_population) )

            else:
                assert False, "Shouldn't get here - event_message.event_type should be covered in the "
                "if statement above"
