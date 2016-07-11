#!/usr/bin/env python

import sys
import math
import logging
logger = logging.getLogger(__name__)
# control logging level with: logger.setLevel()
# this enables debug output: logging.basicConfig( level=logging.DEBUG )

from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)
    
"""
The cell's state, which represents the state of its species.
"""

from random import Random

class StochasticRound( object ):
    """Stochastically round floating point values.
    
    A float is rounded to one of the two nearest integers. The mean of the rounded values for a set of floats
    converges to the mean of the floats. This is achieved by making P[rounding x down] = 1 - (x - floor(x) ), and
    P[rounding x up] = 1 - P[rounding x down].
    This avoids the bias that would arise from always using floor() or ceiling(), especially with 
    small populations.
    
    Attributes:
        RNG: A Random instance, initialized on creation of a StochasticRound.
    """

    def __init__( self, seed=None ):
        """Initialize a StochasticRound.
        
        Args:
            seed: a hashable object; optional; to deterministically initialize the basic random number generator 
            provide seed. Otherwise some system-dependent randomness source will be used to initialize 
            the generator. See Python documentation for random.seed().
        """
        if seed:
            self.RNG = Random( seed )
        else:
            self.RNG = Random( )
        
    def Round( self, x ):
        """Stochastically round a floating point value.
        
        Args:
            x: a float to be stochastically rounded.
            
        Returns:
            A stochastically round of x.
        """
        floor_x = math.floor( x )
        fraction = x - floor_x
        if 0==fraction:
            return x
        else:
            if self.RNG.random( ) < fraction:
                return floor_x + 1
            else:
                return floor_x
    
# TODO(Arthur): generate plots of copy number vs. time; better colors and symbols than in ppt

class Specie(object):
    """
    Specie tracks the population of a single specie.
    
    We have these cases. Suppose that the specie's population is adjusted by:
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
    
    A Specie stores the (time, population) of the most recent adjustment and (time, flux) of the most
    recent continuous adjustment. Naming these R_time, R_pop, C_time, and C_flux, the population p at time t is 
    estimated by:
        interp = 0
        if C_time:
            interp = (t - C_time)*C_flux
        p = R_pop + interp

    This approach is completely general, and can be applied to any simulation value.
    
    Real versus integer copy numbers: Clearly, specie copy number values are non-negative integers. However,
    continuous models may estimate copy numbers as real number values. E.g., ODEs calculate real valued concentrations.
    But SSA models do not naturally handle non-integral copy numbers, and models that represent the state of 
    individual species -- such as a bound molecule -- are not compatible with non-integral copy numbers. Therefore,
    Specie stores a real valued population, but reports only a non-negative integral population. 
    In particular, the population reported by get_population() is rounded. Rounding is done  stochastically to avoid the 
    bias that would arise from always using floor() or ceiling(), especially with small populations.
    
    Attributes:
        last_population: population after the most recent adjustment
        continuous_flux: flux provided by the most recent adjustment by a continuous model, 
            if there has been an adjustment by a continuous model; otherwise, uninitialized
        continuous_time: time of the most recent adjustment by a continuous model; None if the specie has not
            received a continuous adjustment
            
        Specie objects do not include the specie's name, as we assume they'll be stored in structures
        which contain the names.
    
    # TODO(Arthur): optimization: put Specie functionality into CellState, avoiding overhead of a Class instance
        for each Specie. I'm writing it in a class now while the math and logic are being developed.
    # TODO(Arthur): look at how COPASI handles 
    """
    # use __slots__ to save space
    __slots__ = "last_population continuous_time continuous_flux stochasticRounder".split()

    def __init__( self, initial_population, randomSeed=None  ):
        """Initialize a Specie object.
        
        Args:
            initial_population: number; initial population of the specie
            randomSeed: optional; seed for the random number generator;
                the generator is initialized as described in the random module documentation
        """
        # TODO(Arthur): important: optional initial continuous flux
        # TODO(Arthur): perhaps: add optional arg to not round copy number values reported
        assert 0 <= initial_population, '__init__(): population should be >= 0'
        self.last_population = initial_population
        self.continuous_time = None
        if randomSeed:
            self.stochasticRounder = StochasticRound( seed=randomSeed ).Round
        else:
            self.stochasticRounder = StochasticRound( ).Round

      
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
            return self.stochasticRounder( self.last_population )
        else:
            if time == None:
                raise ValueError( "get_population(): time needed because "
                    "continuous adjustment received at time {:.2f}.\n".format( self.continuous_time ) )
            if time < self.continuous_time:
                raise ValueError( "get_population(): time < self.continuous_time: {:.2f} < {:.2f}\n".format( 
                    time, self.continuous_time ) )
            interpolation = (time - self.continuous_time) * self.continuous_flux
            real_copy_number = self.last_population + interpolation
            return self.stochasticRounder( real_copy_number )

    
class CellState( SimulationObject ): 
    """The cell's state, which represents the state of its species.
    
    CellState tracks the population of a set of species. After initialization, population values 
    are incremented or decremented, not written. To enable multi-algorithmic modeling, 
    it supports adjustments to a specie's population by both discrete and continuous models. E.g., discrete models
    might synthesize proteins while a continuous model degrades them.

    # TODO(Arthur): More desc.
    
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
    
    SENT_MESSAGE_TYPES = [ MessageTypes.GIVE_POPULATION ]
    # TODO(Arthur): can Python automatically get the object name (e.g. 'CellState') here?
    MessageTypes.set_sent_message_types( 'CellState', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
        MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
        MessageTypes.GET_POPULATION ]

    MessageTypes.set_receiver_priorities( 'CellState', MESSAGE_TYPES_BY_PRIORITY )

    def __init__( self, name, initial_population, debug=False, write_plot_output=False):
        """Initialize a CellState object.
        
        Initialize a CellState object. Establish its initial population, and set debugging booleans.
        
        Args:
            name: 
            initial_population: initial population for some species;
                dict: specie -> population
                species that are not initialized in this argument are assumed to be initialized with no population
                # TODO(Arthur): evalaute whether this is a good assumption

        # TODO(Arthur): add an arg that creates all species with a specified random seed, so results are deterministic
            
        Raises:
            AssertionError: if the population cannot be initialized.
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
        """Handle a CellState simulation event.
        
        If an event message adjusts the population of a specie that is not known, initialize the Specie with
        no population. 
        
        More desc. 
        
        Args:
            event_list: list of event messages to process
            
        Raises:
            ValueError: if a GET_POPULATION message requests the population of an unknown species

        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        super( CellState, self ).handle_event( event_list )
        
        if self.debug:
            logger.debug( ' ' + self.event_queue_to_str() )
            
        for event_message in event_list:
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

                # species is a GET_POPULATION_body object
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
