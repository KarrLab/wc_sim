#!/usr/bin/env python

import sys
import logging
from Sequential_WC_Simulator.core.LoggingConfig import setup_logger

from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)
from Sequential_WC_Simulator.core.utilities import StochasticRound
    
"""
The cell's state, which represents the state of its species.
"""
    
class Specie(object):
    """
    Specie tracks the population of a single specie.
    
    We have these cases. Suppose that the specie's population is adjusted by:
        DISCRETE model only: estimating the population obtains the last value written
        CONTINUOUS model only: estimating the population obtains the last value written plus a linear interpolated change
            since the last continuous adjustment
        Both model types: reading the populations obtains the last value written plus a linear interpolated change 
            based on the last continuous adjustment
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
        interpolation = 0
        if C_time:
            interpolation = (t - C_time)*C_flux
        p = R_pop + interpolation

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
    __slots__ = "specie_name last_population continuous_time continuous_flux stochasticRounder".split()

    def __init__( self, specie_name, initial_population, initial_flux=None, random_seed=None ):
        """Initialize a Specie object.
        
        Args:
            specie_name: string; the specie's name; not logically needed, but may be helpful for logging, debugging, etc.
            initial_population: non-negative number; initial population of the specie
            initial_flux: number; required for Specie whose population is partially estimated by a 
                continuous mode; initial flux for the specie
            random_seed: hashable object; optional; seed for the random number generator;
                the generator is initialized as described in Python's random module documentation
        """
        
        # TODO(Arthur): perhaps: add optional arg to not round copy number values reported
        assert 0 <= initial_population, '__init__(): population should be >= 0'
        self.specie_name = specie_name
        self.last_population = initial_population
        if initial_flux == None:
            self.continuous_time = None
        else:
            self.continuous_time = 0
            self.continuous_flux = initial_flux
        if random_seed:
            self.stochasticRounder = StochasticRound( seed=random_seed ).Round
        else:
            self.stochasticRounder = StochasticRound( ).Round

    def discrete_adjustment( self, population_change ):
        """A discrete model adjusts the specie's population.

        Args:
            population_change: number; modeled increase or decrease in the specie's population
            
        Raises:
            ValueError: if population goes negative
        """
        # TODO(Arthur): perhaps: an optimization: disable test in production
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
            ValueError: if initial flux was not provided for a continuously updated variable
            ValueError: if time is <= the time of the most recent continuous adjustment
            ValueError: if population goes negative
            AssertionError: if the time is negative
        """
        if self.continuous_time == None:
            raise ValueError( "continuous_adjustment(): initial flux was not provided\n" )
        assert 0 <= time, 'negative time: {:.2f}'.format( time )
        # multiple continuous adjustments at a time that does not advance the specie's state do not make sense
        if time <= self.continuous_time:
            raise ValueError( "continuous_adjustment(): time <= self.continuous_time: {:.2f} < {:.2f}\n".format( 
                time, self.continuous_time ) )
        if self.last_population + population_change < 0:
            raise ValueError( "continuous_adjustment(): negative population from "
                "self.last_population + population_change ({:.2f} + {:.2f})\n".format( 
                self.last_population, population_change ) )
        self.continuous_time = time
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
    
    Attributes:
        population: a set species, represented by Specie objects
        debug: whether to log debugging data
    
    Event messages:

        ADJUST_POPULATION_BY_DISCRETE_MODEL
        ADJUST_POPULATION_BY_CONTINUOUS_MODEL
        GET_POPULATION
        
        See message descriptions in MessageTypes.
        At any time instant, ADJUST_POPULATION_* has priority over GET_POPULATION
        
    # TODO(Arthur): extend beyond population, e.g., to represent binding sites for individual macromolecules
    # TODO(Arthur): optimize to represent just the state of shared species
    # TODO(Arthur): abstract the functionality of CellState so it can be integrated into a submodel
    # TODO(Arthur): report error if a Specie instance is updated by multiple continuous sub-models
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

    def __init__( self, name, initial_population, initial_fluxes=None, 
        shared_random_seed=None, debug=False, write_plot_output=False, log=False):
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
            shared_random_seed: hashable object; optional; set to deterministically initialize all random number
                generators used by the Specie objects
            debug: boolean; log debugging output
            write_plot_output: boolean; log output for plotting simulation 
            log: boolean; log population dynamics of species
            # TODO(Arthur): remove debug and write_plot_output, incoporate into logging

        Raises:
            AssertionError: if the population cannot be initialized.
        """
        super( CellState, self ).__init__( name, plot_output=write_plot_output )

        self.population = {}
        try:
            for specie_name in initial_population.keys():
                initial_flux_given = None
                if initial_fluxes and specie_name in initial_fluxes:
                    initial_flux_given = initial_fluxes[specie_name]
                self.population[specie_name] = Specie( specie_name, initial_population[specie_name], 
                    initial_flux=initial_flux_given, random_seed=shared_random_seed )
        except AssertionError as e:
            sys.stderr.write( "Cannot initialize CellState: {}.\n".format( e.message ) )
            
        self.logger_name = "CellState_{}".format( name )
        if debug:
            # make a logger for this CellState
            # TODO(Arthur): eventually control logging when creating SimulationObjects, and pass in the logger
            setup_logger( self.logger_name, level=logging.DEBUG )
            mylog = logging.getLogger(self.logger_name)
            # write initialization data
            mylog.debug( "initial_population: {}".format( str(initial_population) ) )
            mylog.debug( "initial_fluxes: {}".format( str(initial_fluxes) ) )
            mylog.debug( "shared_random_seed: {}".format( str(shared_random_seed) ) )
            mylog.debug( "write_plot_output: {}".format( str(write_plot_output) ) )
            mylog.debug( "log: {}".format( str(log) ) )

        # if log, make a logger for each specie
        my_level = logging.NOTSET
        if log:
            my_level = logging.DEBUG
        for specie_name in initial_population.keys():
            setup_logger(specie_name, level=my_level )
            log = logging.getLogger(specie_name)
            # write log header
            log.debug( '\t'.join( 'Sim_time Adjustment_type New_population New_flux'.split() ) )
            # log initial state
            self.log_event( 'initial_state', self.population[specie_name] )


    def handle_event( self, event_list ):
        """Handle a CellState simulation event.
        
        If an event message adjusts the population of a specie that is not known, initialize the Specie with
        no population. 
        
        More desc. 
        
        Args:
            event_list: list of event messages to process
            
        Raises:
            ValueError: if a GET_POPULATION message requests the population of an unknown species
            ValueError: if an ADJUST_POPULATION_BY_CONTINUOUS_MODEL event acts on a non-existent species

        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        super( CellState, self ).handle_event( event_list )
        
        logging.getLogger( self.logger_name ).debug( self.event_queue_to_str() ) 

        for event_message in event_list:
            # switch/case on event message type
            if event_message.event_type == MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL:

                # population_changes is an ADJUST_POPULATION_BY_DISCRETE_MODEL_body object
                population_changes = event_message.event_body

                logging.getLogger( self.logger_name ).debug( 
                    "ADJUST_POPULATION_BY_DISCRETE_MODEL: {}".format( str(population_changes) ) ) 
                for specie_name in population_changes.population_change.keys():
                    if not specie_name in self.population:
                        self.population[specie_name] = Specie( specie_name, 0 )
                    
                    self.population[specie_name].discrete_adjustment( 
                        population_changes.population_change[specie_name] )
                    self.log_event( 'discrete_adjustment', self.population[specie_name] )
                    
            elif event_message.event_type == MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL:
            
                # population_changes is an ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body object
                population_changes = event_message.event_body
                logging.getLogger( self.logger_name ).debug( 
                    "ADJUST_POPULATION_BY_CONTINUOUS_MODEL: {}".format( str(population_changes) ) ) 

                for specie_name in population_changes.population_change.keys():
                    # raise exeception if an ADJUST_POPULATION_BY_CONTINUOUS_MODEL event acts on a 
                    # non-existent species because the species has no flux, and we don't want a default flux
                    if not specie_name in self.population:
                        raise ValueError( "Error: {} message requests population of unknown species '{}' in {}\n".format(
                            MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, specie_name, event_message ) )
                    
                    #(population_change, flux) = population_changes[specie_name]
                    self.population[specie_name].continuous_adjustment( 
                        population_changes.population_change[specie_name].change, 
                        self.time, 
                        population_changes.population_change[specie_name].flux )
                    self.log_event( 'continuous_adjustment', self.population[specie_name] )

            elif event_message.event_type == MessageTypes.GET_POPULATION:

                # species is a GET_POPULATION_body object
                species = event_message.event_body
                # detect species requested that are not stored by this CellState
                invalid_species = species.species - set( self.population.keys() )
                # TODO(Arthur): test this case:                
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

    def log_event( self, event_type, specie ):
        """Log simulation events that modify a specie's population.
        
        Log the simulation time, event type, specie population, and current flux for each simulation
        event message that adjusts the population. The log record is written to a separate log for each specie.
        
        Args:
            event_type: string; description of the event's type
            specie: Specie object; the object whose adjustment is being logged
        """
        log = logging.getLogger( specie.specie_name )
        try:
            flux = specie.continuous_flux
        except AttributeError:
            flux = None
        values = [ self.time, event_type, specie.last_population, flux ]
        values = map( lambda x: str(x), values )
        # log Sim_time Adjustment_type New_population New_flux
        log.debug( '\t'.join( values ) )
