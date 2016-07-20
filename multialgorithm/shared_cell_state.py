"""
A shared-memory species counts cell state. Derived from class CellState.

Created 2016/06/17
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

import sys
import logging

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from specie import Specie

    
class SharedMemoryCellState( object ): 
    """The cell's state, which represents the state of its species.
    
    SharedMemoryCellState tracks the population of a set of species. Population values (copy numbers)
    can be read or written. To enable multi-algorithmic modeling, it supports writes to a specie's
    population by both discrete and continuous models. 
    
    All operations occur at the current simulation time.
    
    Attributes:
        name: the name of this object
        time: the time of the current operation
        population: a set of species, represented by Specie objects
        logger_name: string; name of logger for the SharedMemoryCellState
    
    # TODO(Arthur): extend beyond population, e.g., to represent binding sites for individual macromolecules
    # TODO(Arthur): optimize to represent just the state of shared species
    # TODO(Arthur): report error if a Specie instance is updated by multiple continuous sub-models
    """
    
    def __init__( self, name, initial_population, initial_fluxes=None, 
        shared_random_seed=None, debug=False, log=False):
        """Initialize a SharedMemoryCellState object.
        
        Initialize a SharedMemoryCellState object. Establish its initial population, and set debugging booleans.
        
        Args:
            initial_population: initial population for some species;
                dict: specie_name -> initial_population
            initial_fluxes: optional; dict: specie_name -> initial_flux; 
                initial fluxes for all species whose populations are estimated by a continuous model
                fluxes are ignored for species not specified in initial_population
            shared_random_seed: hashable object; optional; set to deterministically initialize all random number
                generators used by the Specie objects
            debug: boolean; log debugging output
            log: boolean; log population dynamics of species
        # TODO(Arthur): add: write_plot_output: boolean; log output for plotting simulation 

        Raises:
            AssertionError: if the population cannot be initialized.
        """

        self.name = name
        self.time = 0
        self.population = {}
        try:
            for specie_name in initial_population.keys():
                initial_flux_given = None
                if initial_fluxes is not None and specie_name in initial_fluxes:
                    initial_flux_given = initial_fluxes[specie_name]
                self.population[specie_name] = Specie( specie_name, initial_population[specie_name], 
                    initial_flux=initial_flux_given, random_seed=shared_random_seed )
        except AssertionError as e:
            sys.stderr.write( "Cannot initialize SharedMemoryCellState: {}.\n".format( e.message ) )
            
        self.logger_name = "SharedMemoryCellState_{}".format( name )
        if debug:
            # make a logger for this SharedMemoryCellState
            # TODO(Arthur): eventually control logging when creating SimulationObjects, and pass in the logger
            setup_logger( self.logger_name, level=logging.DEBUG )
            mylog = logging.getLogger(self.logger_name)
            # write initialization data
            mylog.debug( "initial_population: {}".format( str(initial_population) ) )
            mylog.debug( "initial_fluxes: {}".format( str(initial_fluxes) ) )
            mylog.debug( "shared_random_seed: {}".format( str(shared_random_seed) ) )
            mylog.debug( "debug: {}".format( str(debug) ) )
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

    def check_species( self, species ):
        """Check whether the species are known by this SharedMemoryCellState.
        
        Raises:
            ValueError: adjustment attempts to change the population of a non-existent species
        """
        unknown_species = set( species ) - set( self.population.keys() ) 
        if unknown_species:
            # raise exeception if some species are non-existent
            raise ValueError( "Error: request for population of unknown specie(s): {}\n".format( 
                ', '.join(map( lambda x: "'{}'".format( str(x) ), unknown_species ) ) ) )

    def read( self, time, species ):
        """Read the predicted population of a list of species at a particular time.
        
        Args:
            time: float; the time at which the population should be read
            species: list; identifiers of the species to read

        Returns:
            species counts: dict: species_id -> copy_number; the predicted copy number of each 
            requested species at time

        Raises:
            ValueError: the population of unknown specie(s) were requested
        """
        self.check_species( species )
        self.time = time
        return { specie:self.population[specie].get_population(time) for specie in species}
            
    def adjust_discretely( self, time, adjustments ):
        """A discrete model adjusts the population of a set of species at a particular time.
        
        Args:
            time: float; the time at which the population is being adjusted
            adjustments: dict: specie_ids -> population_adjustment; adjustments to be made to some species populations

        Raises:
            ValueError: adjustment attempts to change the population of a non-existent species
            ValueError: if population goes negative
        """
        self.check_species( adjustments.keys() )
        self.time = time
        for specie in adjustments.keys():
            self.population[specie].discrete_adjustment( adjustments[specie] )
            self.log_event( 'discrete_adjustment', self.population[specie] )
    
    def adjust_continuously( self, time, adjustments ):
        """A continuous model adjusts the population of a set of species at a particular time.
        
        Args:
            time: float; the time at which the population is being adjusted
            adjustments: dict: specie_ids -> (population_adjustment, flux); adjustments to be made to some species populations

        Raises:
            ValueError: adjustment attempts to change the population of a non-existent species
            ValueError: if population goes negative
        """
        self.check_species( adjustments.keys() )
        self.time = time
        for specie,(adjustment,flux) in adjustments.items():
            self.population[specie].continuous_adjustment( adjustment, time, flux )
            self.log_event( 'continuous_adjustment', self.population[specie] )

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
