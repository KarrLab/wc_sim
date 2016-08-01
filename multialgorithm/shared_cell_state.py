"""
A shared-memory species counts cell state. Derived from class CellState.

Created 2016/06/17
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

# TODO(Arthur): globally, properly name elements that are protected or private within a class
# TODO(Arthur): IMPORTANT: unittest new code

import sys
import logging

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from Sequential_WC_Simulator.multialgorithm.config import WC_SimulatorConfig
from specie import Specie

    
class SharedMemoryCellState( object ): 
    """The cell's state, which represents the state of its species.
    
    SharedMemoryCellState tracks the population of a set of species. Population values (copy numbers)
    can be read or written. To enable multi-algorithmic modeling, it supports writes to a specie's
    population by both discrete and continuous models. 
    
    For any given specie, all operations must occur in non-decreasing simulation time order.
    Record history operations must also occur in non-decreasing time order. 
    
    Attributes:
        name: the name of this object
        time: the time of the current operation
        population: dict: specie_name -> Specie(); the species whose counts are stored,
            represented by Specie objects
        last_access_time: dict: species_name -> last_time; the last time at which the specie was accessed
        logger_name: string; name of logger for the SharedMemoryCellState
        history: nested dict; an optional history of the cell's state, created if retain_history is set.
            the population history is recorded at each continuous adjustment.
    
    # TODO(Arthur): standardize on specie_name or specie_id
    # TODO(Arthur): extend beyond population, e.g., to represent binding sites for individual macromolecules
    # TODO(Arthur): optimize to represent just the state of shared species
    # TODO(Arthur): report error if a Specie instance is updated by multiple continuous sub-models
    """
    
    def __init__( self, name, initial_population, initial_fluxes=None, retain_history=False, 
        shared_random_seed=None, debug=False, log=False):
        """Initialize a SharedMemoryCellState object.
        
        Initialize a SharedMemoryCellState object. Establish its initial population, and set debugging booleans.
        
        Args:
            initial_population: initial population for some species;
                dict: specie_name -> initial_population
            initial_fluxes: optional; dict: specie_name -> initial_flux; 
                initial fluxes for all species whose populations are estimated by a continuous model
                fluxes are ignored for species not specified in initial_population
            retain_history: boolean; whether to retain species population history
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
        self.last_access_time = {}
        
        try:
            if initial_fluxes is not None:
                for specie_id in initial_population.keys():
                    self.init_cell_state_specie( specie_id, initial_population[specie_id], initial_fluxes[specie_id] )
            else:
                for specie_id in initial_population.keys():
                    self.init_cell_state_specie( specie_id, initial_population[specie_id] )
        except AssertionError as e:
            sys.stderr.write( "Cannot initialize SharedMemoryCellState: {}.\n".format( e.message ) )
        
        if retain_history:
            self._initialize_history()
        
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

    def init_cell_state_specie( self, specie_name, population, initial_flux_given=None ):
        """Initialize a specie with the given population at the start of the simulation.
        
        Args:
            specie: string; a unique specie name
            population: float; initial population of the specie
            initial_flux_given: float; optional; initial flux for the specie

        Raises:
            ValueError: if the specie is already stored by this SharedMemoryCellState
        """
        if specie_name in self.population:
            raise ValueError( "Error: specie_name '{}' already stored by this SharedMemoryCellState".format( specie_name ) )
        self.population[specie_name] = Specie( specie_name, population, initial_flux=initial_flux_given )
        self.last_access_time[specie_name] = 0
        if self._recording_history(): self._add_specie_to_history( specie_name )

    def _initialize_history(self):
        """Initialize the population history."""
        self.history = {}
        self.history['time'] = [0]
        self.history['population'] = { }    # dict: specie_name -> copy number
        
    def _recording_history(self):
        """Is history being recorded?
        
        Return:
            True if history is being recorded.
        """
        return hasattr(self, 'history')

    def _add_specie_to_history(self, specie):
        """Add a specie to the history."""
        self.history['population'][specie] = [0]
        
    def _record_history(self):
        """Record the current population in the history.
        
        The current time is obtained from self.time.
        
        Raises:
            ValueError if the current time is not greater than the previous time at which the 
            history was recorded.
        """
        if not self.history['time'][-1] < self.time:
            raise ValueError( "time of previous _record_history() ({}) not less than current time ({})".format(
                self.history['time'][-1], self.time ) )
        self.history['time'].append( self.time )
        for specie_id, population in self.read( self.time,  self.population.keys() ).items():
            self.history['population'][specie_id].append( population )
        
    def report_history(self):
        """Provide species count history.
        
        Raises:
            ValueError if the history was not recorded
        """
        # TODO(Arthur): also provide history in np array for plot engine
        if self._recording_history():
            return self.history
        else:
            raise ValueError( "history not recorded" )

    def history_debug(self):
        """Print a bit of species count history.
        
        Raises:
            ValueError if the history was not recorded
        """
        if self._recording_history():
            # print subset of history for debugging
            print "#times\tfirst\tlast"
            print "{}\t{}\t{}".format( len(self.history['time']), self.history['time'][0], 
                self.history['time'][-1] )
            print "Specie\t#values\tfirst\tlast"
            for s in self.history['population'].keys():
                print "{}\t{}\t{}\t{}".format( s, len(self.history['population'][s]), 
                    self.history['population'][s][0], self.history['population'][s][-1] )
        else:
            raise ValueError( "history not recorded" )
        
    def _check_species( self, time, species ):
        """Check whether the species are a list, and not known by this SharedMemoryCellState.
        
        Raises:
            ValueError: species are not a list
            ValueError: adjustment attempts to change the population of a non-existent species
            ValueError: if a specie in species is being accessed at a time earlier than a prior access
        """
        if not isinstance( species, list ):
            raise ValueError( "Error: species '{}' must be a list".format( species ) )
        unknown_species = set( species ) - set( self.population.keys() ) 
        if unknown_species:
            # raise exeception if some species are non-existent
            raise ValueError( "Error: request for population of unknown specie(s): {}".format( 
                ', '.join(map( lambda x: "'{}'".format( str(x) ), unknown_species ) ) ) )
        self.__check_access_time( time, species )

    def __check_access_time( self, time, species ):
        """Check whether the species are being accessed in non-decreasing time order.
        
        Raises:
            ValueError: if specie in species is being accessed at a time earlier than a prior access
        """
        early_accesses = filter( lambda s: time < self.last_access_time[s], species)
        if early_accesses:
            raise ValueError( "Error: earlier access of specie(s): {}".format( early_accesses ))

    def __update_access_times( self, time, species ):
        for specie_name in species:
            self.last_access_time[specie_name] = time

    # TODO(Arthur): IMPORTANT; add optional use_interpolation, so we can compare with and wo
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
        self._check_species( time, species )
        self.time = time
        self.__update_access_times( time, species )
        return { specie:self.population[specie].get_population(time) for specie in species }
            
    def adjust_discretely( self, time, adjustments ):
        """A discrete model adjusts the population of a set of species at a particular time.
        
        Args:
            time: float; the time at which the population is being adjusted
            adjustments: dict: specie_ids -> population_adjustment; adjustments to be made to 
                some species populations

        Raises:
            ValueError: adjustment attempts to change the population of an unknown species
            ValueError: if population goes negative
        """
        self._check_species( time, adjustments.keys() )
        self.time = time
        for specie in adjustments.keys():
            try:
                self.population[specie].discrete_adjustment( adjustments[specie] )
                self.__update_access_times( time, [specie] )
            except ValueError as e:
                raise ValueError( "Error: on specie {}: {}".format( specie, e ) )
            self.log_event( 'discrete_adjustment', self.population[specie] )
    
    def adjust_continuously( self, time, adjustments ):
        """A continuous model adjusts the population of a set of species at a particular time.
        
        Args:
            time: float; the time at which the population is being adjusted
            adjustments: dict: specie_ids -> (population_adjustment, flux); adjustments to be made 
                to some species populations

        Raises:
            ValueError: adjustment attempts to change the population of a non-existent species
            ValueError: if population goes negative
        """
        self._check_species( time, adjustments.keys() )
        self.time = time

        # record simulation state history
        # TODO(Arthur): may want to also do it in adjust_discretely(), or before executeReaction() 
        # in simple_SSA_submodel(), as JK recommended
        if self._recording_history(): self._record_history() 
        for specie,(adjustment,flux) in adjustments.items():
            try:
                self.population[specie].continuous_adjustment( adjustment, time, flux )
                self.__update_access_times( time, [specie] )
            except ValueError as e:
                # TODO(Arthur): IMPORTANT; return to raising exceptions with negative population
                # when initial values get debugged
                # raise ValueError( "Error: on specie {}: {}".format( specie, e ) )
                e = str(e).strip()
                logging.getLogger(self.logger_name).debug( "Error: on specie {}: {}".format( specie, e ) )

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
