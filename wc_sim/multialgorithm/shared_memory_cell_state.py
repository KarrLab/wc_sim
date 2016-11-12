'''Maintain the population of a set of species.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-06-17
:Copyright: 2016, Karr Lab
:License: MIT
'''

# TODO(Arthur): globally, properly name elements that are protected or private within a class
import sys
import numpy as np
from threading import RLock

from wc_utils.util.dict import DictUtil

# logging
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
debug_log = debug_logs.get_log( 'wc.debug.file' )

from wc_sim.multialgorithm.utils import species_compartment_name
from wc_sim.multialgorithm.specie import Specie

class SharedMemoryCellState( object ):
    '''The cell's state, which represents the state of its species.

    SharedMemoryCellState tracks the population of a set of species. Population values (copy numbers)
    can be read or written. To enable multi-algorithmic modeling, it supports writes to a specie's
    population by both discrete and continuous models.

    All accesses to this state must provide a simulation time, which supports simple synchronization
    of interactions between sub-models: reads access the previous writes (called adjustments).

    For any given specie, all operations must occur in non-decreasing simulation time order.
    Record history operations must also occur in non-decreasing time order.

    Attributes:
        name: the name of this object
        time: the time of the current operation
        population: dict: specie_name -> Specie(); the species whose counts are stored,
            represented by Specie objects
        last_access_time: dict: species_name -> last_time; the last time at which the specie was accessed
        history: nested dict; an optional history of the cell's state, created if retain_history is set.
            the population history is recorded at each continuous adjustment.
    '''

    # TODO(Arthur): IMPORTANT: support tracking the population history of species added at any time in the simulation
    # TODO(Arthur): standardize on specie_name or specie_id
    # TODO(Arthur): extend beyond population, e.g., to represent binding sites for individual macromolecules
    # TODO(Arthur): optimize to represent just the state of shared species
    # TODO(Arthur): report error if a Specie instance is updated by multiple continuous sub-models

    def __init__( self, model, name, initial_population, initial_fluxes=None, retain_history=True ):
        '''Initialize a SharedMemoryCellState object.

        Initialize a SharedMemoryCellState object. Establish its initial population, and set debugging booleans.

        Args:
            model: object ref; the model containing this SharedMemoryCellState
            initial_population: initial population for some species;
                dict: specie_name -> initial_population
            initial_fluxes: optional; dict: specie_name -> initial_flux;
                initial fluxes for all species whose populations are estimated by a continuous model
                fluxes are ignored for species not specified in initial_population
            retain_history: boolean; whether to retain species population history

        Raises:
            AssertionError: if the population cannot be initialized.
        '''

        self.model = model
        self.name = name
        self.time = 0
        self._population = {}
        self._population_lock = RLock()   # make accesses to self._population thread-safe
        self.last_access_time = {}

        if retain_history:
            self._initialize_history()

        try:
            if initial_fluxes is not None:
                for specie_id in initial_population:
                    self.init_cell_state_specie( specie_id, initial_population[specie_id],
                        initial_fluxes[specie_id] )
            else:
                for specie_id in initial_population:
                    self.init_cell_state_specie( specie_id, initial_population[specie_id] )
        except AssertionError as e:
            sys.stderr.write( "Cannot initialize SharedMemoryCellState: {}.\n".format( e.message ) )

        # write initialization data
        debug_log.debug( "initial_population: {}".format( DictUtil.to_string_sorted_by_key(initial_population) ),
            sim_time=self.time )
        debug_log.debug( "initial_fluxes: {}".format( DictUtil.to_string_sorted_by_key(initial_fluxes) ),
            sim_time=self.time )

    def init_cell_state_specie( self, specie_name, population, initial_flux_given=None ):
        '''Initialize a specie with the given population and flux.

        Add a specie to the cell state. The specie's population is set at the current time.

        Args:
            specie: string; a unique specie name
            population: float; initial population of the specie
            initial_flux_given: float; optional; initial flux for the specie

        Raises:
            ValueError: if the specie is already stored by this SharedMemoryCellState
        '''
        if specie_name in self._population:
            raise ValueError( "Error: specie_name '{}' already stored by this "
                "SharedMemoryCellState".format( specie_name ) )
        with self._population_lock:
            self._population[specie_name] = Specie( specie_name, population, initial_flux=initial_flux_given )
        self.last_access_time[specie_name] = self.time
        self._add_to_history(specie_name)

    def _initialize_history(self):
        '''Initialize the population history with current population.'''
        self._history = {}
        self._history['time'] = [self.time]  # a list of times at which population is recorded
        # the value of self._history['population'][specie_id] is a list of
        # the population of specie_id at the times history is recorded
        self._history['population'] = { }

    def _add_to_history(self, specie_id):
        '''Add a specie to the history.'''
        if self._recording_history():
            with self._population_lock:
                population = self.read( self.time, [specie_id] )[specie_id]
                self._history['population'][specie_id] = [population]

    def _recording_history(self):
        '''Is history being recorded?

        Return:
            True if history is being recorded.
        '''
        return hasattr(self, '_history')

    def _record_history(self):
        '''Record the current population in the history.

        Snapshot the current population of all species in the history. The current time
        is obtained from self.time.

        Raises:
            ValueError if the current time is not greater than the previous time at which the
            history was recorded.
        '''
        if not self._history['time'][-1] < self.time:
            raise ValueError( "time of previous _record_history() ({}) not less than current time ({})".format(
                self._history['time'][-1], self.time ) )
        self._history['time'].append( self.time )
        with self._population_lock:
            for specie_id, population in self.read( self.time, list(self._population.keys()) ).items():
                self._history['population'][specie_id].append( population )

    # TODO(Arthur): unit test this with numpy_format=True
    def report_history(self, numpy_format=False ):
        '''Provide the time and species count history.

        Args:
            numpy_format: boolean; if set return history in numpy data structures

        Returns:
            The time and species count history. By default, the return value rv is a dict, with
            rv['time'] = list of time samples
            rv['population'][specie_id] = list of counts for specie_id at the time samples
            If numpy_format set, return the same data structure as was used in WcTutorial.

        Raises:
            ValueError if the history was not recorded
        '''
        if self._recording_history():
            if numpy_format:
                timeHist = np.asarray( self._history['time'] )
                speciesCountsHist = np.zeros((len(self.model.species), len(self.model.compartments),
                    len(self._history['time'])))
                for specie_index,specie in list(enumerate(self.model.species)):
                    for comp_index,compartment in list(enumerate(self.model.compartments)):
                        for time_index in range(len(self._history['time'])):
                            specie_comp_id = species_compartment_name(specie, compartment)
                            speciesCountsHist[specie_index,comp_index,time_index] = \
                                self._history['population'][specie_comp_id][time_index]

                return (timeHist, speciesCountsHist)
            else:
                return self._history
        else:
            raise ValueError( "Error: history not recorded" )

    def history_debug(self):
        '''Provide some of the history in a string.

        Provide a string containing the start and end time of the history and
        a table with the first and last population value for each specie.

        Return:
            string with first and last times and a
            tab-separated matrix of rows with species name, first, last population values

        Raises:
            ValueError if the history was not recorded
        '''
        if self._recording_history():
            lines = []
            lines.append( "#times\tfirst\tlast" )
            lines.append( "{}\t{}\t{}".format( len(self._history['time']), self._history['time'][0],
                self._history['time'][-1] ) )
            lines.append(  "Specie\t#values\tfirst\tlast" )
            for s in self._history['population'].keys():
                lines.append( "{}\t{}\t{:.1f}\t{:.1f}".format( s, len(self._history['population'][s]),
                    self._history['population'][s][0], self._history['population'][s][-1] ) )
            return '\n'.join( lines )
        else:
            raise ValueError( "Error: history not recorded" )

    def _check_species( self, time, species ):
        '''Check whether the species are a list, and not known by this SharedMemoryCellState.

        Raises:
            ValueError: species are not a list
            ValueError: adjustment attempts to change the population of a non-existent species
            ValueError: if a specie in species is being accessed at a time earlier than a prior access
        '''
        if not isinstance( species, list ):
            raise ValueError( "Error: species '{}' must be a list".format( species ) )
        with self._population_lock:
            unknown_species = set( species ) - set( list(self._population.keys()) )
        if unknown_species:
            # raise exeception if some species are non-existent
            raise ValueError( "Error: request for population of unknown specie(s): {}".format(
                ', '.join(map( lambda x: "'{}'".format( str(x) ), unknown_species ) ) ) )
        self.__check_access_time( time, species )

    def __check_access_time( self, time, species ):
        '''Check whether the species are being accessed in non-decreasing time order.

        Raises:
            ValueError: if specie in species is being accessed at a time earlier than a prior access
        '''
        early_accesses = filter( lambda s: time < self.last_access_time[s], species)
        if any(early_accesses):
            raise ValueError( "Error: earlier access of specie(s): {}".format( early_accesses ))

    def __update_access_times( self, time, species ):
        for specie_name in species:
            self.last_access_time[specie_name] = time

    # TODO(Arthur): IMPORTANT; create read_one (or read_list) to avoid cumbersome reading
    # of one specie
    # TODO(Arthur): IMPORTANT; add optional use_interpolation, so we can compare with and wo
    # interpolation
    def read( self, time, species ):
        '''Read the predicted population of a list of species at a particular time.

        Args:
            time: float; the time at which the population should be read
            species: list; identifiers of the species to read

        Returns:
            species counts: dict: species_id -> copy_number; the predicted copy number of each
            requested species at time

        Raises:
            ValueError: the population of unknown specie(s) were requested
        '''
        self._check_species( time, species )
        self.time = time
        self.__update_access_times( time, species )
        return { specie:self._population[specie].get_population(time) for specie in species }

    def adjust_discretely( self, time, adjustments ):
        '''A discrete model adjusts the population of a set of species at a particular time.

        Args:
            time: float; the time at which the population is being adjusted
            adjustments: dict: specie_ids -> population_adjustment; adjustments to be made to
                some species populations

        Raises:
            ValueError: adjustment attempts to change the population of an unknown species
            ValueError: if population goes negative
        '''
        self._check_species( time, list( adjustments.keys() ) )
        self.time = time
        for specie in adjustments:
            try:
                self._population[specie].discrete_adjustment( adjustments[specie], self.time )
                self.__update_access_times( time, [specie] )
            except ValueError as e:
                raise ValueError( "Error: on specie {}: {}".format( specie, e ) )
            self.log_event( 'discrete_adjustment', self._population[specie] )

    def adjust_continuously( self, time, adjustments ):
        '''A continuous model adjusts the population of a set of species at a particular time.

        Args:
            time: float; the time at which the population is being adjusted
            adjustments: dict: specie_ids -> (population_adjustment, flux); adjustments to be made
                to some species populations

        Raises:
            ValueError: adjustment attempts to change the population of a non-existent species
            ValueError: if population goes negative
        '''
        self._check_species( time, list( adjustments.keys() ) )
        self.time = time

        # record simulation state history
        # TODO(Arthur): may want to also do it in adjust_discretely()
        if self._recording_history(): self._record_history()
        for specie,(adjustment,flux) in adjustments.items():
            try:
                self._population[specie].continuous_adjustment( adjustment, time, flux )
                self.__update_access_times( time, [specie] )
            except ValueError as e:
                # TODO(Arthur): IMPORTANT; return to raising exceptions with negative population
                # when initial values get debugged
                # raise ValueError( "Error: on specie {}: {}".format( specie, e ) )
                e = str(e).strip()
                debug_log.error( "Error: on specie {}: {}".format( specie, e ),
                    sim_time=self.time )

            self.log_event( 'continuous_adjustment', self._population[specie] )

    def log_event( self, event_type, specie ):
        '''Log simulation events that modify a specie's population.

        Log the simulation time, event type, specie population, and current flux for each simulation
        event message that adjusts the population.

        Args:
            event_type: string; description of the event's type
            specie: Specie object; the object whose adjustment is being logged
        '''
        try:
            flux = specie.continuous_flux
        except AttributeError:
            flux = None
        values = [ event_type, specie.last_population, flux ]
        values = map( lambda x: str(x), values )
        # log Sim_time Adjustment_type New_population New_flux
        debug_log.debug( '\t'.join( values ), local_call_depth=1, sim_time=self.time )
