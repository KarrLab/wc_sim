'''Maintain the population of a set of species.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-06-17
:Copyright: 2016, Karr Lab
:License: MIT
'''

import sys
import numpy as np

from wc_utils.util.dict import DictUtil

# logging
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
debug_log = debug_logs.get_log( 'wc.debug.file' )

from wc_sim.multialgorithm.utils import species_compartment_name
from wc_sim.multialgorithm.specie import Specie
from wc_sim.multialgorithm.abc_for_species_pop_access import AccessSpeciesPopulationInterface

class LocalSpeciesPopulation(AccessSpeciesPopulationInterface):
    '''Maintain the population of a set of species.

    LocalSpeciesPopulation tracks the population of a set of species. Population values (copy numbers)
    can be read or written. To enable multi-algorithmic modeling, it supports writes to a specie's
    population by both discrete and continuous models.

    All accesses to this object must provide a simulation time, which can detect errors in shared
    access by sub-models in a sequential simulator: reads access the previous writes (called
    adjustments).

    For any given specie, all operations must occur in non-decreasing simulation time order.
    Record history operations must also occur in time order.

    A LocalSpeciesPopulation object is accessed via local method calls. It can be wrapped as a
    DES simulation object to provide distributed access.

    Attributes:
        model (:obj:`Model`): the `Model` containing this LocalSpeciesPopulation.
        name (str): the name of this object.
        time (float): the time of the current operation.
        _population (:obj:`dict` of :obj:`Specie`): map: specie_id -> Specie(); the species whose
            counts are stored, represented by Specie objects.
        last_access_time (:obj:`dict` of `float`): map: species_name -> last_time; the last time at
            which the specie was accessed.
        history (:obj:`dict`) nested dict; an optional history of the species' state. The population
            history is recorded at each continuous adjustment.
    '''

    # TODO(Arthur): IMPORTANT: support tracking the population history of species added at any time
    # in the simulation
    # TODO(Arthur): report error if a Specie instance is updated by multiple continuous sub-models

    def __init__( self, model, name, initial_population, initial_fluxes=None, retain_history=True ):
        '''Initialize a LocalSpeciesPopulation object.

        Initialize a LocalSpeciesPopulation object. Establish its initial population, and set debugging booleans.

        Args:
            model (:obj:`Model`): the `Model` containing this LocalSpeciesPopulation.
            initial_population (:obj:`dict` of float): initial population for some species;
                dict: specie_id -> initial_population.
            initial_fluxes (:obj:`dict` of float, optional): map: specie_id -> initial_flux;
                initial fluxes for all species whose populations are estimated by a continuous model
                fluxes are ignored for species not specified in initial_population.
            retain_history (bool): whether to retain species population history.

        Raises:
            AssertionError: if the population cannot be initialized.
        '''

        # TODO(Arthur): IMPORTANT: stop using model, which might not be in the same address space as this object
        self.model = model
        self.name = name
        self.time = 0
        self._population = {}
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
            sys.stderr.write( "Cannot initialize LocalSpeciesPopulation: {}.\n".format( e.message ) )

        # write initialization data
        debug_log.debug( "initial_population: {}".format( DictUtil.to_string_sorted_by_key(
            initial_population) ), sim_time=self.time )
        debug_log.debug( "initial_fluxes: {}".format( DictUtil.to_string_sorted_by_key(initial_fluxes) ),
            sim_time=self.time )

    def init_cell_state_specie( self, specie_id, population, initial_flux_given=None ):
        '''Initialize a specie with the given population and flux.

        Add a specie to the cell state. The specie's population is set at the current time.

        Args:
            specie_id (str): a unique specie identifier.
            population (float): initial population of the specie.
            initial_flux_given (:obj:`float`, optional): initial flux for the specie.

        Raises:
            ValueError: if the specie is already stored by this LocalSpeciesPopulation.
        '''
        if specie_id in self._population:
            raise ValueError( "Error: specie_id '{}' already stored by this "
                "LocalSpeciesPopulation".format( specie_id ) )
        self._population[specie_id] = Specie( specie_id, population, initial_flux=initial_flux_given )
        self.last_access_time[specie_id] = self.time
        self._add_to_history(specie_id)

    def _check_species( self, time, species ):
        '''Check whether the species are a set, and not known by this LocalSpeciesPopulation.

        Raises:
            ValueError: species is not a set.
            ValueError: adjustment attempts to change the population of a non-existent species.
            ValueError: if a specie in species is being accessed at a time earlier than a prior access.
        '''
        if not isinstance( species, set ):
            raise ValueError( "Error: species '{}' must be a set".format( species ) )
        unknown_species = species - set( list(self._population.keys()) )
        if unknown_species:
            # raise exception if some species are non-existent
            raise ValueError( "Error: request for population of unknown specie(s): {}".format(
                ', '.join(map( lambda x: "'{}'".format( str(x) ), unknown_species ) ) ) )
        self.__check_access_time( time, species )

    def __check_access_time( self, time, species ):
        '''Check whether the species are being accessed in non-decreasing time order.

        Raises:
            ValueError: if specie in species is being accessed at a time earlier than a prior access.
        '''
        early_accesses = list(filter( lambda s: time < self.last_access_time[s], species))
        if early_accesses:
            raise ValueError( "Error: earlier access of specie(s): {}".format(early_accesses))

    def __update_access_times( self, time, species ):
        for specie_id in species:
            self.last_access_time[specie_id] = time

    def read_one(self, time, specie_id):
        '''Read the predicted population of a specie at a particular time.

        Args:
            time (float): the time at which the population should be estimated.
            specie_id (list): identifiers of the species to read.

        Returns:
            float: the predicted copy number of `specie_id` at `time`.

        Raises:
            ValueError: if the population of unknown specie(s) were requested.
        '''
        specie_id_in_set = {specie_id}
        self._check_species(time, specie_id_in_set)
        self.time = time
        self.__update_access_times(time, specie_id_in_set)
        return self._population[specie_id].get_population(time)

    def read( self, time, species ):
        '''Read the predicted population of a list of species at a particular time.

        Args:
            time (float): the time at which the population should be estimated.
            species (set): identifiers of the species to read.

        Returns:
            species counts: dict: species_id -> copy_number; the predicted copy number of each
            requested species at `time`.

        Raises:
            ValueError: if the population of unknown specie(s) were requested.
        '''
        self._check_species( time, species )
        self.time = time
        self.__update_access_times( time, species )
        return { specie:self._population[specie].get_population(time) for specie in species }

    def adjust_discretely( self, time, adjustments ):
        '''A discrete model adjusts the population of a set of species at a particular time.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of float): map: specie_ids -> population_adjustment; adjustments
                to be made to some species populations.

        Raises:
            ValueError: if any adjustment attempts to change the population of an unknown species.
            ValueError: if any population estimate would become negative.
        '''
        self._check_species( time, set( adjustments.keys() ) )
        self.time = time
        for specie in adjustments:
            try:
                self._population[specie].discrete_adjustment( adjustments[specie], self.time )
                self.__update_access_times( time, {specie} )
            except ValueError as e:
                raise ValueError( "Error: on specie {}: {}".format( specie, e ) )
            self.log_event( 'discrete_adjustment', self._population[specie] )

    def adjust_continuously( self, time, adjustments ):
        '''A continuous model adjusts the population of a set of species at a particular time.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of `tuple`): map: specie_ids -> (population_adjustment, flux);
                adjustments to be made to some species populations.

        Raises:
            ValueError: if any adjustment attempts to change the population of an unknown species.
            ValueError: if any population estimate would become negative.
        '''
        self._check_species( time, set( adjustments.keys() ) )
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
            event_type (str): description of the event's type.
            specie (:obj:`Specie`): the object whose adjustment is being logged.
        '''
        try:
            flux = specie.continuous_flux
        except AttributeError:
            flux = None
        values = [ event_type, specie.last_population, flux ]
        values = map( lambda x: str(x), values )
        # log Sim_time Adjustment_type New_population New_flux
        debug_log.debug( '\t'.join( values ), local_call_depth=1, sim_time=self.time )

    def _initialize_history(self):
        '''Initialize the population history with current population.'''
        self._history = {}
        self._history['time'] = [self.time]  # a list of times at which population is recorded
        # the value of self._history['population'][specie_id] is a list of
        # the population of specie_id at the times history is recorded
        self._history['population'] = { }

    def _add_to_history(self, specie_id):
        '''Add a specie to the history.

        Args:
            specie_id (str): a unique specie identifier.
        '''
        if self._recording_history():
            population = self.read_one( self.time, specie_id )
            self._history['population'][specie_id] = [population]

    def _recording_history(self):
        '''Is history being recorded?

        Returns:
            True if history is being recorded.
        '''
        return hasattr(self, '_history')

    def _record_history(self):
        '''Record the current population in the history.

        Snapshot the current population of all species in the history. The current time
        is obtained from `self.time`.

        Raises:
            ValueError if the current time is not greater than the previous time at which the
            history was recorded.
        '''
        if not self._history['time'][-1] < self.time:
            raise ValueError( "time of previous _record_history() ({}) not less than current time ({})".format(
                self._history['time'][-1], self.time ) )
        self._history['time'].append( self.time )
        for specie_id, population in self.read( self.time, set(self._population.keys()) ).items():
            self._history['population'][specie_id].append( population )

    # TODO(Arthur): unit test this with numpy_format=True
    def report_history(self, numpy_format=False ):
        '''Provide the time and species count history.

        Args:
            numpy_format (bool): if set return history in numpy data structures.

        Returns:
            The time and species count history. By default, the return value rv is a dict, with
            rv['time'] = list of time samples
            rv['population'][specie_id] = list of counts for specie_id at the time samples
            If numpy_format set, return the same data structure as was used in WcTutorial.

        Raises:
            ValueError if the history was not recorded.
        '''
        if self._recording_history():
            if numpy_format:
                # TODO(Arthur): IMPORTANT: stop using model, as it may not be in this address space
                # instead, don't provide the history in 'numpy_format'
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

        Returns:
            srt: the start and end time of he history and a
            tab-separated matrix of rows with species id, first, last population values.

        Raises:
            ValueError if the history was not recorded.
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
