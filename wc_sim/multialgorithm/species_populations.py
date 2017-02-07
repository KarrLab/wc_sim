'''Store species populations, and partition them among submodel private species and shared species.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-04
:Copyright: 2016-2017, Karr Lab
:License: MIT
'''
import abc, six
import numpy as np
import sys
from collections import defaultdict
from six import iteritems

from wc_sim.core.simulation_engine import MessageTypesRegistry
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.multialgorithm import message_types

from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm.multialgorithm_errors import NegativePopulationError, SpeciesPopulationError
from wc_sim.multialgorithm.utils import species_compartment_name
from wc_utils.config.core import ConfigManager
from wc_utils.util.dict import DictUtil
from wc_utils.util.misc import isclass_by_name as check_class
from wc_utils.util.rand import RandomStateManager

@six.add_metaclass(abc.ABCMeta)
class AccessSpeciesPopulationInterface():
    '''An abstract base class defining the interface between a submodel and its species population stores.

    A submodel in a WC simulation will interact with multiple components that store the population
    of species it accesses. This architecture is needed for parallelism. All these stores should
    implement this interface which defines read and write operations on the species in a store.
    '''

    @abc.abstractmethod
    def read_one(self, time, specie_id):
        '''Obtain the predicted population of a specie at a particular time.'''
        pass

    @abc.abstractmethod
    def read( self, time, species ):
        '''Obtain the predicted population of a list of species at a particular time.'''
        pass

    @abc.abstractmethod
    def adjust_discretely( self, time, adjustments ):
        '''A discrete model adjusts the population of a set of species at a particular time.'''
        pass

    @abc.abstractmethod
    def adjust_continuously( self, time, adjustments ):
        '''A continuous model adjusts the population of a set of species at a particular time.'''
        pass


config_multialgorithm = \
    ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']
LOCAL_POP_STORE = 'LOCAL_POP_STORE'  # the name of the local population store


class AccessSpeciesPopulations(AccessSpeciesPopulationInterface):
    '''Interface a submodel with the components that store the species populations it models.

    Each submodel is a distinct simulation object. In the current population-based model,
    species are represented by their populations. (A hybrid instance-population model would change
    that.) Each submodel accesses a subset of the species in a model. A submodel's
    species can be partitioned into those that are accessed ONLY by the submodel and those that it
    shares with other submodels. These are stored in a local LocalSpeciesPopulation
    which is private to this submodel, and a set of SpeciesPopSimObjects which are shared with other
    submodels. LocalSpeciesPopulation objects are accessed via local memory operations whereas
    SpeciesPopSimObjects, which are distinct simulation objects, are accessed via simulation event
    messages.

    AccessSpeciesPopulations enables a submodel to access all of the species populations that it
    uses through a single convenient R/W interface. The submodel simply indicates the specie(s)
    being used and the operation type. This object then maps the specie(s) to the entity or entities
    storing them, and executes the operation on each entity.

    Attributes:
        submodel (:obj:`Submodel`): the submodel which is using this AccessSpeciesPopulations
        species_locations (:obj:`dict` of `str`): a map indicating the store for each specie used
            by the submodel using this object, that is, the local submodel.
        local_pop_store (:obj:`LocalSpeciesPopulation`): a store of local species.
        remote_pop_stores (:obj:`dict` of identifiers of`SpeciesPopSimObject`): a map from store name
            to a system identifier for the remote population store(s) that the local submodel uses.
            For a shared memory implementation system identifiers can be object references; for a
            distributed implementation they must be network object identifiers.
        species_population_cache (:obj:`SpeciesPopulationCache`): a cache of populations for species
            that are stored remotely in the SpeciesPopSimObjects in remote_pop_stores; values for
            remote populations are pre-fetched at the right simulation time (via GetPopulation and
            GivePopulation messages) into this cache and then read from it when needed.
    '''

    def __init__(self, local_pop_store, remote_pop_stores):
        '''Initialize an AccessSpeciesPopulations object.

        The attributes submodel and species_population_cache both reference objects that also
        must reference this AccessSpeciesPopulations instance. This object must be
        instantiated first; then these other objects are created and they set their references
        with the set_*() methods below.

        Raises:
            SpeciesPopulationError: if the remote_pop_stores contains a store named
                'LOCAL_POP_STORE', which is a reserved store identifier for the local_pop_store.
        '''
        self.local_pop_store = local_pop_store
        if LOCAL_POP_STORE in remote_pop_stores:
            raise SpeciesPopulationError("AccessSpeciesPopulations.__init__: {} not a valid "
                "remote_pop_store name".format(LOCAL_POP_STORE))
        self.remote_pop_stores = remote_pop_stores
        self.species_locations = {}

    def set_submodel(self, submodel):
        '''Set the submodel that uses this AccessSpeciesPopulations.'''
        self.submodel = submodel

    def set_species_population_cache(self, species_population_cache):
        '''Set the SpeciesPopulationCache used by this AccessSpeciesPopulations.'''
        self.species_population_cache = species_population_cache

    def add_species_locations(self, store_name, specie_ids, replace=False):
        '''Add species locations to the species location map.

        Record that the species listed in `species_ids` are stored by the species population store
        identified by `store_name`. To replace existing location map values without raising an
        exception, set `replace` to True.

        Args:
            store_name (str): the globally unique name of a species population store.
            specie_ids (:obj:`list` of `str`): a list of species ids.

        Raises:
            SpeciesPopulationError: if store `store_name` is unknown.
            SpeciesPopulationError: if `replace` is False and any specie_id in `specie_ids` is
                already mapped to a different store than `store_name`.
        '''
        if not store_name in self.remote_pop_stores.keys() and store_name != LOCAL_POP_STORE:
            raise SpeciesPopulationError("add_species_locations: '{}' not a known population "
                "store.".format(store_name))
        if replace:
            for specie_id in specie_ids:
                self.species_locations[specie_id] = store_name
        else:
            assigned = list(filter(lambda s: s in self.species_locations.keys(), specie_ids))
            if assigned:
                raise SpeciesPopulationError("add_species_locations: species {} already have assigned "
                    "locations.".format(sorted(assigned)))
            for specie_id in specie_ids:
                self.species_locations[specie_id] = store_name

    def del_species_locations(self, specie_ids, force=False):
        '''Delete entries from the species location map.

        Remove species location mappings for the species in `specie_ids`. To avoid raising an
        exception when a specie is not in the location map, set `force` to `True`.

        Args:
            specie_ids (:obj:`list` of specie_ids): a list of species ids.

        Raises:
            SpeciesPopulationError: if `force` is False and any specie_id in `specie_ids` is not in the
                species location map.
        '''
        if force:
            for specie_id in specie_ids:
                try:
                    del self.species_locations[specie_id]
                except KeyError:
                    pass
        else:
            unassigned = list(filter(lambda s: s not in self.species_locations.keys(), specie_ids))
            if unassigned:
                raise SpeciesPopulationError("del_species_locations: species {} are not in the location "
                    "map.".format(sorted(unassigned)))
            for specie_id in specie_ids:
                del self.species_locations[specie_id]

    def locate_species(self, specie_ids):
        '''Locate the component(s) that store a set of species.

        Given a list of species identifiers in `species_ids`, partition them into the storage
        component(s) that store their populations. This method is widely used by code that accesses
        species. It returns a dictionary that maps from store name to the ids of species whose
        populations are modeled by the store.

        `LOCAL_POP_STORE` represents a special store, the local
        `wc_sim.multialgorithm.local_species_population.LocalSpeciesPopulation` instance. Each other
        store is identified by the name of a remote
        `from wc_sim.multialgorithm.species_pop_sim_object.SpeciesPopSimObject` instance.

        Args:
            specie_ids (:obj:`list` of `str`): a list of species identifiers.

        Returns:
            dict: a map from store_name -> a set of species_ids whose populations are stored
                by component store_name.

        Raises:
            SpeciesPopulationError: if a store cannot be found for a specie_id in `specie_ids`.
        '''
        unknown = list(filter(lambda s: s not in self.species_locations.keys(), specie_ids))
        if unknown:
            raise SpeciesPopulationError("locate_species: species {} are not "
                "in the location map.".format(sorted(unknown)))
        inverse_loc_map = defaultdict(set)
        for specie_id in specie_ids:
            store = self.species_locations[specie_id]
            inverse_loc_map[store].add(specie_id)
        return inverse_loc_map

    def read_one(self, time, specie_id):
        '''Obtain the predicted population of specie `specie_id` at the time `time`.

        If the specie is stored in the local_pop_store, obtain its population there. Otherwise obtain
        the population from the species_population_cache. If the specie's primary store is a
        remote_pop_store, then its population should be in the cache because the population should
        have been prefetched.

        Args:
            time (float): the time at which the population should be obtained.
            specie_id (str): identifier of the specie whose population will be obtained.

        Returns:
            float: the predicted population of `specie_id` at time `time`.

        Raises:
            SpeciesPopulationError: if `specie_id` is an unknown specie.
        '''
        if specie_id not in self.species_locations:
            raise SpeciesPopulationError("read_one: specie '{}' not in the location map.".format(
                specie_id))
        store = self.species_locations[specie_id]
        if store==LOCAL_POP_STORE:
            return self.local_pop_store.read_one(time, specie_id)
        else:
            # TODO(Arthur): convert print() to log message
            # print('submodel {} reading {} from cache at {:.2f}'.format(self.submodel.name,
            #   specie_id, time))
            return self.species_population_cache.read_one(time, specie_id)

    def read(self, time, species_ids):
        '''Obtain the population of the species identified in `species_ids` at the time `time`.

        Obtain the species from the local_pop_store and/or the species_population_cache. If some of
        the species' primary stores are remote_pop_stores, then their populations should be in the
        cache because they should have been prefetched.

        Args:
            time (float): the time at which the population should be obtained.
            species_ids (set): identifiers of the species whose populations will be obtained.

        Returns:
            dict: species_id -> population; the predicted population of all requested species at
            time `time`.

        Raises:
            SpeciesPopulationError: if a store cannot be found for a specie_id in `specie_ids`.
            SpeciesPopulationError: if any of the species were cached at a time that differs from `time`.
        '''
        local_species = self.locate_species(species_ids)[LOCAL_POP_STORE]
        remote_species = set(species_ids) - set(local_species)

        local_pops = self.local_pop_store.read(time, local_species)
        cached_pops = self.species_population_cache.read(time, remote_species)

        cached_pops.update(local_pops)
        return cached_pops

    def adjust_discretely(self, time, adjustments):
        '''A discrete submodel adjusts the population of a set of species at the time `time`.

        Distribute the adjustments among the population stores managed by this object.
        Iterate through the components that store the population of species listed in `adjustments`.
        Update the local population store immediately and send AdjustPopulationByDiscreteModel
        messages to the remote stores. Since these messages are asynchronous, this method returns
        as soon as they are sent.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of float): map: specie_ids -> population_adjustment; adjustments
                to be made to some species populations.

        Returns:
            list: the names of the stores for the species whose populations are adjusted.
        '''
        stores=[]
        for store,species_ids in iteritems(self.locate_species(adjustments.keys())):
            stores.append(store)
            store_adjustments = DictUtil.filtered_dict(adjustments, species_ids)
            if store==LOCAL_POP_STORE:
                self.local_pop_store.adjust_discretely(time, store_adjustments)
            else:
                self.submodel.send_event(time-self.submodel.time,
                    self.remote_pop_stores[store],
                    message_types.AdjustPopulationByDiscreteModel,
                    event_body=message_types.AdjustPopulationByDiscreteModel.Body(store_adjustments))
        return stores

    def adjust_continuously(self, time, adjustments):
        '''A continuous submodel adjusts the population of a set of species at the time `time`.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of `tuple`): map: specie_ids -> (population_adjustment, flux);
                adjustments to be made to some species populations.

        See the description for `adjust_discretely` above.

        Returns:
            list: the names of the stores for the species whose populations are adjusted.
        '''
        stores=[]
        for store,species_ids in iteritems(self.locate_species(adjustments.keys())):
            stores.append(store)
            store_adjustments = DictUtil.filtered_dict(adjustments, species_ids)
            if store==LOCAL_POP_STORE:
                self.local_pop_store.adjust_continuously(time, store_adjustments)
            else:
                self.submodel.send_event(time-self.submodel.time,
                    self.remote_pop_stores[store],
                    message_types.AdjustPopulationByContinuousModel,
                    event_body=message_types.AdjustPopulationByContinuousModel.Body(store_adjustments))
        return stores

    def prefetch(self, delay, species_ids):
        '''Obtain species populations from remote stores when they will be needed in the future.

        Generate GetPopulation queries that obtain species populations whose primary stores are
        remote at `delay` in the future. The primary stores (`SpeciesPopSimObject` objects)
        will respond to the GetPopulation queries with GivePopulation responses.

        To ensure that the remote store object executes the GetPopulation at an earlier simulation
        time than the submodel will need the data, decrease the event time of the GetPopulation
        event to the previous floating point value.

        Args:
            delay (float): the populations will be needed at now + `delay`.
            specie_ids (:obj:`list` of specie_ids): a list of species ids.

        Returns:
            list: the names of the stores for the species whose populations are adjusted.
        '''
        if delay<=0:
            raise SpeciesPopulationError("prefetch: {} provided, but delay must "
                "be non-negative.".format(delay))
        stores=[]
        epsilon = config_multialgorithm['epsilon']
        for store,species_ids in iteritems(self.locate_species(species_ids)):
            if store!=LOCAL_POP_STORE:
                stores.append(store)
                # advance the receipt of GetPopulation so the SpeciesPopSimObject executes it before
                # the submodel needs the value
                self.submodel.send_event(delay - epsilon*0.5,
                    self.remote_pop_stores[store],
                    message_types.GetPopulation,
                    event_body=message_types.GetPopulation.Body(set(species_ids)))
        return stores

    def __str__(self):
        '''Provide readable AccessSpeciesPopulations state.

        Provide the submodel's name, the name of the local_pop_store, and the id and store name of
        each specie accessed by this AccessSpeciesPopulations.

        Returns:
            str: a multi-line string describing this AccessSpeciesPopulations' state.
        '''
        state=['AccessSpeciesPopulations state:']
        if hasattr(self, 'submodel'):
            state.append('submodel: {}'.format(self.submodel.id))
        state.append('local_pop_store: {}'.format(self.local_pop_store.name))
        state.append('species locations:')
        state.append('specie_id\tstore_name')
        state += ['{}\t{}'.format(k,self.species_locations[k])
            for k in sorted(self.species_locations.keys())]
        return '\n'.join(state)


epsilon = config_multialgorithm['epsilon']
class SpeciesPopulationCache(object):
    '''Cache the population of species whose primary stores are remote population stores.

    Attributes:
        access_species_populations (:obj:`AccessSpeciesPopulations`): the `AccessSpeciesPopulations`
            containing this Object.
        _cache (:obj:`dict` of :obj:`tuple`): map: specie_id -> (time, population); the species
            whose counts are cached, containing the last write time in `time`, and the population.
    '''

    def __init__(self, access_species_populations):
        '''Initialize a SpeciesPopulationCache object.'''
        self.access_species_populations = access_species_populations
        self._cache = {}

    def clear_cache(self):
        '''Clear the cache.'''
        self._cache = {}

    def cache_population(self, time, populations):
        '''Cache some population values.

        Args:
            time (float): the time of the cached values.
            populations (:obj:`dict` of float): map: specie_ids -> population; the population
                of the species at `time`.

        Raises:
            SpeciesPopulationError: if the species are stored in the ASP's local store; they should
                not be cached.
            SpeciesPopulationError: if, for any specie, `time` is not greater than its previous
                cache time.
        '''
        # raise exception if the species are stored in the ASP's local store
        store_name_map = self.access_species_populations.locate_species(populations.keys())
        if LOCAL_POP_STORE in store_name_map:
            raise SpeciesPopulationError("cache_population: some species are stored in the "
                "AccessSpeciesPopulations's local store: {}.".format(
                    list(store_name_map[LOCAL_POP_STORE])))
        # TODO(Arthur): could raise an exception if the species are not stored in the ASP's remote stores
        for specie_id,population in iteritems(populations):
            # raise exception if the time of this cache is not greater than the previous cache time
            if specie_id in self._cache and time <= self._cache[specie_id][0]:
                raise SpeciesPopulationError("cache_population: caching an earlier population: "
                    "specie_id: {}; current time: {} <= previous time {}.".format(specie_id, time,
                    self._cache[specie_id][0]))
            self._cache[specie_id] = (time, population)

    def read_one(self, time, specie_id):
        '''Obtain the cached population of a specie at a particular time.

        Args:
            time (float): the expected time of the cached values.
            specie_id (str): identifier of the specie to obtain.

        Returns:
            float: the cached population of `specie_id` at time `time`.

        Raises:
            SpeciesPopulationError: if the species are stored in the ASP's local store, which means
                that they should not be cached.
            SpeciesPopulationError: if `time` is not greater than a specie's previous cache time.
        '''
        if specie_id not in self._cache:
            raise SpeciesPopulationError("SpeciesPopulationCache.read_one: specie '{}' not "
                "in cache.".format(specie_id))
        if self._cache[specie_id][0] + epsilon < time:
            raise SpeciesPopulationError("SpeciesPopulationCache.read_one: cache age of {} too big "
                "for read at time {} of specie '{}'.".format(
                    time-self._cache[specie_id][0], time, specie_id))
        return self._cache[specie_id][1]

    def read(self, time, species_ids):
        '''Read the cached population of a set of species at a particular time.

        Args:
            time (float): the time at which the population should be obtained.
            species_ids (set): identifiers of the species to read.

        Returns:
            species counts: dict: species_id -> copy_number; the cached copy number of each
            requested species at time `time`.

        Raises:
            SpeciesPopulationError: if any of the species are not stored in the cache.
            SpeciesPopulationError: if any of the species were cached at a time that differs
                from `time`.
        '''
        missing = list(filter(lambda specie: specie not in self._cache, species_ids))
        if missing:
            raise SpeciesPopulationError("SpeciesPopulationCache.read: species {} not in cache.".format(
                str(missing)))
        mistimed = list(filter(lambda s_id: self._cache[s_id][0] + epsilon < time, species_ids))
        if mistimed:
            raise SpeciesPopulationError("SpeciesPopulationCache.read: species {} not reading "
                "recently cached value(s).".format(str(mistimed)))
        return {specie:self._cache[specie][1] for specie in species_ids}


# logging
debug_log = debug_logs.get_log( 'wc.debug.file' )

class LocalSpeciesPopulation(AccessSpeciesPopulationInterface):
    '''Maintain the population of a set of species.

    LocalSpeciesPopulation tracks the population of a set of species. Population values (copy numbers)
    can be read or written. To enable multi-algorithmic modeling, it supports writes to a specie's
    population by both discrete and continuous models.

    All accesses to this object must provide a simulation time, which enables detection of errors in
    shared access by sub-models in a sequential simulator. In particular, a read() must access the
    previous write().

    For any given specie, all operations must occur in non-decreasing simulation time order.
    Record history operations must also occur in time order.

    A LocalSpeciesPopulation object is accessed via local method calls. It can be wrapped as a
    DES simulation object to provide distributed access, as is done by `SpeciesPopSimObject`.

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

        Initialize a LocalSpeciesPopulation object. Establish its initial population, and initialize
            the history if `retain_history` is `True`.

        Args:
            initial_population (:obj:`dict` of float): initial population for some species;
                dict: specie_id -> initial_population.
            initial_fluxes (:obj:`dict` of float, optional): map: specie_id -> initial_flux;
                initial fluxes for all species whose populations are estimated by a continuous
                model. Fluxes are ignored for species not specified in initial_population.
            retain_history (bool): whether to retain species population history.

        Raises:
            SpeciesPopulationError: if the population cannot be initialized.
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
            SpeciesPopulationError: if the specie is already stored by this LocalSpeciesPopulation.
        '''
        if specie_id in self._population:
            raise SpeciesPopulationError( "Error: specie_id '{}' already stored by this "
                "LocalSpeciesPopulation".format( specie_id ) )
        self._population[specie_id] = Specie( specie_id, population, initial_flux=initial_flux_given )
        self.last_access_time[specie_id] = self.time
        self._add_to_history(specie_id)

    def _check_species( self, time, species ):
        '''Check whether the species are a set, or not known by this LocalSpeciesPopulation.

        Also checks whether the species are being accessed in time order.

        Args:
            time (float): the time at which the population might be accessed.
            species (set): set of species_ids.

        Raises:
            SpeciesPopulationError: if species is not a set.
            SpeciesPopulationError: if any species in `species are non-existent.
            SpeciesPopulationError: if a specie in `species` is being accessed at a time earlier
                than a prior access.
        '''
        if not isinstance( species, set ):
            raise SpeciesPopulationError( "Error: species '{}' must be a set".format( species ) )
        unknown_species = species - set( list(self._population.keys()) )
        if unknown_species:
            # raise exception if some species are non-existent
            raise SpeciesPopulationError( "Error: request for population of unknown specie(s): {}".format(
                ', '.join(map( lambda x: "'{}'".format( str(x) ), unknown_species ) ) ) )
        early_accesses = list(filter( lambda s: time < self.last_access_time[s], species))
        if early_accesses:
            raise SpeciesPopulationError( "Error: earlier access of specie(s): {}".format(
                early_accesses))

    def __update_access_times( self, time, species ):
        '''Update the access time to `time` for all species_ids in `species`.

        Args:
            time (float): the access time which should be set for the species.
            species (set): a set of species_ids.
        '''
        for specie_id in species:
            self.last_access_time[specie_id] = time

    def read_one(self, time, specie_id):
        '''Obtain the predicted population of specie `specie_id` at time `time`.

        Args:
            time (float): the time at which the population should be estimated.
            specie_id (str): identifier of the specie to access.

        Returns:
            float: the predicted population of `specie_id` at time `time`.

        Raises:
            SpeciesPopulationError: if the population of an unknown specie was requested.
        '''
        specie_id_in_set = {specie_id}
        self._check_species(time, specie_id_in_set)
        self.time = time
        self.__update_access_times(time, specie_id_in_set)
        return self._population[specie_id].get_population(time)

    def read( self, time, species ):
        '''Read the predicted population of a list of species at time `time`.

        Args:
            time (float): the time at which the population should be estimated.
            species (set): identifiers of the species to read.

        Returns:
            species counts: dict: species_id -> copy_number; the predicted copy number of each
            requested species at `time`.

        Raises:
            SpeciesPopulationError: if the population of unknown specie(s) are requested.
        '''
        self._check_species( time, species )
        self.time = time
        self.__update_access_times( time, species )
        return { specie:self._population[specie].get_population(time) for specie in species }

    def adjust_discretely( self, time, adjustments ):
        '''A discrete model adjusts the population of a set of species at time `time`.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of float): map: specie_ids -> population_adjustment; adjustments
                to be made to some species populations.

        Raises:
            SpeciesPopulationError: if any adjustment attempts to change the population of an
                unknown species.
            SpeciesPopulationError: if any population estimate would become negative.
        '''
        self._check_species( time, set( adjustments.keys() ) )
        self.time = time
        for specie in adjustments:
            try:
                self._population[specie].discrete_adjustment( adjustments[specie], self.time )
                self.__update_access_times( time, {specie} )
            except SpeciesPopulationError as e:
                raise SpeciesPopulationError( "Error: on specie {}: {}".format( specie, e ) )
            self.log_event( 'discrete_adjustment', self._population[specie] )

    def adjust_continuously( self, time, adjustments ):
        '''A continuous model adjusts the population of a set of species at time `time`.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of `tuple`): map: specie_ids -> (population_adjustment, flux);
                adjustments to be made to some species populations.

        Raises:
            SpeciesPopulationError: if any adjustment attempts to change the population of an
                unknown species.
            SpeciesPopulationError: if any population estimate would become negative.
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
            except SpeciesPopulationError as e:
                # TODO(Arthur): IMPORTANT; return to raising exceptions with negative population
                # when initial values get debugged
                raise SpeciesPopulationError( "Error: on specie {}: {}".format( specie, e ) )
                e = str(e).strip()
                debug_log.error( "Error: on specie {}: {}".format( specie, e ),
                    sim_time=self.time )

            self.log_event( 'continuous_adjustment', self._population[specie] )

    def log_event( self, event_type, specie ):
        '''Log an event that modifies a specie's population.

        Log the event's simulation time, event type, specie population, and current flux (if
        specified).

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
            SpeciesPopulationError if the current time is not greater than the previous time at which the
            history was recorded.
        '''
        if not self._history['time'][-1] < self.time:
            raise SpeciesPopulationError( "time of previous _record_history() ({}) not less than current time ({})".format(
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
            SpeciesPopulationError if the history was not recorded.
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
            raise SpeciesPopulationError( "Error: history not recorded" )

    def history_debug(self):
        '''Provide some of the history in a string.

        Provide a string containing the start and end time of the history and
        a table with the first and last population value for each specie.

        Returns:
            srt: the start and end time of he history and a
            tab-separated matrix of rows with species id, first, last population values.

        Raises:
            SpeciesPopulationError if the history was not recorded.
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
            raise SpeciesPopulationError( "Error: history not recorded" )

    def __str__(self):
        '''Provide readable LocalSpeciesPopulation state.

        Provide the name of this LocalSpeciesPopulation, the current time, and the id and population
        of each specie stored by this object.

        Returns:
            str: a multi-line string describing this LocalSpeciesPopulation's state.
        '''
        state=[]
        state.append('name: {}'.format(self.name))
        state.append('time: {}'.format(str(self.time)))
        state.append(Specie.heading())
        for specie_id in sorted(self._population.keys()):
            state.append(self._population[specie_id].row())
        return '\n'.join(state)


class SpeciesPopSimObject(LocalSpeciesPopulation,SimulationObject):
    '''Maintain the population of a set of species in a simulation object that can be parallelized.

    A whole-cell PDES must run multiple submodels in parallel. These share cell state, such as
    species populations, by accessing shared simulation objects. A SpeciesPopSimObject provides that
    functionality by wrapping a LocalSpeciesPopulation as a DES object accessed only by
    simulation event messages.
    '''

    def __init__(self, name, initial_population, initial_fluxes=None, retain_history=True ):
        '''Initialize a SpeciesPopSimObject object.

        Initialize a SpeciesPopSimObject object. Initialize its base classes.

        Args:
            name (str): the name of the simulation object and local species population object.

        For remaining args and exceptions, see `__init__()` documentation for
        `wc_sim.multialgorithm.SimulationObject` and `wc_sim.multialgorithm.LocalSpeciesPopulation`.
        (Perhaps Sphinx can automate this, but the documentation is unclear.)
        '''
        SimulationObject.__init__(self, name)
        LocalSpeciesPopulation.__init__(self, None, name, initial_population, initial_fluxes)

    def handle_event(self, event_list):
        '''Handle a SpeciesPopSimObject simulation event.

        Process event messages for this SpeciesPopSimObject.

        Args:
            event_list (:obj:`list` of :obj:`wc_sim.core.Event`): list of Events to process.

        Raises:
            SpeciesPopulationError: if a GetPopulation message requests the population of an
                unknown species.
            SpeciesPopulationError: if an AdjustPopulationByContinuousModel event acts on a
                non-existent species.
        '''
        # call handle_event() in class SimulationObject to perform generic tasks on the event list
        super(SpeciesPopSimObject, self).handle_event(event_list)
        for event_message in event_list:

            # switch/case on event message type
            if check_class(event_message.event_type, message_types.AdjustPopulationByDiscreteModel):
                population_change = event_message.event_body.population_change
                self.adjust_discretely( self.time, population_change )

            elif check_class(event_message.event_type, message_types.AdjustPopulationByContinuousModel):
                population_change = event_message.event_body.population_change
                self.adjust_continuously( self.time, population_change )

            elif check_class(event_message.event_type, message_types.GetPopulation):
                species = event_message.event_body.species
                self.send_event( 0, event_message.sending_object, message_types.GivePopulation,
                    event_body=self.read( self.time, species) )

            else:
                assert False, "Shouldn't get here - {} should be covered"\
                    " in the if statement above".format(event_message.event_type)

# Register sent message types
SENT_MESSAGE_TYPES = [ message_types.GivePopulation ]
MessageTypesRegistry.set_sent_message_types(SpeciesPopSimObject, SENT_MESSAGE_TYPES)

# At any time instant, process messages in this order
MESSAGE_TYPES_BY_PRIORITY = [
    message_types.AdjustPopulationByDiscreteModel,
    message_types.AdjustPopulationByContinuousModel,
    message_types.GetPopulation ]
MessageTypesRegistry.set_receiver_priorities(SpeciesPopSimObject, MESSAGE_TYPES_BY_PRIORITY)

class Specie(object):
    '''Specie tracks the population of a single specie in a multi-algorithmic model.

    A specie is a shared object that can be read and written by multiple submodels in a
    multi-algorithmic model. We assume that a sequence of accesses of a specie instance will
    occur in non-decreasing simulation time order. (This assumption holds for conservative discrete
    event simulations and all committed parts of optimistic parallel simulations like Time Warp.)

    Consider a multi-algorithmic model that contains both discrete-time submodels, like the
    stochastic simulation algorithm (SSA), and continuous-time submodels, like ODEs and FBA.
    Discrete-time algorithms change system state at discrete time instants. Continuous-time
    algorithms employ continuous models of state change, and sample these models at time instants
    determined by the algorithm. At these instants, continuous-time models typically
    estimate a specie's population and the population's rate of change. We assume this behavior.

    A specie's state in a multi-algorithmic model may be modeled by multiple submodels which model
    reactions in which the specie participates. These can be multiple discrete-time submodels and
    at most one continuous-time submodel. (If multiple continuous-time submodels were allowed to
    predict reactions that involve a specie, a mechanism would be needed to reconsile conflicting
    flux values. We have not addressed that issue yet.)

    Discrete-time and continuous-time models adjust the state of a species by the methods
    `discrete_adjustment()` and `continuous_adjustment()`, respectively. These adjustments take the
    following forms,

    * `discrete_adjustment( population_change, time )`
    * `continuous_adjustment( population_change, time, flux )`

    where `population_change` is the increase or decrease in the specie's population, `time` is the
    time at which that change takes place, and `flux` is the predicted future rate of change of the
    population.

    To improve the accuracy of multi-algorithmic models, we support linear *interpolation* of
    population predictions for species modeled by a continuous-time submodel. An interpolated
    prediction is based on the most recent continuous-time flux prediction. Thus, we assume
    that a population modeled by a continuous model is adjusted sufficiently frequently
    that the most recent adjustment accurately estimates flux.

    A specie instance stores the most recent value of the specie's population in `last_population`,
    which is initialized when the instance is created. If a specie is modeled by a
    continuous-time submodel, it also stores the specie's flux in `continuous_flux` and the time
    of the most recent `continuous_adjustment` in `continuous_time`. Otherwise, `continuous_time`
    is `None`. Interpolation determines the population prediction `p` at time `t` as::

        interpolation = 0
        if not continuous_time is None:
            interpolation = (t - continuous_time)*continuous_flux
        p = last_population + interpolation

    This approach is completely general, and can be applied to any simulation value
    whose dynamics are predicted by a multi-algorithmic model.

    Population values returned by specie's methods use stochastic rounding to provide integer
    values and avoid systematic rounding bias. See more detail in `get_population`'s docstring.

    Attributes:
        specie_name (str): the specie's name; not logically needed, but helpful for error
            reporting, logging, debugging, etc.
        last_population (float): population after the most recent adjustment
        continuous_submodel (bool): whether one of the submodels modeling the species is a
            continuous submodel; must be set at initialization
        continuous_flux (float): if a continuous submodel is modeling the specie, the flux provided
            at initialization or by the most recent adjustment by a continuous model
        continuous_time (float): if a continuous submodel is modeling the specie, the simulation
            time of initialization (0) or the most recent adjustment by the continuous model

    '''
    # use __slots__ to save space
    __slots__ = ['specie_name', 'last_population', 'continuous_time', 'continuous_flux',
        'random_state', 'continuous_submodel']

    def __init__(self, specie_name, initial_population, initial_flux=None):
        '''Initialize a specie object at simulation time 0.

        Args:
            specie_name (str): the specie's name; not logically needed, but helpful for error
                reporting, logging, debugging, etc.
            initial_population (int): non-negative number; initial population of the specie
            initial_flux (number, optional): initial flux for the specie; required for species whose
                population is estimated, at least in part, by a continuous model
        '''
        assert 0 <= initial_population, '__init__(): population should be >= 0'
        self.specie_name = specie_name
        self.last_population = initial_population
        self.continuous_submodel = False
        if initial_flux is not None:
            self.continuous_submodel = True
            self.continuous_time = 0
            self.continuous_flux = initial_flux

        self.random_state = RandomStateManager.instance()

    def discrete_adjustment(self, population_change, time):
        '''Make a discrete adjustment of the specie's population.

        A discrete-time submodel, such as the stochastic simulation algorithm (SSA), must use this
        method to adjust the specie's population.

        Args:
            population_change (number): the modeled increase or decrease in the specie's population
            time (number): the simulation time at which the predicted change occurs; this time is
                used by `get_population` to interpolate continuous-time predictions between integrations.

        Returns:
            int: an integer approximation of the specie's adjusted population

        Raises:
            NegativePopulationError: if the predicted population at `time` is negative or
            if decreasing the population by `population_change` would make the population negative
        '''
        current_population = self.get_population(time)
        if current_population + population_change < 0:
            raise NegativePopulationError('discrete_adjustment', self.specie_name,
                self.last_population, population_change)
        self.last_population += population_change
        return self.get_population(time)

    def continuous_adjustment(self, population_change, time, flux):
        '''A continuous-time submodel adjusts the specie's state.

        A continuous-time submodel, such as an ordinary differential equation (ODE) or a dynamic flux
        balance analysis (FBA) model, uses this method to adjust the specie's state. Each
        integration of a continuous-time model must predict a specie's population change and the
        population's short-term future rate of change, i.e., its `flux`. Further, since an
        integration of a continuous-time model at the current time must depend on this specie's
        population just before the integration, we assume that `population_change` incorporates
        population changes predicted by the flux provided by the previous `continuous_adjustment`
        call.

        Args:
            population_change (number): modeled increase or decrease in the specie's population
            time (number): the simulation time at which the predicted change occurs; this time is
                used by `get_population` to interpolate continuous-time predictions between
                integrations.
            flux (number): the predicted flux of the specie at the provided time

        Returns:
            int: the specie's adjusted population, rounded to an integer

        Raises:
            ValueError: if `time` is not greater than the time of the most recent
                `continuous_adjustment` call on this `specie`
            NegativePopulationError: if applying `population_change` makes the population go negative
        '''
        if not self.continuous_submodel:
            raise ValueError("continuous_adjustment(): initial flux was not provided")
        # the simulation time must advance between adjacent continuous adjustments
        if time <= self.continuous_time:
            raise ValueError("continuous_adjustment(): time <= self.continuous_time: "
                "{:.2f} < {:.2f}".format(time, self.continuous_time))
        if self.last_population + population_change < 0:
            raise NegativePopulationError('continuous_adjustment', self.specie_name,
                self.last_population, population_change, time-self.continuous_time)
        self.continuous_time = time
        self.continuous_flux = flux
        self.last_population += population_change
        return self.get_population(time)

    def get_population(self, time=None):
        '''Provide the specie's current population.

        If one of the submodel(s) predicting the specie's population is a continuous-time model,
        then use the specie's last flux to interpolate the current population, as described in the
        class documentation.

        Clearly, species populations in biological systems are non-negative integers. However,
        continuous-time models approximate populations with continuous representations, and
        therefore predict real, non-integral, populations. But discrete-time models like SSA
        do not naturally handle non-integral copy numbers.

        We resolve this conflict by storing real valued populations within a specie, but
        providing only integral population predictions. To aovid the bias that would arise by
        always using floor() or ceiling() to convert a float to an integer, population predictions
        are stochastically rounded before being returned by `get_population`. *This means
        that a sequence of calls to `get_population` which do not have any interveening
        adjustment operations may **NOT** return a sequence of equal population values.*

        Args:
            time (number, optional): the current simulation time; `time` is required if one of the
                submodels modeling the specie is a continuous-time submodel.

        Returns:
            int: an integer approximation of the specie's adjusted population

        Raises:
            ValueError: if `time` is required but not provided
            ValueError: if `time` is earlier than the time of a previous continuous adjustment
            NegativePopulationError: if interpolation predicts a negative population
        '''
        if not self.continuous_submodel:
            return self.random_state.round( self.last_population )
        else:
            if time is None:
                raise ValueError("get_population(): time needed because "
                    "continuous adjustment received at time {:.2f}".format(self.continuous_time))
            if time < self.continuous_time:
                raise ValueError("get_population(): time < self.continuous_time: {:.2f} < {:.2f}\n".format(
                    time, self.continuous_time))
            interpolation=0
            # TODO(Arthur): compare with and wo interpolation
            if config_multialgorithm['interpolate']:
                interpolation = (time - self.continuous_time) * self.continuous_flux
            if self.last_population + interpolation < 0:
                raise NegativePopulationError('get_population', self.specie_name,
                    self.last_population, interpolation, time - self.continuous_time)
            float_copy_number = self.last_population + interpolation
            return self.random_state.round( float_copy_number )

    def __str__(self):
        if self.continuous_submodel:
            return "specie_name: {}; last_population: {}; continuous_time: {}; continuous_flux: {}".format(
                self.specie_name, self.last_population, self.continuous_time, self.continuous_flux)
        else:
            return "specie_name: {}; last_population: {}".format(
                self.specie_name, self.last_population)

    @staticmethod
    def heading():
        '''Return a heading for a tab-separated table of species data.'''
        return '\t'.join('specie_name last_population continuous_time continuous_flux'.split())

    def row(self):
        '''Return a row for a tab-separated table of species data.'''
        if self.continuous_submodel:
            return "{}\t{:.2f}\t{:.2f}\t{:.2f}".format(self.specie_name, self.last_population, self.continuous_time, self.continuous_flux)
            '\t'.join([])
        else:
            return "{}\t{:.2f}".format(self.specie_name, self.last_population)
            '\t'.join([])
