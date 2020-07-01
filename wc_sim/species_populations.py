""" Store species populations, and partition them among submodel private species and shared species

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-04
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

# TODO(Arthur): for reproducibility, use lists instead of sets
# TODO(Arthur): analyze accuracy with and without interpolation

from collections import defaultdict, namedtuple
from enum import Enum
from scipy.constants import Avogadro
import abc
import math
import numpy as np
import sys

from de_sim.simulation_object import (SimulationObject, ApplicationSimulationObject,
                                      AppSimObjAndABCMeta, ApplicationSimulationObjMeta)
from de_sim.utilities import FastLogger
from wc_sim import distributed_properties, message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.debug_logs import logs as debug_logs
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_errors import (DynamicSpeciesPopulationError, DynamicNegativePopulationError,
                                          SpeciesPopulationError)
from wc_utils.util.dict import DictUtil
from wc_utils.util.rand import RandomStateManager
import wc_lang

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']
RUN_TIME_ERROR_CHECKING = config_multialgorithm['run_time_error_checking']


class AccessSpeciesPopulationInterface(metaclass=abc.ABCMeta):   # pragma: no cover; methods in ABCs aren't run
    """ An abstract base class defining the interface between a submodel and its species population store(s)

    A submodel in a WC simulation can interact with multiple components that store the population
    of the species it models. This architecture is needed to simulate a model in parallel. All these
    stores must
    implement this interface which defines read and write operations on the species in a store.
    Both write operations have the prefix `adjust` in their names because they adjust a store's population.
    All operations require a time argument that indicates the simulation time at which the operation
    executes in the store.
    """

    @abc.abstractmethod
    def read_one(self, time, species_id):
        """ Obtain the predicted population of a species at a particular simulation time """
        raise NotImplemented

    @abc.abstractmethod
    def read(self, time, species):
        """ Obtain the predicted population of a list of species at a particular simulation time """
        raise NotImplemented

    @abc.abstractmethod
    def adjust_discretely(self, time, adjustments):
        """ A discrete submodel adjusts the population of a set of species at a particular simulation time """
        raise NotImplemented

    @abc.abstractmethod
    def adjust_continuously(self, time, adjustments):
        """ A continuous submodel adjusts the population of a set of species at a particular simulation time """
        raise NotImplemented


LOCAL_POP_STORE = 'LOCAL_POP_STORE'  # the name of the local population store


# TODO(Arthur): cover after MVP wc_sim done
class AccessSpeciesPopulations(AccessSpeciesPopulationInterface):   # pragma: no cover
    """ Interface a submodel with the components that store the species populations it models

    Each submodel is a distinct simulation object. In the current population-based model,
    species are represented by their populations. (A hybrid instance-population model would change
    that.) Each submodel accesses a subset of the species in a model. A submodel's
    species can be partitioned into those that are accessed ONLY by the submodel and those that it
    shares with other submodels. These are respectively stored in a local LocalSpeciesPopulation
    which is private to this submodel, and a set of SpeciesPopSimObjects which are shared with other
    submodels. LocalSpeciesPopulation objects are accessed via local memory operations whereas
    SpeciesPopSimObjects, which are distinct simulation objects, are accessed via simulation event
    messages.

    AccessSpeciesPopulations enables a submodel to access all of the species populations that it
    uses through a single convenient interface. The submodel simply indicates the species
    being used and the operation type. This object then maps the species to the entity or entities
    storing them, and executes the operation on each entity.

    Essentially, an AccessSpeciesPopulations multiplexes a submodel's access to multiple population
    stores.

    Attributes:
        submodel (:obj:`DynamicSubmodel`): the submodel that's using this :obj:`AccessSpeciesPopulations`
        species_locations (:obj:`dict` of `str`): a map indicating the store for each species used
            by the submodel using this object, that is, the local submodel.
        local_pop_store (:obj:`LocalSpeciesPopulation`): a store of local species
        remote_pop_stores (:obj:`dict` of identifiers of `SpeciesPopSimObject`): a map from store name
            to a system identifier for the remote population store(s) that the local submodel uses.
            For a shared memory implementation system identifiers can be object references; for a
            distributed implementation they must be network object identifiers.
        species_population_cache (:obj:`SpeciesPopulationCache`): a cache of populations for species
            that are stored remotely in the SpeciesPopSimObjects in remote_pop_stores; values for
            remote populations are pre-fetched at the right simulation time (via GetPopulation and
            GivePopulation messages) into this cache and then read from it when needed.
    """
    def __init__(self, local_pop_store, remote_pop_stores):
        """ Initialize an AccessSpeciesPopulations object

        The submodel object referenced in the attribute submodel must reference this
        AccessSpeciesPopulations instance. This object must be instantiated first; then the submodel
        can be created and set its reference with the set_*() method below.

        Raises:
            :obj:`SpeciesPopulationError`: if the remote_pop_stores contains a store named
                'LOCAL_POP_STORE', which is a reserved store identifier for the local_pop_store.
        """
        self.local_pop_store = local_pop_store
        if LOCAL_POP_STORE in remote_pop_stores:
            raise SpeciesPopulationError("AccessSpeciesPopulations.__init__: {} not a valid "
                                         "remote_pop_store name".format(LOCAL_POP_STORE))
        self.remote_pop_stores = remote_pop_stores
        self.species_locations = {}
        self.species_population_cache = SpeciesPopulationCache(self)

    def set_submodel(self, submodel):
        """ Set the submodel that uses this AccessSpeciesPopulations """
        self.submodel = submodel

    def add_species_locations(self, store_name, species_ids, replace=False):
        """ Add species locations to the species location map

        Record that the species listed in `species_ids` are stored by the species population store
        identified by `store_name`. To replace existing location map values without raising an
        exception, set `replace` to True.

        Args:
            store_name (:obj:`str`): the globally unique name of a species population store. `LOCAL_POP_STORE`
                is a special name that identifies the local population store for private species
            species_ids (:obj:`list` of :obj:`str`): a list of species ids

        Raises:
            :obj:`SpeciesPopulationError`: if store `store_name` is unknown
            :obj:`SpeciesPopulationError`: if `replace` is False and any species_id in `species_ids` is
                already mapped to a different store than `store_name`.
        """
        if not store_name in self.remote_pop_stores.keys() and store_name != LOCAL_POP_STORE:
            raise SpeciesPopulationError("add_species_locations: '{}' not a known population "
                                         "store.".format(store_name))
        if replace:
            for species_id in species_ids:
                self.species_locations[species_id] = store_name
        else:
            assigned = list(filter(lambda s: s in self.species_locations.keys(), species_ids))
            if assigned:
                raise SpeciesPopulationError("add_species_locations: species {} already have assigned "
                                             "locations.".format(sorted(assigned)))
            for species_id in species_ids:
                self.species_locations[species_id] = store_name

    def del_species_locations(self, species_ids, force=False):
        """ Delete entries from the species location map

        Remove species location mappings for the species in `species_ids`. To avoid raising an
        exception when a species is not in the location map, set `force` to `True`.

        Args:
            species_ids (:obj:`list` of species_ids): a list of species ids
            force (:obj:`boolean`, optional): if set, do not raise an exception if a species_id in
                `species_ids` is not found in the species location map.

        Raises:
            :obj:`SpeciesPopulationError`: if `force` is False and any species_id in `species_ids` is not in the
                species location map.
        """
        if force:
            for species_id in species_ids:
                try:
                    del self.species_locations[species_id]
                except KeyError:
                    pass
        else:
            unassigned = list(filter(lambda s: s not in self.species_locations.keys(), species_ids))
            if unassigned:
                raise SpeciesPopulationError("del_species_locations: species {} are not in the location "
                                             "map.".format(sorted(unassigned)))
            for species_id in species_ids:
                del self.species_locations[species_id]

    def locate_species(self, species_ids):
        """ Locate the component(s) that store a set of species

        Given a list of species identifiers in `species_ids`, partition them into the storage
        component(s) that store their populations. This method is widely used by code that accesses
        species. It returns a dictionary that maps from store name to the ids of species whose
        populations are modeled by the store.

        The special name `LOCAL_POP_STORE` represents a special store, the local
        :obj:`wc_sim.local_species_population.LocalSpeciesPopulation` instance. Each other
        store is identified by the name of a remote
        :obj:`wc_sim.species_pop_sim_object.SpeciesPopSimObject` instance.

        Args:
            species_ids (:obj:`list` of :obj:`str`): a list of species identifiers

        Returns:
            dict: a map from store_name -> a set of species_ids whose populations are stored
                by component store_name.

        Raises:
            :obj:`SpeciesPopulationError`: if a store cannot be found for a species_id in `species_ids`
        """
        unknown = list(filter(lambda s: s not in self.species_locations.keys(), species_ids))
        if unknown:
            raise SpeciesPopulationError("locate_species: species {} are not "
                                         "in the location map.".format(sorted(unknown)))
        inverse_loc_map = defaultdict(set)
        for species_id in species_ids:
            store = self.species_locations[species_id]
            inverse_loc_map[store].add(species_id)
        return inverse_loc_map

    def read_one(self, time, species_id):
        """ Obtain the predicted population of species`species_id` at the time `time`

        If the species is stored in the local_pop_store, obtain its population there. Otherwise obtain
        the population from the species_population_cache. If the species' primary store is a
        remote_pop_store, then its population should be in the cache because the population should
        have been prefetched.

        Args:
            time (:obj:`float`): the time at which the population should be obtained
            species_id (:obj:`str`): identifier of the species whose population will be obtained.

        Returns:
            float: the predicted population of `species_id` at simulation time `time`.

        Raises:
            :obj:`SpeciesPopulationError`: if `species_id` is an unknown species
        """
        if species_id not in self.species_locations:
            raise SpeciesPopulationError("read_one: species'{}' not in the location map.".format(
                species_id))
        store = self.species_locations[species_id]
        if store == LOCAL_POP_STORE:
            return self.local_pop_store.read_one(time, species_id)
        else:
            # TODO(Arthur): convert print() to log message
            # print('submodel {} reading {} from cache at {:.2f}'.format(self.submodel.name,
            #   species_id, time))
            return self.species_population_cache.read_one(time, species_id)

    def read(self, time, species_ids):
        """ Obtain the population of the species identified in `species_ids` at the time `time`

        Obtain the species from the local_pop_store and/or the species_population_cache. If some of
        the species' primary stores are remote_pop_stores, then their populations should be in the
        cache because they should have been prefetched.

        Args:
            time (:obj:`float`): the time at which the population should be obtained
            species_ids (set): identifiers of the species whose populations will be obtained.

        Returns:
            dict: species_id -> population; the predicted population of all requested species at
            time `time`.

        Raises:
            :obj:`SpeciesPopulationError`: if a store cannot be found for a species_id in `species_ids`
            :obj:`SpeciesPopulationError`: if any of the species were cached at a time that differs from `time`
        """
        local_species = self.locate_species(species_ids)[LOCAL_POP_STORE]
        remote_species = set(species_ids) - set(local_species)

        local_pops = self.local_pop_store.read(time, local_species)
        cached_pops = self.species_population_cache.read(time, remote_species)

        cached_pops.update(local_pops)
        return cached_pops

    def adjust_discretely(self, time, adjustments):
        """ A discrete submodel adjusts the population of a set of species at the time `time`

        Distribute the adjustments among the population stores managed by this object.
        Iterate through the components that store the population of species listed in `adjustments`.
        Update the local population store immediately and send AdjustPopulationByDiscreteSubmodel
        messages to the remote stores. Since these messages are asynchronous, this method returns
        as soon as they are sent.

        Args:
            time (:obj:`float`): the time at which the population is being adjusted
            adjustments (:obj:`dict` of :obj:`float`): map: species_ids -> population_adjustment; adjustments
                to be made to some species populations

        Returns:
            :obj:`list`: the names of the stores for the species whose populations are adjusted
        """
        stores = []
        for store, species_ids in self.locate_species(adjustments.keys()).items():
            stores.append(store)
            store_adjustments = DictUtil.filtered_dict(adjustments, species_ids)
            if store == LOCAL_POP_STORE:
                self.local_pop_store.adjust_discretely(time, store_adjustments)
            else:
                self.submodel.send_event(time-self.submodel.time,
                                         self.remote_pop_stores[store],
                                         message_types.AdjustPopulationByDiscreteSubmodel(store_adjustments))
        return stores

    def adjust_continuously(self, time, adjustments):
        """ A continuous submodel adjusts the population of a set of species at the time `time`

        Args:
            time (:obj:`float`): the time at which the population is being adjusted
            adjustments (:obj:`dict` of `tuple`): map: species_ids -> population_slope;
                adjustments to be made to some species populations.

        See the description for `adjust_discretely` above.

        Returns:
            list: the names of the stores for the species whose populations are adjusted.
        """
        stores = []
        for store, species_ids in self.locate_species(adjustments.keys()).items():
            stores.append(store)
            store_adjustments = DictUtil.filtered_dict(adjustments, species_ids)
            if store == LOCAL_POP_STORE:
                self.local_pop_store.adjust_continuously(time, store_adjustments)
            else:
                self.submodel.send_event(time-self.submodel.time,
                                         self.remote_pop_stores[store],
                                         message_types.AdjustPopulationByContinuousSubmodel(store_adjustments))
        return stores

    def prefetch(self, delay, species_ids):
        """ Obtain species populations from remote stores when they will be needed in the future

        Generate GetPopulation queries that obtain species populations whose primary stores are
        remote at `delay` in the future. The primary stores (`SpeciesPopSimObject` objects)
        will respond to the GetPopulation queries with GivePopulation responses.

        To ensure that the remote store object executes the GetPopulation at an earlier simulation
        time than the submodel will need the data, decrease the event time of the GetPopulation
        event to the previous floating point value.

        Args:
            delay (:obj:`float`): the populations will be needed at now + `delay`
            species_ids (:obj:`list` of species_ids): a list of species ids

        Returns:
            list: the names of the stores for the species whose populations are adjusted.
        """
        # TODO(Arthur): IMPORTANT optimizations: reduce rate of prefetch
        # 1: store most species locally
        # 2: instead of sending GetPopulation messages to retrieve populations may be unchanged,
        #   make write-through caches which push population updates from reaction executions
        # 3: draw reaction partition boundaries over species (edges) that rarely update
        if delay <= 0:
            raise SpeciesPopulationError("prefetch: {} provided, but delay must "
                                         "be non-negative.".format(delay))
        stores = []
        for store, species_ids in self.locate_species(species_ids).items():
            if store != LOCAL_POP_STORE:
                stores.append(store)
                # advance the receipt of GetPopulation so the SpeciesPopSimObject executes it before
                # the submodel needs the value
                self.submodel.send_event(delay,
                                         self.remote_pop_stores[store],
                                         message_types.GetPopulation(set(species_ids)))
        return stores

    def __str__(self):
        """ Provide readable AccessSpeciesPopulations state

        Provide the submodel's name, the name of the local_pop_store, and the id and store name of
        each species accessed by this AccessSpeciesPopulations.

        Returns:
            :obj:`str`: a multi-line string describing this AccessSpeciesPopulations' state.
        """

        state = ['AccessSpeciesPopulations state:']
        if hasattr(self, 'submodel'):
            state.append('submodel: {}'.format(self.submodel.id))
        state.append('local_pop_store: {}'.format(self.local_pop_store.name))
        state.append('species locations:')
        state.append('species_id\tstore_name')
        state += ['{}\t{}'.format(k, self.species_locations[k])
                  for k in sorted(self.species_locations.keys())]
        return '\n'.join(state)


# TODO(Arthur): cover after MVP wc_sim done
class SpeciesPopulationCache(object):       # pragma: no cover
    """ Cache the population of species whose primary stores are remote population stores

    Attributes:
        access_species_populations (:obj:`AccessSpeciesPopulations`): the `AccessSpeciesPopulations`
            containing this Object.
        _cache (:obj:`dict` of :obj:`tuple`): map: species_id -> (time, population); the species
            whose counts are cached, containing the last write time in `time`, and the population.
    """

    def __init__(self, access_species_populations):
        """ Initialize a SpeciesPopulationCache object """
        self.access_species_populations = access_species_populations
        self._cache = {}

    def clear_cache(self):
        """ Clear the cache """
        self._cache = {}

    def cache_population(self, time, populations):
        """ Cache some population values

        Args:
            time (:obj:`float`): the time of the cached values
            populations (:obj:`dict` of float): map: species_ids -> population; the population
                of the species at `time`.

        Raises:
            :obj:`SpeciesPopulationError`: if the species are stored in the ASP's local store; they should
                not be cached.
            :obj:`SpeciesPopulationError`: if, for any species, `time` is not greater than its previous
                cache time.
        """
        # raise exception if the species are stored in the ASP's local store
        store_name_map = self.access_species_populations.locate_species(populations.keys())
        if LOCAL_POP_STORE in store_name_map:
            raise SpeciesPopulationError("cache_population: some species are stored in the "
                                         "AccessSpeciesPopulations's local store: {}.".format(
                                             list(store_name_map[LOCAL_POP_STORE])))
        # TODO(Arthur): could raise an exception if the species are not stored in the ASP's remote stores
        for species_id, population in populations.items():
            # raise exception if the time of this cache is not greater than the previous cache time
            if species_id in self._cache and time <= self._cache[species_id][0]:
                raise SpeciesPopulationError(f"cache_population: caching an earlier population: "
                                             f"species_id: {species_id}; current time: {time} <= "
                                             f"previous time {self._cache[species_id][0]}.")
            self._cache[species_id] = (time, population)

    def read_one(self, time, species_id):
        """ Obtain the cached population of a species at a particular time

        Args:
            time (:obj:`float`): the expected time of the cached values
            species_id (:obj:`str`): identifier of the species to obtain.

        Returns:
            float: the cached population of `species_id` at simulation time `time`.

        Raises:
            :obj:`SpeciesPopulationError`: if the species are stored in the ASP's local store, which means
                that they should not be cached.
            :obj:`SpeciesPopulationError`: if `time` is not greater than a species' previous cache time
        """
        if species_id not in self._cache:
            raise SpeciesPopulationError("SpeciesPopulationCache.read_one: species'{}' not "
                                         "in cache.".format(species_id))
        if self._cache[species_id][0] < time:
            raise SpeciesPopulationError("SpeciesPopulationCache.read_one: cache age of {} too big "
                                         "for read at time {} of species'{}'.".format(
                                             time-self._cache[species_id][0], time, species_id))
        return self._cache[species_id][1]

    def read(self, time, species_ids):
        """ Read the cached population of a set of species at a particular time

        Args:
            time (:obj:`float`): the time at which the population should be obtained
            species_ids (set): identifiers of the species to read.

        Returns:
            species counts: dict: species_id -> copy_number; the cached copy number of each
            requested species at simulation time `time`.

        Raises:
            :obj:`SpeciesPopulationError`: if any of the species are not stored in the cache
            :obj:`SpeciesPopulationError`: if any of the species were cached at a time that differs
                from `time`.
        """
        missing = list(filter(lambda species: species not in self._cache, species_ids))
        if missing:
            raise SpeciesPopulationError("SpeciesPopulationCache.read: species {} not in cache.".format(
                str(missing)))
        mistimed = list(filter(lambda s_id: self._cache[s_id][0] < time, species_ids))
        if mistimed:
            raise SpeciesPopulationError("SpeciesPopulationCache.read: species {} not reading "
                                         "recently cached value(s).".format(str(mistimed)))
        return {species: self._cache[species][1] for species in species_ids}


# TODO(Arthur): after MVP wc_sim is done, replace references to LocalSpeciesPopulation with
# references to AccessSpeciesPopulations


class LocalSpeciesPopulation(AccessSpeciesPopulationInterface):
    """ Maintain the population of a set of species

    `LocalSpeciesPopulation` tracks the population of a set of species. Population values (copy numbers)
    can be read or modified (adjusted). To enable multi-algorithmic modeling, it supports writes to
    a species' population by both discrete time and continuous time submodels.

    All access operations that read or modify the population must provide a simulation time.
    For any given species, all operations must occur in non-decreasing simulation time order.
    Record history operations must also occur in time order.

    Simulation time arguments enable detection of temporal causality errors by shared accesses from
    different submodels in a sequential
    simulator. In particular, every read operation must access the previous modification.

    A `LocalSpeciesPopulation` object is accessed via local method calls. It can be wrapped as a
    DES simulation object -- a `SimulationObject` -- to provide distributed access, as is done by
    `SpeciesPopSimObject`.

    Attributes:
        name (:obj:`str`): the name of this object
        time (:obj:`float`): the time of the most recent access to this `LocalSpeciesPopulation`
        _molecular_weights (:obj:`dict` of :obj:`float`): map: species_id -> molecular_weight; the
            molecular weight of each species
        _population (:obj:`dict` of :obj:`DynamicSpeciesState`): map: species_id -> DynamicSpeciesState();
            the species whose counts are stored, represented by DynamicSpeciesState objects.
        _cached_species_ids (:obj:`set`) ids of all species in `_population`; cached to enhance performance
        last_access_time (:obj:`dict` of :obj:`float`): map: species_name -> last_time; the last time at
            which the species was accessed.
        _history (:obj:`dict`) nested dict; an optional history of the species' state. The population
            history is recorded at each continuous adjustment.
        random_state (:obj:`np.random.RandomState`): a PRNG used by all `Species`
        fast_debug_file_logger (:obj:`FastLogger`): a fast logger for debugging messages
        temporary_mode (:obj:`bool`): if True, this `LocalSpeciesPopulation` is being accessed through a
            `TempPopulationsLSP`
    """
    # TODO(Arthur): support tracking the population history of species added at any time in the simulation
    # TODO(Arthur): report an error if a DynamicSpeciesState is updated by multiple continuous submodels
    # because modeling a linear superposition of species population slopes is not supported
    # TODO(Arthur): molecular_weights should accept MW of each species type, like the model does
    def __init__(self, name, initial_population, molecular_weights, initial_population_slopes=None,
                 initial_time=0, random_state=None, retain_history=True):
        """ Initialize a :obj:`LocalSpeciesPopulation` object

        Initialize a :obj:`LocalSpeciesPopulation` object. Establish its initial population, and initialize
            the history if `retain_history` is `True`.

        Args:
            name (:obj:`str`): the name of this object
            initial_population (:obj:`dict` of :obj:`float`): initial population for some species;
                dict: species_id -> initial_population
            molecular_weights (:obj:`dict` of :obj:`float`): map: species_id -> molecular_weight,
                provided for computing the mass of lists of species in a `LocalSpeciesPopulation`
            initial_population_slopes (:obj:`dict` of :obj:`float`, optional): map: species_id -> initial_slope;
                initial rate of change must be provided for all species whose populations are predicted
                by a submodel that uses a continuous integration algorithm. These are ignored for
                species not specified in `initial_population`.
            initial_time (:obj:`float`, optional): the initialization time; defaults to 0
            random_state (:obj:`np.random.RandomState`, optional): a PRNG used by all `DynamicSpeciesState`
            retain_history (:obj:`bool`, optional): whether to retain species population history
            _concentrations_api (:obj:`bool`, optional): if set, use concentrations; species amounts
                passed into and returned by methods must be concentrations (molar == mol/L); defaults to `False`

        Raises:
            :obj:`SpeciesPopulationError`: if the population cannot be initialized
        """
        self.name = name
        self.time = initial_time
        self._population = {}
        self._cached_species_ids = set()
        self._molecular_weights = {}
        self.last_access_time = {}
        self.random_state = random_state
        self._concentrations_api = False
        self.temporary_mode = False

        if retain_history:
            self._initialize_history()

        unknown_weights = set(initial_population.keys()) - set(molecular_weights.keys())
        if unknown_weights:
            # raise exception if any species are missing weights
            raise SpeciesPopulationError("Cannot init LocalSpeciesPopulation because some species "
                                         "are missing weights: {}".format(
                                             ', '.join([f"'{str(uw)}'" for uw in unknown_weights])))

        for species_id in initial_population:
            if initial_population_slopes is not None and species_id in initial_population_slopes:
                self.init_cell_state_species(species_id, initial_population[species_id],
                                             molecular_weights[species_id],
                                             initial_population_slopes[species_id])
            else:
                self.init_cell_state_species(species_id, initial_population[species_id],
                                             molecular_weights[species_id])

        # log initialization data
        self.fast_debug_file_logger = FastLogger(debug_logs.get_log('wc.debug.file'), 'debug')
        self.fast_debug_file_logger.fast_log("LocalSpeciesPopulation.__init__: initial_population: {}".format(
                                             DictUtil.to_string_sorted_by_key(initial_population)), sim_time=self.time)
        self.fast_debug_file_logger.fast_log("LocalSpeciesPopulation.__init__: initial_population_slopes: {}".format(
                                             DictUtil.to_string_sorted_by_key(initial_population_slopes)),
                                             sim_time=self.time)

    def init_cell_state_species(self, species_id, population, molecular_weight, initial_population_slope=None):
        """ Initialize a species with the given population and population slope

        Add a species to the cell state. The species' population is set at the current time.

        Args:
            species_id (:obj:`str`): the species' globally unique identifier
            population (:obj:`float`): initial population of the species
            molecular_weight (:obj:`float`): molecular weight of the species
            initial_population_slope (:obj:`float`, optional): an initial rate of change for the species

        Raises:
            :obj:`SpeciesPopulationError`: if the species is already stored by this LocalSpeciesPopulation
        """
        if species_id in self._population:
            raise SpeciesPopulationError("species_id '{}' already stored by this "
                                         "LocalSpeciesPopulation".format(species_id))
        modeled_continuously = initial_population_slope is not None
        self._population[species_id] = DynamicSpeciesState(species_id, self.random_state, population,
                                                           modeled_continuously=modeled_continuously)
        self._add_to_cached_species_ids(species_id)
        self._molecular_weights[species_id] = molecular_weight

        if modeled_continuously:
            self._population[species_id].continuous_adjustment(self.time, initial_population_slope)
        self.last_access_time[species_id] = self.time
        self._add_to_history(species_id)

    def _add_to_cached_species_ids(self, id):
        """ Add a species ID to the cached species IDs

        Args:
            id (:obj:`str`): the ID of a species stored in `_population`
        """
        self._cached_species_ids.add(id)

    def _all_species(self):
        """ Return the IDs of species known by this :obj:`LocalSpeciesPopulation`

        Returns:
            :obj:`set`: the species known by this :obj:`LocalSpeciesPopulation`
        """
        return self._cached_species_ids

    # todo: stop requiring that species be in sets, instead require iterator and remove set(species) below
    def _check_species(self, time, species=None, check_early_accesses=True):
        """ Check whether the species are a set, or not known by this LocalSpeciesPopulation

        Also checks whether the species are being accessed in time order if `check_early_accesses`
        is set.
        Does nothing if `RUN_TIME_ERROR_CHECKING` is `False`.

        Args:
            time (:obj:`float`): the time at which the population might be accessed
            species (:obj:`set`, optional): set of species_ids; if not supplied, read all species
            check_early_accesses (:obj:`bool`, optional): whether to check for early accesses

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if species is not a set
            :obj:`DynamicSpeciesPopulationError`: if any species in `species` do not exist
            :obj:`DynamicSpeciesPopulationError`: if a species in `species` is being accessed at a time earlier
                than a prior access.
        """
        if RUN_TIME_ERROR_CHECKING:
            if not species is None:
                if not isinstance(species, set):
                    raise DynamicSpeciesPopulationError(time, "species '{}' must be a set".format(species))
                unknown_species = species - self._all_species()
                if unknown_species:
                    # raise exception if some species are non-existent
                    raise DynamicSpeciesPopulationError(time, "request for population of unknown species: {}".format(
                        ', '.join(map(lambda x: "'{}'".format(str(x)), unknown_species))))
                if check_early_accesses:
                    early_accesses = list(filter(lambda s: time < self.last_access_time[s], species))
                    if early_accesses:
                        raise DynamicSpeciesPopulationError(time, "access at time {} is an earlier access of "
                                                     "species {} than at {}".format(time, early_accesses,
                                                     [self.last_access_time[s] for s in early_accesses]))

    def _update_access_times(self, time, species=None):
        """ Update the access time to `time` for all species_ids in `species`

        Args:
            time (:obj:`float`): the access time which should be set for the species
            species (:obj:`set`, optional): a set of species_ids; if not provided, read all species
        """
        if species is None:
            species = self._all_species()
        for species_id in species:
            self.last_access_time[species_id] = time

    def _accounted_volumes_for_species(self, time, species, dynamic_model):
        """ Compute the accounted for volumes of the compartments containing some species

        Args:
            time (:obj:`float`): current simulation time, for error reporting
            species (:obj:`iterator` of :obj:`str`): iterator over species id
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model

        Returns:
            :obj:`dict` of :obj:`float`: the accounted for volumes of the compartments containing the
                species; map: compartment id -> volume

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if no species are provided
        """
        if not species:
            raise DynamicSpeciesPopulationError(time, f"no species provided")

        # compute the volume of each compartment only once
        volumes = {}
        for species_id in species:
            compartment_id = self._population[species_id].compartment_id
            if compartment_id not in volumes:
                volumes[compartment_id] = \
                    dynamic_model.dynamic_compartments[compartment_id].accounted_volume()
        return volumes

    def get_continuous_species(self):
        """ Get the species that are modeled continuously

        This is used to compute dependencies in the Next Reaction Method submodel.

        Returns:
            :obj:`set` of :obj:`str`: IDs of the species that are modeled continuously
        """
        continuously_modeled_species = set()
        for species_id, dynamic_species_state in self._population.items():
            if dynamic_species_state.modeled_continuously:
                continuously_modeled_species.add(species_id)
        return continuously_modeled_species

    def populations_to_concentrations(self, time, species, populations, dynamic_model, concentrations,
                                      volumes=None):
        """ Convert species populations, in molecules, to concentrations, in molar

        Args:
            time (:obj:`float`): current simulation time
            species (:obj:`np.ndarray` of :obj:`str`): species ids
            populations (:obj:`np.ndarray` of :obj:`float`): corresponding populations of the species
                in `species`
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            concentrations (:obj:`np.ndarray` of :obj:`float`): array to hold the concentrations of
                the corresponding species in `species`
            volumes (:obj:`dict` of :obj:`float`, optional): map: compartment id -> volume; volumes
                of compartments containing species in `species`; optionally provided as an
                optimization, to avoid computing volumes

        Returns:
            :None`: the concentrations are returned in `concentrations`
        """
        if volumes is None:
            volumes = self._accounted_volumes_for_species(time, species, dynamic_model)

        # concentration (mole/liter) = population (molecule) / (Avogadro (molecule/mole) * volume (liter))
        for i in range(len(species)):
            species_compartment_id = self._population[species[i]].compartment_id
            concentrations[i] = populations[i] / (Avogadro * volumes[species_compartment_id])

    def concentrations_to_populations(self, time, species, concentrations, dynamic_model, populations,
                                      volumes=None):
        """ Convert species concentrations, in molar, to populations, in molecules

        Args:
            time (:obj:`float`): current simulation time
            species (:obj:`np.ndarray` of :obj:`str`): species ids
            concentrations (:obj:`np.ndarray` of :obj:`float`): corresponding concentrations of the
                species in `species`
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            populations (:obj:`np.ndarray` of :obj:`float`): array to hold the populations of
                the corresponding species in `species`
            volumes (:obj:`dict` of :obj:`float`, optional): map: compartment id -> volume; volumes
                of compartments containing species in `species`; optionally provided as an
                optimization, to avoid computing volumes

        Returns:
            :None`: the populations are returned in `populations`
        """
        if volumes is None:
            volumes = self._accounted_volumes_for_species(time, species, dynamic_model)

        # population (molecule) = concentration (mole/liter) * Avogadro (molecule/mole) * volume (liter)
        for i in range(len(species)):
            species_compartment_id = self._population[species[i]].compartment_id
            populations[i] = concentrations[i] * Avogadro * volumes[species_compartment_id]

    def concentrations_api(self):
        """ Get the status of the concentrations API; if `True`, species amounts must be in molar

        Returns:
            :obj:`bool`: the status of the concentrations API; if set, species amounts must be in molar
        """
        return self._concentrations_api

    def concentrations_api_on(self):
        """ Turn the concentrations API on; species amounts must be in concentrations (molar)
        """
        self._concentrations_api = True

    def concentrations_api_off(self):
        """ Turn the concentrations API on; species amounts must be in populations (molecule)
        """
        self._concentrations_api = False

    def read_one(self, time, species_id):
        """ Obtain the predicted population of species `species_id` at simulation time `time`

        If a temporary population value is available, return it.

        Args:
            time (:obj:`float`): the time at which the population should be estimated
            species_id (:obj:`str`): identifier of the species to access.

        Returns:
            :obj:`float`: the predicted population of `species_id` at simulation time `time`.
        """
        species_id_in_set = {species_id}
        self._check_species(None, species_id_in_set, check_early_accesses=False)
        dynamic_species_state = self._population[species_id]
        if not dynamic_species_state.get_temp_population_value() is None:
            return dynamic_species_state.get_temp_population_value()
        self._check_species(time, species_id_in_set)
        self.time = time
        self._update_access_times(time, species_id_in_set)
        return dynamic_species_state.get_population(time, temporary_mode=self.temporary_mode)

    def read(self, time, species=None, round=True):
        """ Read the predicted population of a multiple species at simulation time `time`

        Ignores temporary species populations

        Args:
            time (:obj:`float`): the time at which the population should be estimated
            species (:obj:`set`, optional): identifiers of the species to read; if not supplied, read all species
            round (:obj:`bool`, optional): if `round` then round the populations to integers

        Returns:
            :obj:`dict`: species counts: species_id -> copy_number; the predicted copy number of each
            requested species at `time`
        """
        if species is None:
            species = self._all_species()
        self._check_species(time, species)
        self.time = time
        self._update_access_times(time, species)
        return {s: self._population[s].get_population(time, round=round) for s in species}

    def read_into_array(self, time, species, populations, round=True):
        """ Obtain the predicted population of an iterator over species at simulation time `time`

        Args:
            time (:obj:`float`): the time at which the population should be predicted
            species (:obj:`iterator` of :obj:`str`): identifiers of the species to read
            populations (:obj:`np.ndarray` of :obj:`float`): array to hold the populations of
                the corresponding species in `species`
            round (:obj:`bool`, optional): if `round` then round the populations to integers

        Returns:
            :None`: the populations are returned in `populations`
        """
        self._check_species(time, set(species))
        self.time = time
        self._update_access_times(time, species)
        for idx, species_id in enumerate(species):
            populations[idx] = self._population[species_id].get_population(time, round=round)
    # todo: combine read() and read_into_array() into 1 method

    def set_temp_populations(self, populations):
        """ Set temporary population values for multiple species

        Used to solve ODE submodels

        Args:
            populations (:obj:`dict` of :obj:`float`): map: species_id -> temporary_population_value

        Raises:
            :obj:`SpeciesPopulationError`: if any of the species_ids in `populations` are unknown,
                or if any population value would become negative
        """
        species_ids = set(populations)
        self._check_species(None, species_ids, check_early_accesses=False)
        errors = []
        for species_id, population in populations.items():
            if population < 0:
                errors.append(f"cannot use negative population {population} for species {species_id}")
        if errors:
            raise SpeciesPopulationError("set_temp_populations error(s):\n{}".format('\n'.join(errors)))
        for species_id, population in populations.items():
            self._population[species_id].set_temp_population_value(population)

    def clear_temp_populations(self, species_ids):
        """ Clear temporary population values for multiple species

        Used to solve ODE submodels

        Args:
            species_ids (:obj:`iterator`): an iterator over some species ids
        """
        species_ids = set(species_ids)
        self._check_species(None, species_ids, check_early_accesses=False)
        for species_id in species_ids:
            self._population[species_id].clear_temp_population_value()

    def adjust_discretely(self, time, adjustments):
        """ A submodel adjusts the population of a set of species at simulation time `time`

        Args:
            time (:obj:`float`): the simulation time of the population adjustedment
            adjustments (:obj:`dict` of :obj:`float`): map: species_ids -> population_adjustment; adjustments
                to be made to the population of some species

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if any adjustment attempts to change the population
                of an unknown species
            :obj:`DynamicSpeciesPopulationError`: if any population estimate would become negative
        """
        self._check_species(time, set(adjustments.keys()))
        self.time = time
        errors = []
        for species in adjustments:
            try:
                self._population[species].discrete_adjustment(self.time, adjustments[species])
                self._update_access_times(time, {species})
                self.log_event('discrete_adjustment', self._population[species])
            except DynamicNegativePopulationError as e:
                errors.append(str(e))
        if errors:
            raise DynamicSpeciesPopulationError(time, "adjust_discretely error(s):\n{}".format(
                                                '\n'.join(errors)))
        # TODO(Arthur): exact caching: adjust mass
        '''
        # adjustment to mass of each compartment
        mass_adjustments = defaultdict(float)
        for species_id in adjustments:
                comp_id = self._population[species_id].compartment_id
                mw = self._molecular_weights[species_id]
                mass_adjustments[comp_id] += mw * adjustments[species_id]
        # make mass_adjustments to masses of compartments stored by DynamicModel
        return mass_adjustments
        '''

    def adjust_continuously(self, time, population_slopes):
        """ A continuous submodel adjusts the population slopes of a set of species at simulation time `time`

        Species retain the population slopes to interpolate the population until the next
        call to `adjust_continuously`.

        Args:
            time (:obj:`float`): the time at which the population is being adjusted
            population_slopes (:obj:`dict` of :obj:`float`): map: species_id -> population_slope;
                updated population slopes for some, or all, species populations

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if any adjustment attempts to change the population slope
                of an unknown species,
                or if any population estimate would become negative
        """
        self._check_species(time, set(population_slopes.keys()))
        self.time = time

        errors = []
        for species_id, population_slope in population_slopes.items():
            try:
                self._population[species_id].continuous_adjustment(time, population_slope)
                self._update_access_times(time, [species_id])
                self.log_event('continuous_adjustment', self._population[species_id])
            except (DynamicSpeciesPopulationError, DynamicNegativePopulationError) as e:
                errors.append(str(e))
        if errors:
            self.fast_debug_file_logger.fast_log("LocalSpeciesPopulation.adjust_continuously: error: on species {}: {}".format(
                                                 species_id, '\n'.join(errors)), sim_time=self.time)
            raise DynamicSpeciesPopulationError(time, "adjust_continuously error(s):\n{}".format('\n'.join(errors)))

    # TODO(Arthur): don't need compartment_id, because compartment is part of the species_ids
    # TODO(Arthur): to speed-up convert species ids, molecular weights and populations into arrays
    # TODO(Arthur): raise an error if time is in the future
    def compartmental_mass(self, compartment_id, species_ids=None, time=None):
        """ Compute the current mass of some, or all, species in a compartment

        Args:
            compartment_id (:obj:`str`): the ID of the compartment
            species_ids (:obj:`list` of `str`, optional): identifiers of the species whose mass will be
                obtained; if not provided, then compute the mass of all species in the compartment
            time (number, optional): the current simulation time

        Returns:
            :obj:`float`: the current total mass of the specified species in compartment `compartment_id`, in grams

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if a species' molecular weight was not provided to
                `__init__()` in `molecular_weights`
        """
        if species_ids is None:
            species_ids = self._all_species()
        if time is None:
            time = self.time
        mass = 0.
        for species_id in species_ids:
            comp = self._population[species_id].compartment_id
            if comp == compartment_id:
                try:
                    mw = self._molecular_weights[species_id]
                except KeyError as e:   # pragma: no cover
                    raise DynamicSpeciesPopulationError(time, f"molecular weight not available for '{species_id}'")
                if not np.isnan(mw):
                    mass += mw * self.read_one(time, species_id)
        return mass / Avogadro

    def invalid_weights(self, species_ids=None):
        """ Find the species that do not have a positive, numerical molecular weight

        Args:
            species_ids (:obj:`list` of `str`, optional): identifiers of the species whose molecular weights
                will be checked; if not provided, then check all species

        Returns:
            :obj:`set`: the ids of species that do not have a positive, numerical molecular weight
        """
        if species_ids is None:
            species_ids = self._all_species()
        species_with_invalid_mw = set()
        for species_id in species_ids:
            try:
                mw = self._molecular_weights[species_id]
            except KeyError:
                species_with_invalid_mw.add(species_id)
                continue
            try:
                if 0 < mw:
                    continue
                species_with_invalid_mw.add(species_id)
            except (TypeError, ValueError):
                species_with_invalid_mw.add(species_id)
        return species_with_invalid_mw

    def log_event(self, message, species):
        """ Log an event that modifies a species' population

        Log the event's simulation time, event type, species population, and current population slope
        (if specified).

        Args:
            message (:obj:`str`): description of the event's type.
            species (:obj:`DynamicSpeciesState`): the object whose adjustment is being logged
        """
        try:
            population_slope = species.population_slope
        except AttributeError:
            population_slope = None
        values = [message, species.last_population, population_slope]
        values = map(lambda x: str(x), values)
        # log Sim_time Adjustment_type New_population New_population_slope
        self.fast_debug_file_logger.fast_log('LocalSpeciesPopulation.log_event: ' + '\t'.join(values),
                                             sim_time=self.time)

    def _initialize_history(self):
        """ Initialize the population history with current population """
        self._history = {}
        self._history['time'] = [self.time]  # a list of times at which population is recorded
        # the value of self._history['population'][species_id] is a list of
        # the population of species_id at the times history is recorded
        self._history['population'] = {}

    def _add_to_history(self, species_id):
        """ Add a species to the history

        Args:
            species_id (:obj:`str`): a unique species identifier.
        """
        if self._recording_history():
            population = self.read_one(self.time, species_id)
            self._history['population'][species_id] = [population]

    def _recording_history(self):
        """ Is history being recorded?

        Returns:
            True if history is being recorded.
        """
        return hasattr(self, '_history')

    def _record_history(self):
        """ Record the current population in the history

        Snapshot the current population of all species in the history. The current time
        is obtained from `self.time`.

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if the current time is not greater than the
                previous time at which the history was recorded
        """
        # todo: more comprehensible error message here
        if not self._history['time'][-1] < self.time:
            raise DynamicSpeciesPopulationError(self.time, f"time of previous _record_history() "
                                                f"({self._history['time'][-1]}) not less than current time ({self.time})")
        self._history['time'].append(self.time)
        for species_id, population in self.read(self.time, self._all_species()).items():
            self._history['population'][species_id].append(population)

    # TODO(Arthur): fix this docstring
    def report_history(self, numpy_format=False, species_type_ids=None, compartment_ids=None):
        """ Provide the time and species count history

        Args:
            numpy_format (:obj:`bool`, optional): if set, return history in a 3 dimensional np array
            species_type_ids (:obj:`list` of :obj:`str`, optional): the ids of species_types in the
                `Model` being simulated
            compartment_ids (:obj:`list` of :obj:`str`, optional): the ids of the compartments in the
                `Model` being simulated

        Returns:
            :obj:`dict`: The time and species count history. By default, return a `dict`, with
            `rv['time']` = list of time samples
            `rv['population'][species_id]` = list of counts for species_id at the time samples
            If `numpy_format` set, return a tuple containing a pair of np arrays that contain
            the time and population histories, respectively.

        Raises:
            :obj:`SpeciesPopulationError`: if the history was not recorded
            :obj:`SpeciesPopulationError`: if `numpy_format` set but `species` or `compartments` are
                not provided
        """
        if not self._recording_history():
            raise SpeciesPopulationError("history not recorded")
        if numpy_format:
            if species_type_ids is None or compartment_ids is None:
                raise SpeciesPopulationError(
                    "species_type_ids and compartment_ids must be provided if numpy_format is set")
            time_hist = np.asarray(self._history['time'])
            species_counts_hist = np.zeros((len(species_type_ids), len(compartment_ids),
                                               len(self._history['time'])))
            for species_type_index, species_type_id in list(enumerate(species_type_ids)):
                for comp_index, compartment_id in list(enumerate(compartment_ids)):
                    for time_index in range(len(self._history['time'])):
                        species_id = wc_lang.Species._gen_id(species_type_id, compartment_id)
                        if species_id in self._history['population']:
                            species_counts_hist[species_type_index, comp_index, time_index] = \
                                self._history['population'][species_id][time_index]

            return (time_hist, species_counts_hist)
        else:
            return self._history

    def history_debug(self):
        """ Provide some of the history in a string

        Provide a string containing the start and end time of the history and
        a table with the first and last population value for each species.

        Returns:
            :obj:`str`: the start and end time of he history and a
                tab-separated matrix of rows with species id, first, last population values

        Raises:
            :obj:`SpeciesPopulationError`: if the history was not recorded.
        """
        if self._recording_history():
            lines = []
            lines.append("#times\tfirst\tlast")
            lines.append("{}\t{}\t{}".format(len(self._history['time']), self._history['time'][0],
                                             self._history['time'][-1]))
            lines.append("Species\t#values\tfirst\tlast")
            for s in self._history['population'].keys():
                lines.append("{}\t{}\t{:.1f}\t{:.1f}".format(s, len(self._history['population'][s]),
                                                             self._history['population'][s][0],
                                                             self._history['population'][s][-1]))
            return '\n'.join(lines)
        else:
            raise SpeciesPopulationError("history not recorded")

    def __str__(self):
        """ Provide readable `LocalSpeciesPopulation` state

        Provide the name of this `LocalSpeciesPopulation`, the current time, and the id, population
        of each species stored by this object. Species modeled by continuous time submodels also
        have the most recent continuous time adjustment and the current population slope.

        Returns:
            :obj:`str`: a multi-line string describing this LocalSpeciesPopulation's state
        """
        state = []
        state.append('name: {}'.format(self.name))
        state.append('time: {}'.format(str(self.time)))
        state.append(DynamicSpeciesState.heading())
        for species_id in sorted(self._population.keys()):
            state.append(self._population[species_id].row())
        return '\n'.join(state)


class TempPopulationsLSP(object):
    """ A context manager for using temporary population values in a LocalSpeciesPopulation
    """
    def __init__(self, local_species_population, temporary_populations):
        """ Set populations temporarily, as specified in `temporary_populations`

        Args:
            local_species_population (:obj:`LocalSpeciesPopulation`): an existing `LocalSpeciesPopulation`
            temporary_populations (:obj:`dict` of :obj:`float`): map: species_id -> temporary_population_value;
                temporary populations for some species in `local_species_population`
        """
        self.local_species_population = local_species_population
        local_species_population.set_temp_populations(temporary_populations)
        self.species_ids = set(temporary_populations)
        self.local_species_population.temporary_mode = True

    def __enter__(self):
        return self.local_species_population

    def __exit__(self, type, value, traceback):
        """ Clear the temporary population values
        """
        self.local_species_population.clear_temp_populations(self.species_ids)
        self.local_species_population.temporary_mode = False


class MakeTestLSP(object):
    """ Make a LocalSpeciesPopulation for testing

    Because a LocalSpeciesPopulation takes about 10 lines of code to make, and they're
    widely needed for testing wc_sim, provide a configurable class that creates a test LSP.

    Attributes:
        local_species_pop (:obj:`LocalSpeciesPopulation`): the `LocalSpeciesPopulation` created
    """
    DEFAULT_NUM_SPECIES = 10
    DEFAULT_ALL_POPS = 1E6
    DEFAULT_ALL_MOL_WEIGHTS = 50

    def __init__(self, name=None, initial_population=None, molecular_weights=None,
                 initial_population_slopes=None, record_history=False, **kwargs):
        """ Initialize a `MakeTestLSP` object

        All initialized arguments are applied to the local species population being created.
        Valid keys in `kwargs` are `num_species`, `all_pops`, and `all_mol_weights`, which default to
        `MakeTestLSP.DEFAULT_NUM_SPECIES`, `MakeTestLSP.DEFAULT_ALL_POPS`, and
        `MakeTestLSP.DEFAULT_ALL_MOL_WEIGHTS`, respectively. These make a uniform population of
        num_species, with population of all_pops, and molecular weights of all_mol_weights

        Args:
            name (:obj:`str`, optional): the name of the local species population being created
            initial_population (:obj:`dict` of :obj:`float`, optional): initial population for some species;
                dict: species_id -> initial_population
            molecular_weights (:obj:`dict` of :obj:`float`, optional): map: species_id -> molecular_weight,
                provided for computing the mass of lists of species in a `LocalSpeciesPopulation`
            initial_population_slopes (:obj:`dict` of :obj:`float`, optional): map: species_id -> initial_population_slope;
                initial population slopes for all species whose populations are estimated by a continuous
                submodel. Population slopes are ignored for species not specified in initial_population.
            record_history (:obj:`bool`, optional): whether to record the history of operations
        """
        name = 'test_lsp' if name is None else name
        if initial_population is None:
            self.num_species = kwargs['num_species'] if 'num_species' in kwargs else MakeTestLSP.DEFAULT_NUM_SPECIES
            self.species_nums = list(range(0, self.num_species))
            self.all_pops = kwargs['all_pops'] if 'all_pops' in kwargs else MakeTestLSP.DEFAULT_ALL_POPS
            comp_id = 'comp_id'
            self.species_ids = list(map(lambda x: "species_{}[{}]".format(x, comp_id), self.species_nums))
            self.initial_population = dict(zip(self.species_ids, [self.all_pops]*len(self.species_nums)))
        else:
            self.initial_population = initial_population
            self.species_ids = list(initial_population.keys())

        if molecular_weights is None:
            if 'all_mol_weights' in kwargs:
                self.all_mol_weights = kwargs['all_mol_weights']
            else:
                self.all_mol_weights = MakeTestLSP.DEFAULT_ALL_MOL_WEIGHTS
            self.molecular_weights = dict(zip(self.species_ids, [self.all_mol_weights]*len(self.species_ids)))
        else:
            self.molecular_weights = molecular_weights
        random_state = RandomStateManager.instance()
        self.local_species_pop = LocalSpeciesPopulation(name, self.initial_population,
                                                        self.molecular_weights,
                                                        initial_population_slopes=initial_population_slopes,
                                                        random_state=random_state,
                                                        retain_history=record_history)


# TODO(Arthur): cover after MVP wc_sim done
class SpeciesPopSimObject(LocalSpeciesPopulation, ApplicationSimulationObject,
                          metaclass=AppSimObjAndABCMeta):  # pragma: no cover
    """ Maintain the population of a set of species in a simulation object that can be parallelized

    A whole-cell PDES must run multiple submodels in parallel. These share cell state, such as
    species populations, by accessing shared simulation objects. A SpeciesPopSimObject provides that
    functionality by wrapping a LocalSpeciesPopulation in a `SimulationObject`.
    """

    def send_initial_events(self): pass
    """ No initial events to send"""

    def get_state(self):
        return 'object state to be provided'

    def __init__(self, name, initial_population, molecular_weights, initial_population_slopes=None,
                 random_state=None, retain_history=True):
        """ Initialize a SpeciesPopSimObject object

        Initialize a SpeciesPopSimObject object. Initialize its base classes.

        Args:
            name (:obj:`str`): the name of the simulation object and local species population object.

        For remaining args and exceptions, see `__init__()` documentation for
        `de_sim.simulation_object.SimulationObject` and `wc_sim.LocalSpeciesPopulation`.
        """
        SimulationObject.__init__(self, name)
        LocalSpeciesPopulation.__init__(self, name, initial_population, molecular_weights,
                                        initial_population_slopes, random_state=random_state)

    def handle_adjust_discretely_event(self, event):
        """ Handle a simulation event

        Args:
            event (:obj:`de_sim.event.Event`): an `Event` to process
        """
        population_change = event.message.population_change
        self.adjust_discretely(self.time, population_change)

    def handle_adjust_continuously_event(self, event):
        """ Handle a simulation event

        Args:
            event (:obj:`de_sim.event.Event`): an `Event` to process

        Raises:
            :obj:`SpeciesPopulationError`: if an `AdjustPopulationByContinuousSubmodel` event acts on a
                non-existent species.
        """
        population_change = event.message.population_change
        self.adjust_continuously(self.time, population_change)

    def handle_get_population_event(self, event):
        """ Handle a simulation event

        Args:
            event (:obj:`de_sim.event.Event`): an `Event` to process

        Raises:
            :obj:`SpeciesPopulationError`: if a `GetPopulation` message requests the population of an
                unknown species.
        """
        species = event.message.species
        self.send_event(0, event.sending_object,
                        message_types.GivePopulation(self.read(self.time, species)))

    def handle_get_current_property_event(self, event):
        """ Handle a simulation event

        Args:
            event (:obj:`de_sim.event.Event`): an `Event` to process

        Raises:
            :obj:`SpeciesPopulationError`: if an `GetCurrentProperty` message requests an unknown
                property.
        """
        property_name = event.message.property_name
        if property_name == distributed_properties.MASS:
            self.send_event(0, event.sending_object,
                            message_types.GiveProperty(property_name, self.time, self.mass()))
        else:
            raise SpeciesPopulationError("Error: unknown property_name: '{}'".format(
                property_name))

    # register the event handler for each type of message received
    event_handlers = [
        # At any time instant, messages are processed in this order
        (message_types.AdjustPopulationByDiscreteSubmodel, handle_adjust_discretely_event),
        (message_types.AdjustPopulationByContinuousSubmodel, handle_adjust_continuously_event),
        (message_types.GetPopulation, handle_get_population_event),
        (message_types.GetCurrentProperty, handle_get_current_property_event)]

    # register the message types sent
    messages_sent = [message_types.GivePopulation, message_types.GiveProperty]


# todo: support multiple continuous-time algorithms by additive Superposition of their slopes
# todo: have continuous_adjustment accept population or population_slope
# todo: if continuous_adjustment is given population_slope, have it automatically incorporate
# population change predicted by the prior slope,
# that is, use: population_change = (time - self.continuous_time) * self.population_slope
class DynamicSpeciesState(object):
    """ Track the population of a single species in a multi-algorithmic model

    A species is a shared object that can be read and written by multiple submodels in a
    multi-algorithmic model. We assume that a sequence of accesses of a species instance will
    occur in non-decreasing simulation time order.

    Consider a multi-algorithmic model that contains both submodels that execute discrete-time algorithms,
    like the stochastic simulation algorithm (SSA), and submodels that execute continuous-time integration
    algorithms, like ODEs and FBA.
    Discrete-time algorithms change system state at discrete time instants. Continuous-time
    algorithms approximate species populations as continuous variables, and solve for these variables
    at time instants determined by the algorithm. At these instants, continuous-time models typically
    estimate a species' population, or the population's rate of change, or both. We assume this behavior.

    A species' state in a multi-algorithmic model may be modeled by multiple submodels that model
    reactions in which the species participates. These can be multiple discrete-time submodels and
    at most one continuous-time submodel. (If multiple continuous-time submodels were allowed to
    predict reactions that involve a species, a mechanism would be needed to reconsile conflicting
    `population_slope` values. We have not addressed that issue yet.)

    Discrete-time and continuous-time models adjust the state of a species by the methods
    `discrete_adjustment()` and `continuous_adjustment()`, respectively. These adjustments take the
    following forms,

    * `discrete_adjustment(time, population_change)`
    * `continuous_adjustment(time, population_slope)`

    where `time` is the time at which that change takes place, `population_change` is the increase or
    decrease in the species' population, and `population_slope` is the predicted future rate of
    change of the population.

    To improve the accuracy of multi-algorithmic models, we support linear interpolation of
    population predictions for species modeled by a continuous-time submodel. An interpolated
    prediction is based on the most recent continuous-time population slope prediction. Thus, we assume
    that a population modeled by a continuous model is adjusted sufficiently frequently
    that the most recent adjustment accurately estimates population slope.

    A species instance stores the most recent value of the species' population in `last_population`,
    which is initialized when the instance is created. If a species is modeled by a
    continuous-time submodel, it also stores the species' rate of change in `population_slope` and the time
    of the most recent `continuous_adjustment` in `continuous_time`. Interpolation determines the
    population prediction `p` at time `t` as::

        interpolation = 0
        if modeled_continuously:
            interpolation = (t - continuous_time)*population_slope
        p = last_population + interpolation

    This approach is completely general, and can be applied to any simulation value
    whose dynamics are predicted by a multi-algorithmic model.

    Population values returned by methods in :obj:`DynamicSpeciesState` use stochastic rounding to
    provide integer values and avoid systematic rounding bias. See more detail in `get_population`'s
    docstring.

    Attributes:
        species_name (:obj:`str`): the species' name; not logically needed, but helpful for error
            reporting, logging, debugging, etc.
        compartment_id (:obj:`str`): the species' compartment's id; optimization to avoid parsing species id
            at run-time
        random_state (:obj:`np.random.RandomState`): a shared PRNG, used to round populations
            to integers
        last_population (:obj:`float`): species population at the most recent adjustment
        modeled_continuously (:obj:`bool`): whether one of the submodels modeling the species is a
            continuous submodel; must be set at initialization
        population_slope (:obj:`float`): if a continuous submodel is modeling the species, the rate of
            change to the population provided by the most recent adjustment by a
            continuous model
        continuous_time (:obj:`float`): if a continuous submodel is modeling the species, the time of
            the most recent adjustment by the continuous model; initialized to `None` to indicate that a
            continuous adjustment has not been made yet
        last_adjustment_time (:obj:`float`): the time of the latest adjustment; used to prevent
            reads in the past
        last_read_time (:obj:`float`): the time of the latest read; used to prevent prior adjustments
        _record_history (:obj:`bool`): whether to record history of operations
        _history (:obj:`list`): history of operations
        _temp_population_value (:obj:`float`): a temporary population for a temporary computation
    """
    MINIMUM_ALLOWED_POPULATION = config_multialgorithm['minimum_allowed_population']

    # use __slots__ to save space
    __slots__ = ['species_name', 'compartment_id', 'random_state', 'last_population', 'modeled_continuously',
                 'population_slope', 'continuous_time', 'last_adjustment_time', 'last_read_time',
                 '_record_history', '_history', '_temp_population_value']

    def __init__(self, species_name, random_state, initial_population, modeled_continuously=False,
                 record_history=False):
        """ Initialize a species object, defaulting to a simulation time start time of 0

        Args:
            species_name (:obj:`str`): the species' name; not logically needed, but helpful for error
                reporting, logging, debugging, etc.
            random_state (:obj:`np.random.RandomState`): a shared PRNG
            initial_population (:obj:`int`): non-negative number; initial population of the species
            modeled_continuously (:obj:`bool`, optional): whether a continuous submodel models this species;
                default=`False`
            record_history (:obj:`bool`, optional): whether to record a history of all operations;
                default=`False`
        """
        self.species_name = species_name
        _, self.compartment_id = ModelUtilities.parse_species_id(species_name)
        assert 0 <= initial_population, f"DynamicSpeciesState '{species_name}': population should be >= 0"
        # if a population is not modeled continuously then it must be a non-negative integer
        assert modeled_continuously or float(initial_population).is_integer(), \
            (f"DynamicSpeciesState '{species_name}': discrete population must be a "
             f"non-negative integer, but {initial_population} isn't")
        self.random_state = random_state
        self.last_population = initial_population
        self.modeled_continuously = modeled_continuously
        if modeled_continuously:
            self.population_slope = None
            # continuous_time is None indicates that a continuous_adjustment() has not been made yet
            self.continuous_time = None
        self.last_adjustment_time = -float('inf')
        self.last_read_time = -float('inf')
        self._record_history = record_history
        if record_history:
            self._history = []
            self._record_operation_in_hist(0, 'initialize', initial_population)
        self._temp_population_value = None

    def _update_last_adjustment_time(self, adjustment_time):
        """ Advance the last adjustment time to `adjustment_time`

        Args:
            adjustment_time (:obj:`float`): the time at which the population is being adjusted
        """
        self.last_adjustment_time = max(self.last_adjustment_time, adjustment_time)

    def _update_last_read_time(self, read_time):
        """ Advance the last read time to `read_time`

        Args:
            read_time (:obj:`float`): the time at which the population is being read
        """
        self.last_read_time = max(self.last_read_time, read_time)

    def _validate_adjustment_time(self, adjustment_time, method):
        """ Raise an exception if `adjustment_time` is too early

        Args:
            adjustment_time (:obj:`float`): the time at which the population is being adjusted
            method (:obj:`str`): name of the method making the adjustment

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if `adjustment_time` is earlier than latest prior adjustment,
                 or if adjustment_time is earlier than latest prior read
        """
        if adjustment_time < self.last_adjustment_time:
            raise DynamicSpeciesPopulationError(adjustment_time,
                "{}(): {}: adjustment_time is earlier than latest prior adjustment: "
                "{:.2f} < {:.2f}".format(method, self.species_name, adjustment_time, self.last_adjustment_time))
        if adjustment_time < self.last_read_time:
            raise DynamicSpeciesPopulationError(adjustment_time,
                "{}(): {}: adjustment_time is earlier than latest prior read: "
                "{:.2f} < {:.2f}".format(method, self.species_name, adjustment_time, self.last_read_time))

    def _validate_read_time(self, read_time, method):
        """ Raise an exception if `read_time` is too early

        Args:
            read_time (:obj:`float`): the time at which the population is being read
            method (:obj:`str`): name of the method making the read

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if `read_time` is earlier than latest prior adjustment
        """
        if read_time < self.last_adjustment_time:
            raise DynamicSpeciesPopulationError(read_time,
                "{}(): {}: read_time is earlier than latest prior adjustment: "
                "{:.2f} < {:.2f}".format(method, self.species_name, read_time, self.last_adjustment_time))

    class Operation(Enum):
        """ Types of operations on DynamicSpeciesState, for use in history """
        initialize = 1
        discrete_adjustment = 2
        continuous_adjustment = 3

    HistoryRecord = namedtuple('HistoryRecord', 'time, operation, argument')
    HistoryRecord.__doc__ += ': entry in a DynamicSpeciesState history'
    HistoryRecord.time.__doc__ = 'simulation time of the operation'
    HistoryRecord.operation.__doc__ = 'type of the operation'
    HistoryRecord.argument.__doc__ = "operation's argument: initialize: population; "\
        "discrete_adjustment: population_change; continuous_adjustment: population_slope"

    def _record_operation_in_hist(self, time, method, argument):
        """ Record a history entry

        Args:
            time (:obj:`float`): simulation time of the operation
            method (:obj:`str`): the operation type
            argument (:obj:`float`): the operation's argument
        """
        if self._record_history:
            operation = self.Operation[method]
            self._history.append(self.HistoryRecord(time, operation, argument))

    def discrete_adjustment(self, time, population_change):
        """ Make a discrete adjustment of the species' population

        A submodel running a discrete-time integration algorithm, such as the stochastic simulation
        algorithm (SSA), must use this method to adjust its species' populations.

        Args:
            time (:obj:`float`): the simulation time at which the predicted change occurs; this time is
                used by `get_population` to interpolate continuous-time predictions between integrations
            population_change (:obj:`float`): the modeled increase or decrease in the species' population

        Returns:
            :obj:`int`: an integer approximation of the species' adjusted population

        Raises:
            :obj:`DynamicNegativePopulationError`: if the predicted population at `time` is negative or
                if decreasing the population by `population_change` would make the population negative
        """
        # merge with other branch
        assert float(population_change).is_integer(), \
            "DynamicSpeciesState '{}': population_change must be an integer, but {} isn't".format(
                self.species_name, population_change)
        self._validate_adjustment_time(time, 'discrete_adjustment')
        current_population = self.get_population(time)
        if current_population + population_change < self.MINIMUM_ALLOWED_POPULATION:
            raise DynamicNegativePopulationError(time, 'discrete_adjustment', self.species_name,
                                                 self.last_population, population_change)
        self.last_population += population_change
        self._update_last_adjustment_time(time)
        self._record_operation_in_hist(time, 'discrete_adjustment', population_change)
        return self.get_population(time)

    def continuous_adjustment(self, time, population_slope):
        """ A continuous-time submodel adjusts the species' state

        A continuous-time submodel, such as an ordinary differential equation (ODE) or a dynamic flux
        balance analysis (FBA) model, uses this method to adjust the species' state. Each
        integration of a continuous-time model must predict a species' population change and the
        population's short-term future rate of change, i.e., its `population_slope`. Further, since an
        integration of a continuous-time model at the current time must depend on this species'
        population just before the integration, we assume that `population_change` incorporates
        population changes predicted by the `population_slope` provided by the previous
        `continuous_adjustment` call.

        Args:
            time (:obj:`float`): the simulation time at which the predicted change occurs; this time is
                used by `get_population` to interpolate continuous-time predictions between
                integrations.
            population_slope (:obj:`float` or :obj:`int`): the predicted rate of change of the
                species at the provided time

        Returns:
            :obj:`int`: the species' adjusted population, rounded to an integer

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if an initial population slope was not provided, or
                if `time` is not greater than the time of the most recent `continuous_adjustment` call
            :obj:`DynamicNegativePopulationError`: if updating the population based on the previous
                `population_slope` makes the population go negative
        """
        assert isinstance(population_slope, (float, int)), \
            (f"continuous_adjustment of species '{self.species_name}': population_slope "
             f"(type=='{type(population_slope).__name__}') must be an int or float")
        if not self.modeled_continuously:
            raise DynamicSpeciesPopulationError(time,
                f"continuous_adjustment(): DynamicSpeciesState for '{self.species_name}' needs "
                f"self.modeled_continuously==True")
        self._validate_adjustment_time(time, 'continuous_adjustment')
        # self.continuous_time is None until the first continuous_adjustment()
        if self.continuous_time is not None:
            if self.last_population + \
                self.population_slope * (time - self.continuous_time) < self.MINIMUM_ALLOWED_POPULATION:
                raise DynamicNegativePopulationError(time, 'continuous_adjustment', self.species_name,
                                                     self.last_population,
                                                     self.population_slope * (time - self.continuous_time),
                                                     delta_time=time - self.continuous_time)
            # add the population change since the last continuous_adjustment
            self.last_population += self.population_slope * (time - self.continuous_time)
        self.continuous_time = time
        self.population_slope = population_slope
        self._update_last_adjustment_time(time)
        self._record_operation_in_hist(time, 'continuous_adjustment', population_slope)
        return self.get_population(time)

    def set_temp_population_value(self, population):
        """ Set a temporary population

        Save a temporary population for a temporary, exploratory computation, like the right-hand-side
        function used to solve ODEs.
        The time associated with a temporary population value is not provided or needed.
        These values aren't saved in the history.

        Args:
            population (:obj:`float`): the temporary population
        """
        self._temp_population_value = population

    def get_temp_population_value(self):
        """ Get the temporary population

        Provide a temporary population that has been stored for temporary, exploratory computations,
        like the right-hand-side function used to solve ODEs.

        Returns:
            :obj:`obj`: if the temporary population is set, the species' population as a :obj:`float`,
                otherwise :obj:`None`
        """
        return self._temp_population_value

    def clear_temp_population_value(self):
        """ Clear a temporary population value
        """
        self._temp_population_value = None

    def get_population(self, time, interpolate=None, round=True, temporary_mode=False):
        """ Provide the species' population at time `time`

        If one of the submodel(s) predicting the species' population is a continuous-time model,
        then use the species' last `population_slope` to interpolate the current population, as
        described in the class documentation.

        Clearly, species populations in biological systems are non-negative integers. However,
        continuous-time models approximate populations with continuous representations, and
        therefore predict real, non-integral, populations. But discrete-time models like SSA
        do not naturally handle non-integral copy numbers.

        We resolve this conflict by storing real valued populations within a species, but
        providing integral population predictions if `round` is `True`. To aovid the bias that would arise by
        always using `floor()` or `ceiling()` to convert a float to an integer, population predictions
        are stochastically rounded before being returned by `get_population`. *This means
        that a sequence of calls to `get_population` with `round=True`
        may **NOT** return a sequence of equal population values.*

        Args:
            time (:obj:`float`): the current simulation time
            round (:obj:`bool`, optional): if `round` then round the population to an integer
            interpolate (:obj:`bool`, optional): if not `None` then control interpolation;
                otherwise it's controlled by the 'interpolate' config variable
            temporary_mode (:obj:`bool`, optional): whether the calling `LocalSpeciesPopulation` is in
                temporary mode; if set, do not update `last_read_time`

        Returns:
            :obj:`int`: if `round`, an integer approximation of the species' population, otherwise
                the floating population

        Raises:
            :obj:`DynamicSpeciesPopulationError`: if `time` is earlier than the time of a previous continuous
                or discrete adjustment
            :obj:`DynamicNegativePopulationError`: if interpolation predicts a negative population
        """

        self._validate_read_time(time, 'get_population')
        if not self.modeled_continuously:
            if not temporary_mode:
                self._update_last_read_time(time)
            # self.last_population does not need to be rounded as discrete submodels only change
            # populations by integral amounts
            return self.last_population
        else:
            interpolation = 0.
            if self.continuous_time is not None:
                if interpolate is None:
                    interpolate = config_multialgorithm['interpolate']
                if interpolate:
                    interpolation = (time - self.continuous_time) * self.population_slope
                    if self.last_population + interpolation < self.MINIMUM_ALLOWED_POPULATION:
                        raise DynamicNegativePopulationError(time, 'get_population', self.species_name,
                            self.last_population, interpolation, time - self.continuous_time)
            float_copy_number = self.last_population + interpolation
            if not temporary_mode:
                self._update_last_read_time(time)
            # if round then round the return value to an integer, otherwise don't
            if round:
                # this cannot return a negative number
                return self.random_state.round(float_copy_number)
            return float_copy_number

    def get_history(self):
        """ Obtain this `DynamicSpeciesState`'s history

        Returns:
            :obj:`list`: a list of `HistoryRecord`s, ordered by time

        Raises:
            :obj:`SpeciesPopulationError`: if the history wasn't recorded
        """
        if self._record_history:
            return self._history
        else:
            raise SpeciesPopulationError('history not recorded')

    def __str__(self):
        if self.modeled_continuously:
            return "species_name: {}; last_population: {}; continuous_time: {}; population_slope: {}".format(
                self.species_name, self.last_population, self.continuous_time, self.population_slope)
        else:
            return f"species_name: {self.species_name}; last_population: {self.last_population}"

    @staticmethod
    def heading():
        """ Return a heading for a tab-separated table of species data """
        return '\t'.join('species_name last_population continuous_time population_slope'.split())

    def row(self):
        """ Return a row for a tab-separated table of species data """
        if self.modeled_continuously:
            if self.continuous_time is None:
                return "{}\t{:.2f}".format(self.species_name, self.last_population)
            else:
                return "{}\t{:.2f}\t{:.2f}\t{:.2f}".format(self.species_name, self.last_population,
                self.continuous_time, self.population_slope)
        else:
            return "{}\t{:.2f}".format(self.species_name, self.last_population)
