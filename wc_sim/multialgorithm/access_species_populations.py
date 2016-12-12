'''The interface between a submodel and the components that store the species populations it models.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-04
:Copyright: 2016, Karr Lab
:License: MIT
'''

from collections import defaultdict
from six import iteritems
from wc_utils.util.dict import DictUtil
import numpy

from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.abc_for_species_pop_access import AccessSpeciesPopulationInterface
from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation
from wc_sim.multialgorithm.species_pop_sim_object import SpeciesPopSimObject

from wc_utils.config.core import ConfigManager
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
config_multialgorithm = \
    ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']
epsilon = config_multialgorithm['epsilon']

LOCAL_POP_STORE = 'LOCAL_POP_STORE'  # the name of the local population store


class AccessSpeciesPopulations(AccessSpeciesPopulationInterface):
    '''Interface a submodel with the components that store the species populations it models.

    Each submodel is a distinct simulation object. In the current population-based model,
    species are represented by their populations. (A hybrid instance-population model would change
    that.) Each submodel accesses a subset of the species in a model. WLG we assume that a submodel's
    species can be partitioned into those that are accessed by ONLY the submodel and those that it
    shares with other submodels. These are respectively stored in a local LocalSpeciesPopulation
    which is private to this submodel, and a set of SpeciesPopSimObjects which are shared with other
    submodels. LocalSpeciesPopulation objects are accessed via local memory operations whereas
    SpeciesPopSimObjects, which are distinct simulation objects, are accessed via simulation event
    messages.

    AccessSpeciesPopulations enables a submodel to access all of the species it access through a
    single convenient R/W interface. The submodel simply indicates the operation type and the
    specie(s) being used, and this object maps the operation to the correct entity storing the
    specie(s).

    Attributes:
        local_pop_store (:obj:`LocalSpeciesPopulation`): a store of local species, and a cache of
            populations for remotely stored species; all populations reported in GivePopulation
            messages are written to the local_pop_store.
        remote_pop_stores (:obj:`dict` of `SpeciesPopSimObject`): a map from store name to a system
            identifier for the remote population store(s) that the local submodel uses. For a shared
            memory implementation system identifiers can be object references; for a distributed
            implementation they must be network-wide object identifiers.
        species_locations (:obj:`dict` of `str`): a map indicating the store for each specie used
            by the local submodel.
        submodel (:obj:`Submodel`): the submodel which is using this AccessSpeciesPopulations
        species_population_cache (:obj:`SpeciesPopulationCache`): an object for caching populations
            from the remote_pop_stores SpeciesPopSimObjects.
    '''

    def __init__(self, local_pop_store, remote_pop_stores):
        '''Initialize an AccessSpeciesPopulations object.

        TBD ...

        Args:
            model (:obj:`Model`): the `Model` containing this LocalSpeciesPopulation.

        Raises:
        TBD ...
            AssertionError: if the population cannot be initialized.
        '''
        self.local_pop_store = local_pop_store
        if LOCAL_POP_STORE in remote_pop_stores:
            raise ValueError("AccessSpeciesPopulations.__init__: {} not a valid "
                "remote_pop_store name".format(LOCAL_POP_STORE))
        self.remote_pop_stores = remote_pop_stores
        self.species_locations = {}

    def set_submodel(self, submodel):
        self.submodel = submodel

    def set_species_population_cache(self, species_population_cache):
        self.species_population_cache = species_population_cache

    def add_species_locations(self, store_name, specie_ids, replace=False):
        '''Add species locations to the species location map.

        Record that the species listed in `species_ids` are stored by the species population store
        identified by `store_name`. To replace existing location map values without raising an
        exception, set `replace` to True.

        Args:
            store_name (str): the globally unique name of a species population store.
            specie_ids (:obj:`list` of specie_ids): a list of species ids.

        Raises:
            ValueError: if store `store_name` is unknown.
            ValueError: if `replace` is False and any specie_id in `specie_ids` is already mapped
                to a different store than `store_name`.
        '''
        if not store_name in self.remote_pop_stores.keys() and store_name != LOCAL_POP_STORE:
            raise ValueError("add_species_locations: '{}' not a known population store.".format(
                store_name))
        if replace:
            for specie_id in specie_ids:
                self.species_locations[specie_id] = store_name
        else:
            assigned = list(filter(lambda s: s in self.species_locations.keys(), specie_ids))
            if assigned:
                raise ValueError("add_species_locations: species {} already have assigned locations.".format(
                    sorted(assigned)))
            for specie_id in specie_ids:
                self.species_locations[specie_id] = store_name

    def del_species_locations(self, specie_ids, force=False):
        '''Delete entries from the species location map.

        Remove species location mappings for the species in `specie_ids`. To avoid raising an
        exception when a specie is not in the location map, set `force` to `True`.

        Args:
            specie_ids (:obj:`list` of specie_ids): a list of species ids.

        Raises:
            ValueError: if `force` is False and any specie_id in `specie_ids` is not in the
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
                raise ValueError("del_species_locations: species {} are not in the location "
                    "map.".format(sorted(unassigned)))
            for specie_id in specie_ids:
                del self.species_locations[specie_id]

    def locate_species(self, specie_ids):
        '''Locate the components that store a set of species.

        Given a set of `species_ids`, partition them into the components that store their
        populations. This method is widely used by code that accesses species. It returns a
        dictionary that maps from store name to the ids of species whose populations are modeled
        by the store.

        `LOCAL_POP_STORE` represents a special store, the local
        `wc_sim.multialgorithm.local_species_population.LocalSpeciesPopulation` instance. Each other
        store is named by the name of a remote
        `from wc_sim.multialgorithm.species_pop_sim_object.SpeciesPopSimObject` instance.

        Args:
            specie_ids (:obj:`list` of specie_ids): a list of species ids.

        Returns:
            dict: a map from store_name -> (:obj:`set` of specie_ids) whose populations are stored
                by component store_name.

        Raises:
            ValueError: if a store cannot be found for a specie_id in `specie_ids`.
        '''
        unknown = list(filter(lambda s: s not in self.species_locations.keys(), specie_ids))
        if unknown:
            raise ValueError("locate_species: species {} are not in the location map.".format(
                sorted(unknown)))
        inverse_loc_map = defaultdict(set)
        for specie_id in specie_ids:
            store = self.species_locations[specie_id]
            inverse_loc_map[store].add(specie_id)
        return inverse_loc_map

    def read_one(self, time, specie_id):
        '''Read the predicted population of a specie at a particular time.

        If the specie is stored in the local_pop_store, obtain the value there. Otherwise obtain
        the value from the species_population_cache. The value will be in the cache because if the
        specie's primary store is a remote_pop_store, then a remote request via a GetPopulation
        query and a GivePopulation response will have already cached the population in the
        local_pop_store.

        Args:
            time (float): the time at which the population should be predicted.
            specie_id (str): identifier of the specie to read.

        Returns:
            float: the predicted population of `specie_id` at time `time`.

        Raises:
            ValueError: if `specie_id` is an unknown specie.
        '''
        if specie_id not in self.species_locations:
            raise ValueError("read_one: specie '{}' not in the location map.".format(specie_id))
        store = self.species_locations[specie_id]
        if store==LOCAL_POP_STORE:
            return self.local_pop_store.read_one(time, specie_id)
        else:
            # TODO(Arthur): convert print() to log message
            # print('submodel {} reading {} from cache at {:.2f}'.format(self.submodel.name,
            #   specie_id, time))
            return self.species_population_cache.read_one(time, specie_id)

    def read(self, time, species_ids):
        '''Obtain the predicted population of a list of species at a particular time.

        Access the species from the local_pop_store. If some of the species' primary stores are
        remote_pop_stores, then remote requests via a prefetch() operation will have already cached
        the population in the local_pop_store.

        Args:
            time (float): the time at which the population should be obtained.
            species_ids (set): identifiers of the species to whose populations are obtained.

        Returns:
            species counts: dict: species_id -> copy_number; the predicted copy number of all
            requested species at time `time`.

        Raises:
            ValueError: if a store cannot be found for a specie_id in `specie_ids`.
            ValueError: if any of the species were cached at a time that differs from `time`.
        '''
        local_species = self.locate_species(species_ids)[LOCAL_POP_STORE]
        remote_species = set(species_ids) - set(local_species)

        local_pops = self.local_pop_store.read(time, local_species)
        cached_pops = self.species_population_cache.read(time, remote_species)

        cached_pops.update(local_pops)
        return cached_pops

    def adjust_discretely(self, time, adjustments):
        '''A discrete submodel adjusts the population of a set of species at a particular time.

        Distribute the adjustments among the population stores managed by this object.
        Iterate through the components that store the population of species listed in `adjustments`,
        updating the local population store immediately and sending AdjustPopulationByDiscreteModel
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
        '''A continuous submodel adjusts the population of a set of species at a particular time.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of `tuple`): map: specie_ids -> (population_adjustment, flux);
                adjustments to be made to some species populations.

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
            raise ValueError('prefetch: {} provided, but delay must be non-negative.'.format(delay))
        stores=[]
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

        TBD
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

# TODO(Arthur): replace ValueError with new Error type
