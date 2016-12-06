'''The interface between a submodel and the components that store the species populations it models.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-04
:Copyright: 2016, Karr Lab
:License: MIT
'''

from six import iteritems
from collections import defaultdict

from wc_sim.multialgorithm.abc_for_species_pop_access import AccessSpeciesPopulationInterface
from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation
from wc_sim.multialgorithm.species_pop_sim_object import SpeciesPopSimObject

_LOCAL_POP_STORE='_LOCAL_POP_STORE'  # the name of the local population store

class AccessSpeciesPopulations(AccessSpeciesPopulationInterface):
    '''Interface a submodel with the components that store the species populations it models.

    details ...

    Attributes:
        submodel (:obj:`Submodel`): the submodel which is using this AccessSpeciesPopulations
        local_pop_store (:obj:`LocalSpeciesPopulation`): a store of local species, and a cache of
            populations for remotely stored species; all GivePopulation results are written to the
            local_pop_store.
        remote_pop_stores (:obj:`dict` of `SpeciesPopSimObject`): a map from store name to a system
            identifier for the remote population store(s) that the local submodel uses. For a shared
            memory implementation system identifiers can be object references; for a distributed
            implementation they must be network-wide object identifiers.
        species_location (:obj:`dict` of `str`): a map indicating the store for each specie used
            by the local submodel.
    '''
    def __init__(self, remote_pop_stores):
        '''Initialize an AccessSpeciesPopulations object.

        TBD ...

        Args:
        TBD ...
            model (:obj:`Model`): the `Model` containing this LocalSpeciesPopulation.

        Raises:
        TBD ...
            AssertionError: if the population cannot be initialized.
        '''
        self.species_locations = {}
        self.remote_pop_stores = remote_pop_stores

    def add_species_locations(self, store_name, specie_ids, replace=False):
        '''Add species locations to the species location map.

        Record that the species listed in `species_ids` are stored by the species population store
        identified by `store_name`. To replace existing location map values without raising an
        exception, set `replace` to True.

        Args:
            store_name (str): the name of a species population store.
            specie_ids (:obj:`list` of specie_ids): a list of species ids.

        Raises:
            ValueError: if store `store_name` is unknown.
            ValueError: if `replace` is False and any specie_id in `specie_ids` is already mapped
                to a different store than `store_name`.
        '''
        if not store_name in self.remote_pop_stores.keys():
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
                raise ValueError("del_species_locations: species {} are not in the location map.".format(
                    sorted(unassigned)))
            for specie_id in specie_ids:
                del self.species_locations[specie_id]


    def locate_species(self, specie_ids):
        '''Locate the components that store a set of species.

        Given a set of `species_ids`, partition them into the components that store their
        populations. This method is widely used by code that accesses species. It returns a
        dictionary that maps from store name to the ids of species whose populations are modeled
        by the store.

        `_LOCAL_POP_STORE` represents a special store, the local
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

        Access the specie from the local_pop_store. If the specie's primary store is a
        remote_pop_store, then a remote request via a GetPopulation query and a GivePopulation
        response will have already cached the population in the local_pop_store.

        TBD ... for Args, Exceptions, etc., see the ABC.
        '''
        return self.local_pop_store.read_one(time, specie_id)


    def read(self, time, species):
        '''Read the predicted population of a list of species at a particular time.

        Access the specie from the local_pop_store. If the specie's primary store is a
        remote_pop_store, then a remote request via a GetPopulation query and a GivePopulation
        response will have already cached the population in the local_pop_store.

        TBD ... for Args, Exceptions, etc., see the ABC.
        '''
        return self.local_pop_store.read(time, species)

    # todo: move to utils
    @staticmethod
    def filtered_dict(d, filter_keys):
        '''Create a new dict from `d`, with keys filtered by `filter_keys`.

        Returns:
            dict: a new dict containing the entries in `d` whose keys are in `filter_keys`.
        '''
        return {k:v for (k,v) in iteritems(d) if k in filter_keys}

    @staticmethod
    def filtered_iteritems(d, filter_keys):
        '''A generator that filters a dict's iteritems to keys in `filter_keys`.

        Yields:
            tuple: (key, value) tuples from `d` whose keys are in `filter_keys`.
        '''
        for key, val in iteritems(d):
            if key not in filter_keys:
                continue
            yield key, val

    def adjust_discretely(self, time, adjustments):
        '''A discrete submodel adjusts the population of a set of species at a particular time.

        Iterate through the components that store the population of species listed in `adjustments`,
        and send each component an AdjustPopulationByDiscreteModel message. Since these messages are
        asynchronous, this method returns as soon as they can be sent.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of float): map: specie_ids -> population_adjustment; adjustments
                to be made to some species populations.

        TBD
        '''
        for (store,species) in self.locate_species(adjustments.keys()):
            store_adjustments = AccessSpeciesPopulations.filtered_dict(adjustments, species)
            if store==LOCAL_POP_STORE:
                self.local_pop_store.adjust_continuously(time, store_adjustments)
            else:
                self.submodel.send_event(time-self.submodel.time,
                    self.remote_pop_stores[store],
                update_time, species_pop_sim_obj, update_message, event_body=msg_body)

    def adjust_continuously(self, time, adjustments):
        '''A continuous submodel adjusts the population of a set of species at a particular time.

        Args:
            time (float): the time at which the population is being adjusted.
            adjustments (:obj:`dict` of `tuple`): map: specie_ids -> (population_adjustment, flux);
                adjustments to be made to some species populations.

        TBD
        '''
