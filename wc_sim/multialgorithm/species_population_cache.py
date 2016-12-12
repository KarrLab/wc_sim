'''Cache the population of species whose primary stores are remote population stores.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-08
:Copyright: 2016, Karr Lab
:License: MIT
'''

from six import iteritems

from wc_sim.multialgorithm.access_species_populations import AccessSpeciesPopulations as ASP
from wc_sim.multialgorithm.access_species_populations import LOCAL_POP_STORE

from wc_utils.config.core import ConfigManager
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
config_multialgorithm = \
    ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']
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
            ValueError: if the species are stored in the ASP's local store; they should not be
                cached.
            ValueError: if, for any specie, `time` is not greater than its previous cache time.
        '''
        # raise exception if the species are stored in the ASP's local store
        store_name_map = self.access_species_populations.locate_species(populations.keys())
        if LOCAL_POP_STORE in store_name_map:
            raise ValueError("cache_population: some species are stored in the "
                "AccessSpeciesPopulations's local store: {}.".format(list(store_name_map[LOCAL_POP_STORE])))
        # TODO(Arthur): could raise an exception if the species are not stored in the ASP's remote stores
        for specie_id,population in iteritems(populations):
            # raise exception if the time of this cache is not greater than the previous cache time
            if specie_id in self._cache and time <= self._cache[specie_id][0]:
                raise ValueError("cache_population: caching an earlier population: specie_id: {}; "
                    "current time: {} <= previous time {}.".format(specie_id, time,
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
            ValueError: if the species are stored in the ASP's local store, which means that they
                should not be cached.
            ValueError: if `time` is not greater than a specie's previous cache time.
        '''
        if specie_id not in self._cache:
            raise ValueError("SpeciesPopulationCache.read_one: specie '{}' not in cache.".format(
                specie_id))
        if self._cache[specie_id][0] + epsilon < time:
            raise ValueError("SpeciesPopulationCache.read_one: cache age of {} too big for read at "
                "time {} of specie '{}'.".format(time-self._cache[specie_id][0], time, specie_id))
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
            ValueError: if any of the species are not stored in the cache.
            ValueError: if any of the species were cached at a time that differs from `time`.
        '''
        missing = list(filter(lambda specie: specie not in self._cache, species_ids))
        if missing:
            raise ValueError("SpeciesPopulationCache.read: species {} not in cache.".format(
                str(missing)))
        mistimed = list(filter(lambda s_id: self._cache[s_id][0] + epsilon < time, species_ids))
        if mistimed:
            raise ValueError("SpeciesPopulationCache.read: species {} not reading "
                "recently cached value(s).".format(str(mistimed)))
        return {specie:self._cache[specie][1] for specie in species_ids}

