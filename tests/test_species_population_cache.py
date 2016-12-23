'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-08
:Copyright: 2016, Karr Lab
:License: MIT
'''
import unittest

from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation
from wc_sim.multialgorithm.species_population_cache import SpeciesPopulationCache
from wc_sim.multialgorithm.access_species_populations import AccessSpeciesPopulations as ASP
from wc_sim.multialgorithm.access_species_populations import LOCAL_POP_STORE
from wc_sim.multialgorithm.multialgorithm_errors import SpeciesPopulationError

def store_i(i):
    return "store_{}".format(i)

class TestSpeciesPopulationCache(unittest.TestCase):

    def setUp(self):
        species_nums = range(1, 5)
        self.species_ids = list(map(lambda x: "specie_{}".format(x), species_nums))
        self.init_populations = dict(zip(self.species_ids, [0]*len(self.species_ids)))
        local_species_population = LocalSpeciesPopulation(None, 'name', self.init_populations)

        remote_pop_stores = {store_i(i):None for i in range(1, 4)}
        self.an_ASP = ASP(local_species_population, remote_pop_stores)
        self.an_ASP.add_species_locations(store_i(1), self.species_ids)
        self.an_ASP.add_species_locations(LOCAL_POP_STORE, ["specie_0"])
        self.species_population_cache = SpeciesPopulationCache(self.an_ASP)
        self.an_ASP.set_species_population_cache(self.species_population_cache)

    def test_species_population_cache(self):
        populations = [x*10 for x in range(len(self.species_ids))]
        population_dict = dict(zip(self.species_ids, populations))
        self.species_population_cache.cache_population(1, population_dict)
        s = self.species_ids[0]
        self.assertEqual(self.species_population_cache.read_one(1, s), population_dict[s])
        self.assertEqual(self.species_population_cache.read(1, self.species_ids),
            population_dict)

    def test_species_population_cache_exceptions(self):
        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.cache_population(1, {"specie_0": 3})
        self.assertIn("some species are stored in the AccessSpeciesPopulations's local store: "
            "['specie_0'].", str(context.exception) )

        self.species_population_cache.cache_population(0, {"specie_1": 3})
        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.cache_population(-1, {"specie_1": 3})
        self.assertIn( "cache_population: caching an earlier population: specie_id: specie_1; "
            "current time: -1 <= previous time 0.", str(context.exception) )

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read_one(1, 'specie_none')
        self.assertIn( "SpeciesPopulationCache.read_one: specie 'specie_none' not in cache.",
            str(context.exception) )
    
        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read_one(1, 'specie_1')
        self.assertIn( "cache age of 1 too big for read at time 1 of specie 'specie_1'",
            str(context.exception) )
    
        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read(0, ['specie_none'])
        self.assertIn( "SpeciesPopulationCache.read: species ['specie_none'] not in cache.",
            str(context.exception) )

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read(1, ['specie_1'])
        self.assertIn( ".read: species ['specie_1'] not reading recently cached value(s)",
            str(context.exception) )
