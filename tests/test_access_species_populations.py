'''Test test_access_species_populations.py.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-04
:Copyright: 2016, Karr Lab
:License: MIT
'''

import os, unittest
import string
from wc_sim.multialgorithm.access_species_populations import AccessSpeciesPopulations as ASP

def store_i(i):
    return "store_{}".format(i)

def specie_l(l):
    return "specie_{}".format(l)

remote_pop_stores = {store_i(i):None for i in range(1, 4)}
species_ids = [specie_l(l) for l in list(string.ascii_lowercase)[0:5]]

class TestAccessSpeciesPopulations(unittest.TestCase):

    def setUp(self):
        self.an_ASP = ASP(remote_pop_stores)

    def test_dict_filtering(self):
        d={'a':1, 'b':2, 'c':3}
        self.assertEqual(ASP.filtered_dict(d, []), {})
        self.assertEqual(ASP.filtered_dict(d, ['a']), {'a':1})
        self.assertEqual(ASP.filtered_dict(d, ['a','d','a','b','c']), d)

        self.assertEqual({(k,v) for k,v in ASP.filtered_iteritems(d, ['a','b','d'])},
            {('a', 1), ('b', 2)})

    def test_species_locations(self):

        self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        map = dict(zip(species_ids[:2], [store_i(1)]*2))
        map = dict.fromkeys(species_ids[:2], store_i(1))
        self.assertEqual(self.an_ASP.species_locations, map)

        self.an_ASP.add_species_locations(store_i(2), species_ids[2:])
        map.update(dict(zip(species_ids[2:], [store_i(2)]*3)))
        self.assertEqual(self.an_ASP.species_locations, map)

        locs = self.an_ASP.locate_species(species_ids[1:4])
        self.assertEqual(locs[store_i(1)], {'specie_b'})
        self.assertEqual(locs[store_i(2)], {'specie_c', 'specie_d'})

        self.an_ASP.del_species_locations([specie_l('b')])
        del map[specie_l('b')]
        self.assertEqual(self.an_ASP.species_locations, map)
        self.an_ASP.del_species_locations(species_ids, force=True)
        self.assertEqual(self.an_ASP.species_locations, {})

    def test_species_locations_exceptions(self):
        with self.assertRaises(ValueError) as cm:
            self.an_ASP.add_species_locations('no_such_store', species_ids[:2])
        self.assertIn("'no_such_store' not a known population store", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.an_ASP.add_species_locations('no_such_store', species_ids[:2])
        self.assertIn("'no_such_store' not a known population store", str(cm.exception))

        self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        with self.assertRaises(ValueError) as cm:
            self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        self.assertIn("species ['specie_a', 'specie_b'] already have assigned locations.", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.an_ASP.del_species_locations([specie_l('d'), specie_l('c')])
        self.assertIn("species ['specie_c', 'specie_d'] are not in the location map", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.an_ASP.locate_species([specie_l('d'), specie_l('c')])
        self.assertIn("species ['specie_c', 'specie_d'] are not in the location map", str(cm.exception))
