'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-10
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import unittest, os
from argparse import Namespace

from wc_lang.io import Reader
from wc_lang.core import RateLawEquation, RateLaw, Reaction, Submodel, Species
from obj_model import utils
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.model_utilities import ModelUtilities


class TestModelUtilities(unittest.TestCase):

    def get_submodel(self, id_val):
        return Submodel.objects.get_one(id=id_val)

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # read a model
        Submodel.objects.reset()
        self.model = Reader().run(self.MODEL_FILENAME)

    def test_find_private_species(self):
        # since set() operations are being used, this test does not ensure that the methods being
        # tested are deterministic and repeatable; repeatability should be tested by running the
        # code under different circumstances and ensuring identical output
        private_species = ModelUtilities.find_private_species(self.model)

        mod1_expected_species_ids = ['specie_1[c]', 'specie_1[e]', 'specie_2[e]']
        mod1_expected_species = Species.get(mod1_expected_species_ids, self.model.get_species())
        self.assertEqual(set(private_species[self.get_submodel('submodel_1')]),
            set(mod1_expected_species))

        mod2_expected_species_ids = ['specie_4[c]', 'specie_5[c]', 'specie_6[c]']
        mod2_expected_species = Species.get(mod2_expected_species_ids, self.model.get_species())
        self.assertEqual(set(private_species[self.get_submodel('submodel_2')]),
            set(mod2_expected_species))

        private_species = ModelUtilities.find_private_species(self.model, return_ids=True)
        self.assertEqual(set(private_species['submodel_1']), set(mod1_expected_species_ids))
        self.assertEqual(set(private_species['submodel_2']), set(mod2_expected_species_ids))

    @unittest.skip("to be fixed")
    def test_find_shared_species(self):
        self.assertEqual(set(ModelUtilities.find_shared_species(self.model)),
            set(Species.get(['specie_2[c]', 'specie_3[c]'], self.model.get_species())))

        self.assertEqual(set(ModelUtilities.find_shared_species(self.model, return_ids=True)),
            set(['specie_2[c]', 'specie_3[c]']))

    def test_get_initial_specie_concentrations(self):
        initial_specie_concentrations = ModelUtilities.initial_specie_concentrations(self.model)
        some_specie_concentrations = {
            'specie_2[e]':2.0E-4,
            'specie_2[c]':5.0E-4 }
        for k in some_specie_concentrations.keys():
            self.assertEqual(initial_specie_concentrations[k], some_specie_concentrations[k])

    def test_parse_specie_id(self):
        self.assertEqual(ModelUtilities.parse_specie_id('good_id[good_compt]'), ('good_id', 'good_compt'))
        with self.assertRaises(ValueError) as context:
            ModelUtilities.parse_specie_id('1_bad_good_id[good_compt]')
        with self.assertRaises(ValueError) as context:
            ModelUtilities.parse_specie_id('good_id[_bad_compt]')

    def test_get_species_types(self):
        self.assertEqual(ModelUtilities.get_species_types([]), [])

        specie_type_ids = [specie_type.id for specie_type in self.model.get_species_types()]
        specie_ids = [specie.serialize() for specie in self.model.get_species()]
        self.assertEqual(sorted(ModelUtilities.get_species_types(specie_ids)), sorted(specie_type_ids))
