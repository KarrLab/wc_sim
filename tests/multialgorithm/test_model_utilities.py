'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-10
:Copyright: 2016-2017, Karr Lab
:License: MIT
'''

import unittest, os
from argparse import Namespace

from wc_lang.io import Reader
from wc_lang.core import RateLawEquation, RateLaw, Reaction
from obj_model import utils
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.model_utilities import ModelUtilities


class TestModelUtilities(unittest.TestCase):

    def get_submodel(self, id):
        return utils.get_component_by_id(self.model.submodels, id)

    def get_specie(self, id):
        for specie in self.model.get_species():
            if specie.serialize() == id:
                return specie
        return None

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # read a model
        self.model = Reader().run(self.MODEL_FILENAME)

    def test_find_private_species(self):
        # since set() operations are being used, this test does not ensure that the methods being
        # tested are deterministic and repeatable; repeatability should be tested by running the
        # code under different circumstances and ensuring identical output
        private_species = ModelUtilities.find_private_species(self.model)
        self.assertEqual(set(private_species[self.get_submodel('submodel_1')]),
            set(map(lambda x: self.get_specie(x), ['specie_1[c]', 'specie_1[e]', 'specie_2[e]',
            'biomass[c]'])))
        self.assertEqual(set(private_species[self.get_submodel('submodel_2')]),
            set(map(lambda x: self.get_specie(x), ['specie_4[c]', 'specie_5[c]', 'specie_6[c]'])))

        private_species = ModelUtilities.find_private_species(self.model, return_ids=True)
        self.assertEqual(set(private_species['submodel_1']), set(['specie_1[c]', 'specie_1[e]',
            'specie_2[e]', 'biomass[c]']))
        self.assertEqual(set(private_species['submodel_2']), set(['specie_4[c]', 'specie_5[c]', 'specie_6[c]']))

    def test_find_shared_species(self):
        self.assertEqual(set(ModelUtilities.find_shared_species(self.model)),
            set(map(lambda x: self.get_specie(x), ['specie_2[c]', 'specie_3[c]'])))

        self.assertEqual(set(ModelUtilities.find_shared_species(self.model, return_ids=True)),
            set(['specie_2[c]', 'specie_3[c]']))

    def test_transcode_and_eval_rate_laws(self):
        overlapping_species_names_model = Reader().run(
            os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model_bad_species_names.xlsx'))
        for mdl in [self.model, overlapping_species_names_model]:
            multialgorithm_simulation = MultialgorithmSimulation(mdl, Namespace())
            # transcode rate laws
            multialgorithm_simulation.transcode_rate_laws()
            concentrations = {}
            for specie in mdl.get_species():
                try:
                    concentrations[specie.serialize()] = specie.concentration.value
                except:
                    pass

            # evaluate the rate laws
            expected = {}
            expected['reaction_1'] = [0.0002]
            expected['reaction_2'] = [1.]
            expected['reaction_3'] = [.5, 0.003]
            expected['reaction_4'] = [0.0005]
            expected['biomass'] = []
            for reaction in mdl.get_reactions():
                rates = ModelUtilities.eval_rate_laws(reaction, concentrations)
                self.assertEqual(rates, expected[reaction.id])

    def test_eval_rate_law_exceptions(self):
        rate_law_equation = RateLawEquation(
            expression='',
            transcoded='',
        )
        rate_law = RateLaw(
            equation=rate_law_equation,
        )
        rate_law_equation.rate_law = rate_law
        reaction = Reaction(
            id='test_reaction',
            name='test_reaction',
            rate_laws=[rate_law]
        )
        rate_law_equation.transcoded='foo foo'
        with self.assertRaises(ValueError) as context:
            ModelUtilities.eval_rate_laws(reaction, {})
        rate_law_equation.transcoded='cos(1.)'
        with self.assertRaises(NameError) as context:
            ModelUtilities.eval_rate_laws(reaction, {})
        rate_law_equation.transcoded='log(1.)'
        self.assertEqual(ModelUtilities.eval_rate_laws(reaction, {}), [0])

    def test_get_initial_specie_concentrations(self):
        initial_specie_concentrations = ModelUtilities.initial_specie_concentrations(self.model)
        some_specie_concentrations = {
            'specie_2[e]':2.0E-4,
            'specie_2[c]':5.0E-4,
            'biomass[c]':0. }
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
