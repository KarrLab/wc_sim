'''Test executable_model.py.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-11-10
:Copyright: 2016, Karr Lab
:License: MIT
'''

import os, unittest
from obj_model import utils
from wc_sim.multialgorithm.executable_model import ExecutableModel
from wc_sim.multialgorithm.submodels.submodel import Submodel
from wc_lang.core import Model
from wc_lang.io import Reader

class TestExecutableModel(unittest.TestCase):
    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def get_submodel(self, id):
        return utils.get_component_by_id(self.model.submodels, id)

    def get_specie(self, id):
        for specie in self.model.get_species():
            if specie.serialize() == id:
                return specie
        return None

    def setUp(self):
        # read a model
        self.model = Reader().run(self.MODEL_FILENAME)

    def test_find_species_sharing(self):
        private_species = ExecutableModel.find_private_species(self.model)
        self.assertEqual(private_species[self.get_submodel('submodel_1')],
            set(map(lambda x: self.get_specie(x), ['specie_1[c]', 'specie_1[e]', 'specie_2[e]'])))
        self.assertEqual(private_species[self.get_submodel('submodel_2')],
            set(map(lambda x: self.get_specie(x), ['specie_4[c]', 'specie_5[c]', 'specie_6[c]'])))

        private_species = ExecutableModel.find_private_species(self.model, return_ids=True)
        self.assertEqual(private_species['submodel_1'], set(['specie_1[c]', 'specie_1[e]', 'specie_2[e]']))
        self.assertEqual(private_species['submodel_2'], set(['specie_4[c]', 'specie_5[c]', 'specie_6[c]']))

        self.assertEqual(ExecutableModel.find_shared_species(self.model),
            set(map(lambda x: self.get_specie(x), ['specie_2[c]', 'specie_3[c]'])))

        self.assertEqual(ExecutableModel.find_shared_species(self.model, return_ids=True),
            set(['specie_2[c]', 'specie_3[c]']))

    @unittest.skip("skip until simulation setup works")
    def test_error_free_set_up_simulation(self):
        all_species = self.model.get_species()
        self.assertEqual(len(all_species), 8)
        self.assertTrue(all(map(lambda v: v in all_species,
            set(map(lambda x: self.get_specie(x), ['specie_1[c]', 'specie_2[e]'])))))

        ExecutableModel.set_up_simulation(self.model)
        reaction_dict = {rxn.id:rxn for rxn in self.model.reactions}
        self.assertAlmostEqual(1.0,
            Submodel.eval_rate_law(reaction_dict['reaction_2'],
            ExecutableModel.get_initial_specie_concentrations(self.model)), 3)

    def test_erroneous_set_up_simulation(self):
        pass
