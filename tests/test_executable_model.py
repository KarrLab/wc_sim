'''Test executable_model.py.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-11-10
:Copyright: 2016, Karr Lab
:License: MIT
'''

import os, unittest
from wc_sim.multialgorithm.executable_model import ExecutableModel
from wc_sim.multialgorithm.submodels.submodel import Submodel
from wc_lang.core import Model
from wc_lang.io import Excel

class TestExecutableModel(unittest.TestCase):
    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # make a model
        self.model = Excel.read(self.MODEL_FILENAME)

    def test_find_private_species(self):
        private_species = ExecutableModel.find_private_species(self.model)
        self.assertEqual(private_species['submodel_1'],
            set(['specie_1[c]', 'specie_1[e]', 'specie_2[e]']))
        self.assertEqual(private_species['submodel_2'],
            set(['specie_4[c]', 'specie_5[c]', 'specie_6[c]']))

    def test_find_shared_species(self):
        shared_species = ExecutableModel.find_shared_species(self.model)
        self.assertEqual(shared_species, {'specie_2[c]', 'specie_3[c]'})

    def test_error_free_set_up_simulation(self):
        all_species = [s for s in ExecutableModel.all_species(self.model)]
        self.assertTrue(all(map(lambda v: v in all_species, ['specie_1[c]', 'specie_6[e]'])))

        ExecutableModel.set_up_simulation(self.model)
        reaction_dict = {rxn.id:rxn for rxn in self.model.reactions}
        self.assertAlmostEqual(1.0,
            Submodel.eval_rate_law(reaction_dict['reaction_2'],
            ExecutableModel.get_initial_specie_concentrations(self.model)), 3)

    def test_erroneous_set_up_simulation(self):
        pass
