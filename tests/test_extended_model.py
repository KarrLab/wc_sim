'''Test extended_model.py.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-11-10
:Copyright: 2016, Karr Lab
:License: MIT
'''

import os, unittest
from wc_sim.multialgorithm.extended_model import ExtendedModel
from wc_lang.core import Model
from wc_lang.io import Excel

class TestExtendedModel(unittest.TestCase):
    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # make a model
        self.model = Excel.read(self.MODEL_FILENAME)

    def test_find_private_species(self):
        private_species = ExtendedModel.find_private_species(self.model)
        self.assertEqual(private_species['submodel_1'],
            set(['specie_1[c]', 'specie_1[e]', 'specie_2[e]']))
        self.assertEqual(private_species['submodel_2'],
            set(['specie_4[c]', 'specie_5[c]', 'specie_6[c]']))

    def test_find_shared_species(self):
        shared_species = ExtendedModel.find_shared_species(self.model)
        self.assertEqual(shared_species, {'specie_2[c]', 'specie_3[c]'})
