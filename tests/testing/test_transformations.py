""" Test model transformations for testing

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-12-11
:Copyright: 2020, Karr Lab
:License: MIT
"""

import os
import unittest

from wc_lang.io import Reader
from wc_sim.testing.transformations import SetStdDevsToZero
import wc_lang


class TestSetStdDevsToZero(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'test_model.xlsx')

    def setUp(self):
        self.model = Reader().run(self.MODEL_FILENAME, ignore_extra_models=True)[wc_lang.Model][0]

    def compt_vol_stds(self):
        return [obj.init_volume.std for obj in self.model.get_compartments()]

    def dist_init_conc_vol_stds(self):
        return [obj.std for obj in self.model.get_distribution_init_concentrations()]

    def param_stds(self):
        return [obj.std for obj in self.model.get_parameters()]

    def test_SetStdDevsToZero(self):
        # ensure that some std. devs. of each model type are non-zero
        self.assertTrue(sum(self.compt_vol_stds()))
        self.assertTrue(sum(self.dist_init_conc_vol_stds()))
        self.assertTrue(sum(self.param_stds()))

        SetStdDevsToZero().run(self.model)
        # ensure that all std. devs. of each model type are 0
        self.assertEqual(sum(self.compt_vol_stds()), 0)
        self.assertEqual(sum(self.dist_init_conc_vol_stds()), 0)
        self.assertEqual(sum(self.param_stds()), 0)
