'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017, Karr Lab
:License: MIT
'''

import os, unittest
from argparse import Namespace
import six
import math
import numpy as np

from wc_sim.multialgorithm.multialgorithm_simulation import (DynamicModel, MultialgorithmSimulation)
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from obj_model import utils
from wc_lang.io import Reader
from wc_lang.core import (Reaction, RateLaw, RateLawEquation, Submodel, SubmodelAlgorithm,
    Species, RateLawDirection, SpeciesType)
from wc_utils.config.core import ConfigManager
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
config_multialgorithm = \
    ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        for model in [Submodel, Reaction, SpeciesType]:
            model.objects.reset()
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME)
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        self.dynamic_model = DynamicModel(self.model, self.multialgorithm_simulation)

    def test_initialize_dynamic_model(self):
        self.dynamic_model.initialize()
        self.assertEqual(self.dynamic_model.extracellular_volume, 1.00E-12)
        self.assertEqual(self.dynamic_model.volume, 4.58E-17)
        self.assertEqual(self.dynamic_model.fraction_dry_weight, 0.3)
        self.assertAlmostEqual(self.dynamic_model.mass, 1.56273063E-42)
        self.assertAlmostEqual(self.dynamic_model.dry_weight, 4.68819190E-43)
        self.assertAlmostEqual(self.dynamic_model.density, 3.41207562E-26)


class TestMultialgorithmSimulation(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        Submodel.objects.reset()
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME)
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        self.dynamic_model = DynamicModel(self.model, self.multialgorithm_simulation)

    @unittest.skip('')
    def test_initialize_simulation(self):
        self.multialgorithm_simulation.initialize()
        self.simulation_engine = self.multialgorithm_simulation.build_simulation()
        self.assertEqual(len(self.simulation_engine.simulation_objects.keys()), 3)
        for name,simulation_obj in six.iteritems(self.simulation_engine.simulation_objects):
            print("\n{}: {} event queue:".format(simulation_obj.__class__.__name__, name))
            print(simulation_obj.event_queue_to_str())
        self.simulation_engine.simulate(10)
