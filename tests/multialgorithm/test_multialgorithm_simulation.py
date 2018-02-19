"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

import os, unittest
from argparse import Namespace
import six
import math
import numpy as np
from scipy.constants import Avogadro

from wc_sim.multialgorithm.dynamic_components import DynamicModel
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from obj_model import utils
from wc_lang.io import Reader
from wc_lang.core import (Reaction, RateLaw, RateLawEquation, Submodel, SubmodelAlgorithm,
    Species, RateLawDirection, SpeciesType)
from wc_sim.multialgorithm.config import core as config_core_multialgorithm
config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class TestMultialgorithmSimulation(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        for base_model in [Submodel, Species, SpeciesType]:
            base_model.objects.reset()
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME)
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        self.dynamic_model = DynamicModel(self.model)

    @unittest.skip("to be fixed")
    def test_initialize_simulation(self):
        self.multialgorithm_simulation.initialize()
        self.simulation_engine = self.multialgorithm_simulation.build_simulation()
        self.assertEqual(len(self.simulation_engine.simulation_objects.keys()), 3)
        for name,simulation_obj in six.iteritems(self.simulation_engine.simulation_objects):
            print("\n{}: {} event queue:".format(simulation_obj.__class__.__name__, name))
            print(simulation_obj.render_event_queue())
        self.simulation_engine.simulate(10)

    def test_static_methods(self):
        initial_species_population = MultialgorithmSimulation.get_initial_species_pop(self.model)
        specie_wo_init_conc = 'specie_3[c]'
        self.assertEqual(initial_species_population[specie_wo_init_conc], 0)
        self.assertEqual(initial_species_population['specie_2[c]'], initial_species_population['specie_4[c]'])
        for concentration in self.model.get_concentrations():
            self.assertGreater(initial_species_population[concentration.species.id()], 0)

        local_species_population = MultialgorithmSimulation.make_local_species_pop(self.model)
        self.assertEqual(local_species_population.read_one(0, specie_wo_init_conc), 0)
