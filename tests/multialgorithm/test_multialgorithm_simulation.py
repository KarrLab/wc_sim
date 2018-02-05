'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017-2018, Karr Lab
:License: MIT
'''

import os, unittest
from argparse import Namespace
import six
import math
import numpy as np
from scipy.constants import Avogadro

from wc_sim.multialgorithm.multialgorithm_simulation import (DynamicModel, DynamicCompartment,
    MultialgorithmSimulation, WATER_ID)
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
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
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')

    def setUp(self):
        for model in [Submodel, Reaction, SpeciesType]:
            model.objects.reset()

    def read_model(self, model_filename):
        # read and initialize a model
        self.model = Reader().run(model_filename)
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        self.dynamic_model = DynamicModel(self.model, self.multialgorithm_simulation)

    def test_initialize_dynamic_model(self):
        self.read_model(self.MODEL_FILENAME)
        self.dynamic_model.initialize()
        self.assertEqual(self.dynamic_model.extracellular_volume, 1.00E-12)
        self.assertEqual(self.dynamic_model.volume, 4.58E-17)
        self.assertEqual(self.dynamic_model.fraction_dry_weight, 0.3)
        self.assertAlmostEqual(self.dynamic_model.mass, 1.562E-42)
        self.assertAlmostEqual(self.dynamic_model.dry_weight, 4.688E-43)
        self.assertAlmostEqual(self.dynamic_model.density, 3.412E-26)

    def test_dry_dynamic_model(self):
        self.read_model(self.DRY_MODEL_FILENAME)
        self.dynamic_model.initialize()
        self.assertEqual(self.dynamic_model.mass, self.dynamic_model.dry_weight)


class TestDynamicCompartment(unittest.TestCase):

    def test_simple_dynamic_compartment(self):

        # make a LocalSpeciesPopulation
        num_species = 100
        species_nums = list(range(0, num_species))
        species = list(map(lambda x: "specie_{}".format(x), species_nums))
        all_pops = 1E6
        init_populations = dict(zip(species, [all_pops]*len(species_nums)))
        all_m_weights = 50
        molecular_weights = dict(zip(species, [all_m_weights]*len(species_nums)))
        local_species_pop = LocalSpeciesPopulation('test', init_populations, molecular_weights)

        # make a DynamicCompartment
        id = 'id'
        name = 'name'
        vol = 1E-17
        dynamic_compartment = DynamicCompartment(id, name, vol, local_species_pop)

        self.assertEqual(dynamic_compartment.volume(), vol)
        self.assertIn(dynamic_compartment.id, str(dynamic_compartment))
        self.assertIn("Fold change volume: 1.0", str(dynamic_compartment))
        estimated_mass = len(species_nums)*all_pops*all_m_weights/Avogadro
        self.assertAlmostEqual(dynamic_compartment.mass(), estimated_mass)

        # compartment containing just the first species
        dynamic_compartment = DynamicCompartment(id, name, vol, local_species_pop, species_ids=species[:1])
        estimated_mass = all_pops*all_m_weights/Avogadro
        self.assertAlmostEqual(dynamic_compartment.mass(), estimated_mass)


class TestMultialgorithmSimulation(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        for base_model in [Submodel, Species, SpeciesType]:
            base_model.objects.reset()
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME)
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        self.dynamic_model = DynamicModel(self.model, self.multialgorithm_simulation)

    @unittest.skip("to be fixed")
    def test_initialize_simulation(self):
        self.multialgorithm_simulation.initialize()
        self.simulation_engine = self.multialgorithm_simulation.build_simulation()
        self.assertEqual(len(self.simulation_engine.simulation_objects.keys()), 3)
        for name,simulation_obj in six.iteritems(self.simulation_engine.simulation_objects):
            print("\n{}: {} event queue:".format(simulation_obj.__class__.__name__, name))
            print(simulation_obj.event_queue_to_str())
        self.simulation_engine.simulate(10)
