"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

import os, unittest
from argparse import Namespace
import math
import numpy as np
from scipy.constants import Avogadro

from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from obj_model import utils
from wc_lang.io import Reader
from wc_lang.core import (Model, Submodel,  SpeciesType, SpeciesTypeType, Species,
                          Reaction, Observable, Compartment,
                          SpeciesCoefficient, ObservableCoefficient, Parameter,
                          RateLaw, RateLawDirection, RateLawEquation, SubmodelAlgorithm, Concentration,
                          BiomassComponent, BiomassReaction, StopCondition)
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

    def test_molecular_weights_for_species(self):
        multi_alg_sim = self.multialgorithm_simulation
        self.assertEqual(multi_alg_sim.molecular_weights_for_species(set()), {})
        expected = {
            'specie_6[c]':6,
            'H2O[c]':18.0152
        }
        self.assertEqual(multi_alg_sim.molecular_weights_for_species(set(expected.keys())),
            expected)

    def test_partition_species(self):
        self.multialgorithm_simulation.partition_species()
        expected_priv_species = dict(
            submodel_1=['specie_1[e]', 'specie_2[e]', 'specie_1[c]'],
            submodel_2=['specie_4[c]', 'specie_5[c]', 'specie_6[c]']
        )
        self.assertEqual(self.multialgorithm_simulation.private_species, expected_priv_species)
        expected_shared_species = set(['specie_2[c]', 'specie_3[c]', 'H2O[e]', 'H2O[c]'])
        self.assertEqual(self.multialgorithm_simulation.shared_species, expected_shared_species)

    def test_dynamic_compartments(self):
        expected_compartments = dict(
            submodel_1=['c', 'e'],
            submodel_2=['c']
        )
        for submodel_id in ['submodel_1', 'submodel_2']:
            submodel = Submodel.objects.get_one(id=submodel_id)
            submodel_dynamic_compartments = self.multialgorithm_simulation.get_dynamic_compartments(submodel)
            self.assertEqual(set(submodel_dynamic_compartments.keys()), set(expected_compartments[submodel_id]))

    def test_static_methods(self):
        initial_species_population = MultialgorithmSimulation.get_initial_species_pop(self.model)
        specie_wo_init_conc = 'specie_3[c]'
        self.assertEqual(initial_species_population[specie_wo_init_conc], 0)
        self.assertEqual(initial_species_population['specie_2[c]'], initial_species_population['specie_4[c]'])
        for concentration in self.model.get_concentrations():
            self.assertGreater(initial_species_population[concentration.species.id()], 0)

        local_species_population = MultialgorithmSimulation.make_local_species_pop(self.model)
        self.assertEqual(local_species_population.read_one(0, specie_wo_init_conc), 0)

    def test_build_simulation(self):
        self.simulation_engine, _ = self.multialgorithm_simulation.build_simulation()
        self.assertEqual(len(self.simulation_engine.simulation_objects.keys()), 2)


class TestRunSimulation(unittest.TestCase):

    def make_test_model(self, model_type):

        # Model
        self.model = Model(id='test_model', wc_lang_version='0.0.1')
        # Compartment
        self.comp = self.model.compartments.create(id='c', initial_volume=1.25)

        # SpeciesTypes, Species and Concentrations
        self.species = []
        for i in range(4):
            spec_type = self.model.species_types.create(
                id='spec_type_{}'.format(i),
                molecular_weight=10 * (i + 1))
            spec = self.comp.species.create(species_type=spec_type)
            self.species.append(spec)
            Concentration(species=spec, value=3)
        # Submodel
        self.submodel = self.model.submodels.create(id='test_submodel', algorithm=SubmodelAlgorithm.ssa,
            compartment=self.comp)
        # Reactions and RateLaws
        self.reaction = self.submodel.reactions.create(id='test_reaction1', reversible=True)
        self.reaction.participants.create(species=self.species[0], coefficient=1)
        self.reaction.participants.create(species=self.species[1], coefficient=-1)
        self.reaction.rate_laws.create(direction=RateLawDirection.forward,
            equation=RateLawEquation(expression='1', transcoded='1'))
        self.reaction.rate_laws.create(direction=RateLawDirection.backward,
            equation=RateLawEquation(expression='1', transcoded='1'))
        # Parameters
        self.model.parameters.create(id='fractionDryWeight', value=0.3)
        self.model.parameters.create(id='carbonExchangeRate', value=12, units='mmol/gDCW/h')
        self.model.parameters.create(id='nonCarbonExchangeRate', value=20, units='mmol/gDCW/h')

        # self.model.pprint(max_depth=3)
        # create Manager indices
        # TODO(Arthur): should be automated in a finalize() method
        for model in [Submodel,  SpeciesType, Reaction, Observable, Compartment, Parameter]:
            model.get_manager().insert_all_new()

    def setUp(self):
        for base_model in [Submodel, Species, SpeciesType]:
            base_model.objects.reset()
        ### make simple model ###
        self.make_test_model(None)
        # SpeciesType.objects._dump_index_dicts()
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)

    def test_run_simulation(self):
        self.simulation_engine, _ = self.multialgorithm_simulation.build_simulation()
        for name,simulation_obj in self.simulation_engine.simulation_objects.items():
            print("\n{}: {} event queue:".format(simulation_obj.__class__.__name__, name))
            print(simulation_obj.render_event_queue())
        # self.simulation_engine.initialize()
