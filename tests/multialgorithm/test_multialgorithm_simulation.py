"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

import unittest
import os
from argparse import Namespace
import math
import re
import numpy as np
from scipy.constants import Avogadro

from obj_model import utils
from wc_lang.io import Reader, Writer
from wc_lang.core import (Model, Submodel,  SpeciesType, SpeciesTypeType, Species,
                          Reaction, Observable, Compartment,
                          SpeciesCoefficient, ObservableCoefficient, Parameter,
                          RateLaw, RateLawDirection, RateLawEquation, SubmodelAlgorithm, Concentration,
                          BiomassComponent, BiomassReaction, StopCondition)
from wc_lang.prepare import PrepareModel, CheckModel
from wc_lang.transform import SplitReversibleReactionsTransform
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.make_models import MakeModels
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.config import core as config_core_multialgorithm
config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class TestMultialgorithmSimulation(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        for base_model in [Submodel, Species, SpeciesType]:
            base_model.objects.reset()
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME, strict=False)
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


class TestRunSSASimulation(unittest.TestCase):

    def make_model_and_simulation(self, model_type, specie_copy_numbers=None):
        # reset indices
        for base_model in [Submodel, Species, SpeciesType]:
            base_model.objects.reset()
        # make simple model
        model = self.make_models.make_test_model(model_type, specie_copy_numbers=specie_copy_numbers)
        args = Namespace()
        multialgorithm_simulation = MultialgorithmSimulation(model, args)
        simulation_engine, _ = multialgorithm_simulation.build_simulation()
        return (model, multialgorithm_simulation, simulation_engine)

    def setUp(self):
        self.make_models = MakeModels()

    # TODO(Arthur): make stochastic tests of SSA
    # TODO(Arthur): catch MultialgorithmErrors from get_specie_concentrations, and elsewhere
    # TODO(Arthur): how does performance compare with and without Docker

    def perform_ssa_test_run(self, model_type, run_time, initial_specie_copy_numbers,
        expected_mean_copy_numbers, delta, iterations=5):
        """ Test SSA by comparing expeccted and actual simulation copy numbers

        Args:
            model_type (:obj:`str`): model type description
            run_time (:obj:`float`): duration of the simulation run
            initial_specie_copy_numbers (:obj:`dict`): initial specie counts, with IDs as keys and counts as values
            expected_mean_copy_numbers (:obj:`str`): expected final mean specie counts, in same format as
                `initial_specie_copy_numbers`
            delta (:obj:`int`): threshold difference between expected and actual counts
            iterations (:obj:`int`): number of simulation runs
        """
        # todo: analytically determine the values for delta
        # todo: invariants (:obj:`list`): list of invariant relationships, to be eval'ed
        final_specie_counts = []
        for i in range(iterations):
            _, multialgorithm_simulation, simulation_engine = self.make_model_and_simulation(
                model_type,
                specie_copy_numbers=initial_specie_copy_numbers)
            simulation_engine.initialize()
            num_events_handled = simulation_engine.run(run_time)
            final_specie_counts.append(multialgorithm_simulation.simulation_submodels[0].get_specie_counts())

        mean_final_specie_counts = dict.fromkeys(list(initial_specie_copy_numbers.keys()), 0)
        # todo: use numpy to more compactly compile the mean final specie counts
        for final_specie_count in final_specie_counts:
            for k,v in final_specie_count.items():
                mean_final_specie_counts[k] += v
        for k,v in mean_final_specie_counts.items():
            mean_final_specie_counts[k] = v/iterations
            self.assertAlmostEqual(mean_final_specie_counts[k], expected_mean_copy_numbers[k], delta=delta)

    def test_run_ssa_suite(self):
        self.perform_ssa_test_run('1 species, 1 reaction',
            run_time=1000,
            initial_specie_copy_numbers={'spec_type_0[c]':3000},
            expected_mean_copy_numbers={'spec_type_0[c]':2000},
            delta=50)
        self.perform_ssa_test_run('2 species, 1 reaction',
            run_time=1000,
            initial_specie_copy_numbers={'spec_type_0[c]':3000, 'spec_type_1[c]':0},
            expected_mean_copy_numbers={'spec_type_0[c]':2000,  'spec_type_1[c]':1000},
            delta=50)
