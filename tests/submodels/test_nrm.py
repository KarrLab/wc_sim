""" Test the Next Reaction Method submodel

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-04-11
:Copyright: 2020, Karr Lab
:License: MIT
"""

import math
import numpy as np
import os
import unittest

from de_sim.simulation_config import SimulationConfig
from wc_sim import message_types
from wc_sim.multialgorithm_errors import DynamicFrozenSimulationError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.submodels.nrm import NrmSubmodel
from wc_sim.testing.make_models import MakeModel
import wc_lang
import wc_lang.io
import wc_lang.transform


class TestNrmSubmodel(unittest.TestCase):

    def make_sim_w_nrm_submodel(self, model, auto_initialize):
        wc_lang.transform.PrepForWcSimTransform().run(model)
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        nrm_options = dict(auto_initialize=auto_initialize)
        options = {'NrmSubmodel': dict(options=nrm_options)}
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config, options)
        return multialgorithm_simulation.build_simulation()

    def setUp(self):
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_next_reaction_method_submodel.xlsx')
        self.model = wc_lang.io.Reader().run(self.MODEL_FILENAME)[wc_lang.Model][0]
        self.simulation_engine, self.dynamic_model = self.make_sim_w_nrm_submodel(self.model, False)
        self.nrm_submodel = self.dynamic_model.dynamic_submodels['nrm_submodel']

    def test_init(self):
        self.assertTrue(isinstance(self.nrm_submodel.options, dict))

        # test NrmSubmodel() with default options=None
        wc_sim_config = WCSimulationConfig(SimulationConfig(time_max=10))
        _, dynamic_model = MultialgorithmSimulation(self.model, wc_sim_config).build_simulation()
        nrm_submodel = dynamic_model.dynamic_submodels['nrm_submodel']
        self.assertEquals(nrm_submodel.options, None)

    def test_initialize(self):
        attributes_initialized = ['dependencies', 'propensities', 'execution_time_priority_queue']
        self.nrm_submodel.initialize()
        for attr in attributes_initialized:
            self.assertTrue(len(getattr(self.nrm_submodel, attr)))

    def test_determine_dependencies(self):
        expected_dependencies = [
            (0, 1,),
            (1, 2,),
            (2, 3,),
            (0, 3,),
            (3, 5,),
            (3, 5,),
        ]
        dependencies = self.nrm_submodel.determine_dependencies()
        self.assertEqual(dependencies, expected_dependencies)

    def test_initialize_propensities(self):
        propensities = self.nrm_submodel.initialize_propensities()
        self.assertEqual(len(propensities), len(self.nrm_submodel.reactions))
        self.assertTrue(np.all(np.less_equal(0.0, propensities)))

    def test_initialize_execution_time_priorities(self):
        self.nrm_submodel.propensities = self.nrm_submodel.initialize_propensities()
        self.nrm_submodel.initialize_execution_time_priorities()
        time_prev = 0.
        reactions = set()
        while len(self.nrm_submodel.execution_time_priority_queue):
            rxn_first, time_first = self.nrm_submodel.execution_time_priority_queue.topitem()
            self.assertGreater(time_first, time_prev)
            time_prev = time_first
            self.assertNotIn(rxn_first, reactions)
            reactions.add(rxn_first)
            self.nrm_submodel.execution_time_priority_queue.pop()
        self.assertEqual(reactions, set(range(len(self.nrm_submodel.reactions))))

    def test_propensities_eq_to_0(self):
        model_filename = os.path.join(os.path.dirname(__file__), 'fixtures',
                                      'test_next_reaction_method_submodel_2.xlsx')
        test_props_eq_0_model = wc_lang.io.Reader().run(model_filename)[wc_lang.Model][0]
        _, dynamic_model = self.make_sim_w_nrm_submodel(test_props_eq_0_model, False)
        nrm_submodel_0_props = dynamic_model.dynamic_submodels['nrm_submodel']

        ### single step a mock simulation ###
        nrm_submodel_0_props.initialize()

        # execute this sequence of reactions, all of which are enabled and have propensity > 0
        reaction_sequence = [0, 1, 1, 1]
        expected_propensities_sequence = [[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 1, 1],
                                          [0, 1, 2]]
        for expected_propensities in expected_propensities_sequence:
            # check propensities & execution_time_priority_queue
            self.assertEqual(list(nrm_submodel_0_props.propensities), expected_propensities)
            for rxn_idx, propensity in enumerate(expected_propensities):
                if propensity == 0:
                    self.assertEqual(nrm_submodel_0_props.execution_time_priority_queue[rxn_idx], float('inf'))
                else:
                    self.assertLess(nrm_submodel_0_props.execution_time_priority_queue[rxn_idx], float('inf'))

            # mock a simulation event for a reaction with a non-zero propensity
            first_reaction = reaction_sequence.pop(0)
            nrm_submodel_0_props.execute_nrm_reaction(first_reaction)
            nrm_submodel_0_props.schedule_next_reaction(first_reaction)

    def test_simulate(self):
        NUM_TRIALS = 20
        RUN_TIME = 50
        # the parameters in test_next_reaction_method_submodel.xlsx give each reaction an
        # initial rate of 1/sec, which will change slowly because init. populations are 1000
        num_events = []
        for _ in range(NUM_TRIALS):
            simulation_engine, _ = self.make_sim_w_nrm_submodel(self.model, True)
            simulation_engine.initialize()
            num_events.append(simulation_engine.simulate(RUN_TIME))
        num_reactions = len(self.model.reactions)
        expected_mean_num_events = num_reactions * RUN_TIME
        sd = math.sqrt(expected_mean_num_events)
        self.assertLess(expected_mean_num_events - sd, np.mean(num_events))
        self.assertLess(np.mean(num_events), expected_mean_num_events + sd)
