""" Test the Next Reaction Method submodel

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-04-11
:Copyright: 2020, Karr Lab
:License: MIT
"""

from pprint import pprint
import os
import unittest
import numpy as np

from de_sim.simulation_config import SimulationConfig
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

    def test_simulate(self):
        num_trials = 10
        run_time = 10
        num_events = []
        for _ in range(num_trials):
            simulation_engine, _ = self.make_sim_w_nrm_submodel(self.model, True)
            simulation_engine.initialize()
            # expect about 6 reactions per second
            num_events.append(simulation_engine.simulate(run_time))
        print('num_events', num_events)
        # self.assertGreater(num_events, 10)

    '''
    statistical testing strategies:
        expected means:
            6 reactions / second
            each reaction, once / second
            reactions recomputed in determine_dependencies as predicted in expected_dependencies
        expected distributions:
    '''
