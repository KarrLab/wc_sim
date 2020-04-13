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

    def make_nrm_submodel(self, model):
        wc_lang.transform.PrepForWcSimTransform().run(model)
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        return multialgorithm_simulation.build_simulation()

    def setUp(self):
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_next_reaction_method_submodel.xlsx')
        self.model = wc_lang.io.Reader().run(self.MODEL_FILENAME)[wc_lang.Model][0]
        self.simulation_engine, self.dynamic_model = self.make_nrm_submodel(self.model)
        self.nrm_submodel = self.dynamic_model.dynamic_submodels['nrm_submodel']

    def test_init(self):
        pass

    def test_prepare(self):
        attributes_prepared = ['dependencies', 'propensities', 'execution_time_priority_queue']
        self.nrm_submodel.prepare()
        for attr in attributes_prepared:
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
        self.nrm_submodel.prepare()
        self.simulation_engine.initialize()
        run_time = 5
        # expect about 6 reactions per second
        num_events = self.simulation_engine.simulate(run_time)
        self.assertGreater(num_events, 10)
