""" Test the Next Reaction Method submodel

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-04-11
:Copyright: 2020, Karr Lab
:License: MIT
"""

import os
import unittest

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
        simulation, dynamic_model = multialgorithm_simulation.build_simulation()
        return dynamic_model.dynamic_submodels['nrm_submodel']

    def setUp(self):
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_next_reaction_method_submodel.xlsx')
        self.model = wc_lang.io.Reader().run(self.MODEL_FILENAME)[wc_lang.Model][0]
        self.nrm_submodel = self.make_nrm_submodel(self.model)

    def test_init(self):
        pass

    def test_determine_dependencies(self):
        expected_dependencies = [
            (0, 1,),
            (1, 2,),
            (2, 3,),
            (0, 3,),
            (3, 5,),
            (3, 5,),
        ]
        from pprint import pprint
        dependencies = self.nrm_submodel.determine_dependencies()
        self.assertEqual(dependencies, expected_dependencies)
