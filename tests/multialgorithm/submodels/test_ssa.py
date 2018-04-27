'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-10-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''
import unittest
import os

from wc_lang.io import Reader
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm.submodels.ssa import SSASubmodel
from wc_sim.multialgorithm.message_types import GivePopulation, ExecuteSsaReaction


@unittest.skip("still broken")
class TestSsaSubmodel(unittest.TestCase):

    def setUp(self):
        '''
        SimulationEngine.reset()
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
        # make a model
        self.model = Reader().run(self.MODEL_FILENAME, strict=False)
        ExecutableModel.set_up_simulation(self.model)
        '''
        self.ssa_submodel = SSASubmodel(
            
        )

    #TODO(Arthur): make stochastic tests of SSA
    def test_SSA_submodel_methods(self):
        for submodel_spec in self.model.submodels:
            if submodel_spec.name == 'RNA degradation':
                '''
                for e in [submodel_spec.name,
                    submodel_spec.id,
                    submodel_spec.reactions,
                    submodel_spec.species,
                    submodel_spec.parameters]:
                    print( obj_2_str(e))
                '''
                ssa1 = SSASubmodel(self.model, submodel_spec.name,
                    self.model.cell_state,
                    submodel_spec.reactions,
                    submodel_spec.species,
                    submodel_spec.parameters)
                self.assertEqual(ssa1.num_SsaWaits, 0)
