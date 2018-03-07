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


def obj_2_str(obj):
    if isinstance(obj,list):
        return( [obj_2_str(e) for e in obj] )
    if hasattr(obj, '__dict__'):
        return(", ".join(["{}='{}'".format(k,str(v)) for k,v in obj.__dict__.items()]))
    else:
        return str(obj)

@unittest.skip("skip until updated")
class TestSsaSubmodel(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
        # make a model
        self.model = Reader().run(self.MODEL_FILENAME)
        ExecutableModel.set_up_simulation(self.model)

    @unittest.skip("skip")
    def test_SSA_submodel_predictions(self):
        pass
    '''
    TODO(Arthur): make stochastic tests of SSA
    perform a monte Carlo simulation of a trivial model, and compare means of SSA's predictions with expected means
    '''

    @unittest.skip("skip until species state issue addressed")
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
