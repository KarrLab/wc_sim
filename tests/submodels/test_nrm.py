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
from wc_onto import onto
from wc_sim import message_types
from wc_sim.multialgorithm_errors import DynamicFrozenSimulationError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.submodels.nrm import NrmSubmodel
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import get_expected_dependencies
import wc_lang
import wc_lang.io


class TestNrmSubmodel(unittest.TestCase):

    def make_sim_w_nrm_submodel(self, model):
        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, ode_time_step=1)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        return multialgorithm_simulation.build_simulation()

    def setUp(self):
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_next_reaction_method_submodel.xlsx')
        self.model = wc_lang.io.Reader().run(self.MODEL_FILENAME)[wc_lang.Model][0]
        self.simulator, self.dynamic_model = self.make_sim_w_nrm_submodel(self.model)
        self.nrm_submodel = self.dynamic_model.dynamic_submodels['nrm_submodel']

    def test_prepare(self):
        attributes_prepared = ['dependencies', 'propensities', 'execution_time_priority_queue']
        for attr in attributes_prepared:
            setattr(self.nrm_submodel, attr, None)
        self.nrm_submodel.prepare_submodel()
        for attr in attributes_prepared:
            self.assertTrue(len(getattr(self.nrm_submodel, attr)))

    def test_determine_dependencies(self):
        expected_dependencies_1 = [
            (1,),
            (2,),
            (3,),
            (0,),
            (3, 5,),
            (3,),
        ]
        dependencies = self.nrm_submodel.determine_dependencies()
        self.assertEqual(dependencies, expected_dependencies_1)

        # test dependencies indirectly propagated through expressions
        dependencies_mdl_file = os.path.join(os.path.dirname(__file__), '..', 'fixtures',
                                             'test_dependencies.xlsx')
        dependencies_model = wc_lang.io.Reader().run(dependencies_mdl_file)[wc_lang.Model][0]
        # make the DSA submodel an NRM submodel
        dsa_submodel = dependencies_model.submodels.get_one(id='dsa_submodel')
        dsa_submodel.id = 'nrm_submodel'
        dsa_submodel.framework = onto['WC:next_reaction_method']
        _, dynamic_model = self.make_sim_w_nrm_submodel(dependencies_model)
        nrm_submodel = dynamic_model.dynamic_submodels['nrm_submodel']
        nrm_dependencies = nrm_submodel.determine_dependencies()

        ode_submodel = dynamic_model.dynamic_submodels['ode_submodel']
        ode_submodel_rxn_ids = [rxn.id for rxn in ode_submodel.reactions]

        # rate laws that depend on reactions in other submodels
        # convert nrm_dependencies entries into ids
        nrm_dependencies_as_ids = {}
        for rxn_idx, rate_law_dependencies in enumerate(nrm_dependencies):
            rxn = nrm_submodel.reactions[rxn_idx]
            nrm_dependencies_as_ids[rxn.id] = set()
            for rate_law_idx in rate_law_dependencies:
                rate_law_id = nrm_submodel.reactions[rate_law_idx].rate_laws[0].id
                nrm_dependencies_as_ids[rxn.id].add(rate_law_id)

        expected_dependencies = get_expected_dependencies()
        expected_nrm_dependencies_as_ids = expected_dependencies['DynamicRateLaw']
        # NRM dependencies do not have entries for reactions in other submodels
        for rxn_id in ode_submodel_rxn_ids:
            del expected_nrm_dependencies_as_ids[rxn_id]

        # NRM dependencies keep only rate laws used by nrm_submodel
        rate_laws_used_by_nrm_submodel = [rxn.rate_laws[0].id for rxn in nrm_submodel.reactions]
        for rxn_id in expected_nrm_dependencies_as_ids:
            expected_nrm_dependencies_as_ids[rxn_id].intersection_update(rate_laws_used_by_nrm_submodel)

        # Each NRM reaction must re-evaluate all of its rate laws that depend on other submodels
        expected_dependencies = get_expected_dependencies()
        rate_laws_modified_by_ode_submodel = set()
        for rxn_id in ode_submodel_rxn_ids:
            rate_laws_modified_by_ode_submodel.update(expected_dependencies['DynamicRateLaw'][rxn_id])
        # keep only rate laws used by nrm_submodel
        rate_laws_modified_by_ode_submodel.intersection_update(rate_laws_used_by_nrm_submodel)
        for rxn_id in expected_nrm_dependencies_as_ids:
            expected_nrm_dependencies_as_ids[rxn_id].update(rate_laws_modified_by_ode_submodel)

        # NRM dependencies do not include self-references
        for rxn_id in expected_nrm_dependencies_as_ids:
            rate_law_id = f"{rxn_id}-forward"
            expected_nrm_dependencies_as_ids[rxn_id].discard(rate_law_id)

        self.assertEqual(nrm_dependencies_as_ids, expected_nrm_dependencies_as_ids)

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
        _, dynamic_model = self.make_sim_w_nrm_submodel(test_props_eq_0_model)
        nrm_submodel_0_props = dynamic_model.dynamic_submodels['nrm_submodel']

        # stop caching for this test, because it repeatedly updates populations & no invalidation is done
        dynamic_model._stop_caching()

        ### single step a mock simulation ###
        nrm_submodel_0_props.prepare_submodel()

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
                    self.assertEqual(nrm_submodel_0_props.execution_time_priority_queue[rxn_idx],
                                     float('inf'))
                else:
                    self.assertLess(nrm_submodel_0_props.execution_time_priority_queue[rxn_idx],
                                    float('inf'))

            # mock a simulation event for a reaction with a non-zero propensity
            next_reaction = reaction_sequence.pop(0)
            nrm_submodel_0_props.execute_nrm_reaction(next_reaction)
            nrm_submodel_0_props.schedule_next_reaction(next_reaction)

    def test_simulate(self):
        NUM_TRIALS = 20
        RUN_TIME = 50
        # the parameters in test_next_reaction_method_submodel.xlsx give each reaction an
        # initial rate of 1/sec, which will change slowly because init. populations are 1000
        num_events = []
        for _ in range(NUM_TRIALS):
            simulator, _ = self.make_sim_w_nrm_submodel(self.model)
            simulator.initialize()
            num_events.append(simulator.simulate(RUN_TIME).num_events)
        num_reactions = len(self.model.reactions)
        expected_mean_num_events = num_reactions * RUN_TIME
        sd = math.sqrt(expected_mean_num_events)
        self.assertLess(expected_mean_num_events - sd, np.mean(num_events))
        self.assertLess(np.mean(num_events), expected_mean_num_events + sd)
