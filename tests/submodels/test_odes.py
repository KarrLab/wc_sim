"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-10-12
:Copyright: 2018, Karr Lab
:License: MIT
"""

from scikits.odes import ode
import numpy as np
import os
import re
import scikits
import unittest

from de_sim.simulation_config import SimulationConfig
from wc_lang.core import ReactionParticipantAttribute
from wc_lang.io import Reader
from wc_sim.dynamic_components import DynamicRateLaw
from wc_sim.message_types import RunOde
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.submodels.odes import OdeSubmodel
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import read_model_for_test, TempConfigFileModifier


class TestOdeSubmodel(unittest.TestCase):

    def setUp(self):
        self.default_species_copy_number = 1000000000.123
        self.mdl_1_spec = \
            MakeModel.make_test_model('2 species, 1 reaction, with rates given by reactant population',
                                      init_vol_stds=[0],
                                      default_species_copy_number=self.default_species_copy_number,
                                      default_species_std=0,
                                      submodel_framework='WC:ordinary_differential_equations')
        self.ode_submodel_1 = self.make_ode_submodel(self.mdl_1_spec)
        self.config_file_modifier = TempConfigFileModifier()

    def tearDown(self):
        self.config_file_modifier.clean_up()

    def make_ode_submodel(self, model, ode_time_step=1.0, submodel_name='submodel_1'):
        """ Make a MultialgorithmSimulation from a wc lang model """
        # assume a single submodel
        self.ode_time_step = ode_time_step
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, ode_time_step=ode_time_step)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
        simulation_engine.initialize()
        return dynamic_model.dynamic_submodels[submodel_name]

    ### test low level methods ###
    def test_ode_submodel_init(self):
        self.assertEqual(self.ode_submodel_1.ode_time_step, self.ode_time_step)

        # test exceptions
        bad_ode_time_step = -2
        with self.assertRaisesRegexp(MultialgorithmError,
            'ode_time_step must be positive, but is {}'.format(bad_ode_time_step)):
            self.make_ode_submodel(self.mdl_1_spec, ode_time_step=bad_ode_time_step)

        with self.assertRaisesRegexp(MultialgorithmError, "ode_time_step must be a number but is "):
            self.make_ode_submodel(self.mdl_1_spec, ode_time_step=None)

    def test_set_up_optimizations(self):
        ode_submodel = self.ode_submodel_1
        self.assertTrue(set(ode_submodel.ode_species_ids) == ode_submodel.ode_species_ids_set \
            == set(ode_submodel.adjustments.keys()))
        self.assertEqual(ode_submodel.populations.shape, ((len(ode_submodel.ode_species_ids), )))

    # todo: for the next 4 tests, check results against raw properties of self.mdl_1_spec
    def test_solver_lock(self):
        self.ode_submodel_empty = OdeSubmodel('test_1', None, [], [], [], None, 1)
        self.assertTrue(self.ode_submodel_empty.get_solver_lock())
        with self.assertRaisesRegexp(DynamicMultialgorithmError, 'OdeSubmodel .*: cannot get_solver_lock'):
            self.ode_submodel_empty.get_solver_lock()
        self.assertTrue(self.ode_submodel_empty.release_solver_lock())

    ### without running the simulator, test solving ###
    def test_set_up_ode_submodel(self):
        self.ode_submodel_1.set_up_ode_submodel()

        rate_of_change_expressions = self.ode_submodel_1.rate_of_change_expressions
        # model has 2 species: spec_type_0[compt_1] and spec_type_1[compt_1]
        self.assertEqual(len(rate_of_change_expressions), 2)
        # spec_type_0[compt_1] participates in 1 reaction
        self.assertEqual(len(rate_of_change_expressions[0]), 1)
        # the reaction consumes 1 spec_type_0[compt_1]
        coeffs, rate_law = rate_of_change_expressions[0][0]
        self.assertEqual(coeffs, -1)
        self.assertEqual(type(rate_law), DynamicRateLaw)

    def test_create_ode_solver(self):
        self.assertTrue(isinstance(self.ode_submodel_1.create_ode_solver(**{}), ode))

    def test_right_hand_side(self):
        pop_change_rates = [0, 0]
        num_reactants = 100
        self.assertEqual(0, self.ode_submodel_1.right_hand_side(0, [num_reactants, 0], pop_change_rates))
        pop_change_rate_spec_1 = pop_change_rates[1]
        self.assertTrue(0 < pop_change_rate_spec_1)
        self.ode_submodel_1.testing = False
        self.assertEqual(0, self.ode_submodel_1.right_hand_side(0, [num_reactants, 0], pop_change_rates))

        # test exceptions
        short_list = [1]
        with self.assertRaisesRegex(DynamicMultialgorithmError, 'OdeSubmodel .*: solver.right_hand_side\(\) failed'):
            self.ode_submodel_1.right_hand_side(0, [num_reactants, 0], short_list)

        # rates decline as reactant converted to product
        for i in range(1, 10):
            self.ode_submodel_1.right_hand_side(0, [num_reactants - i, i], pop_change_rates)
            self.assertTrue(pop_change_rates[1] < pop_change_rate_spec_1)
            pop_change_rate_spec_1 = pop_change_rates[1]

    def do_test_compute_population_change_rates_control_caching(self, caching_settings):
        # TODO(Arthur): exact caching: test with all 3 levels of caching
        ### test with caching specified by caching_settings ###
        self.config_file_modifier.write_test_config_file(caching_settings)

        one_rxn_exponential_file = os.path.join(os.path.dirname(__file__), '..', 'fixtures',
                                                 'dynamic_tests', f'one_rxn_exponential.xlsx')
        one_rxn_exp_mdl = read_model_for_test(one_rxn_exponential_file,
                                              integration_framework='WC:ordinary_differential_equations')
        ode_submodel = self.make_ode_submodel(one_rxn_exp_mdl, submodel_name='submodel')

        # test rxn: A[c] ==> B[c] @ 0.04 * B[c]
        ode_submodel.current_species_populations()
        new_species_populations = ode_submodel.populations.copy()
        initial_population_change_rates = np.zeros(ode_submodel.num_species)
        time = 0
        ode_submodel.compute_population_change_rates(time, new_species_populations,
                                                     initial_population_change_rates)
        # slope of species populations depends linearly on [B[c]]
        # doubling species populations should double population slopes
        new_species_populations = 2 * new_species_populations
        updated_population_change_rates = np.zeros(ode_submodel.num_species)
        ode_submodel.compute_population_change_rates(time, new_species_populations,
                                                     updated_population_change_rates)
        np.testing.assert_allclose(2 * initial_population_change_rates, updated_population_change_rates)

        # compute_population_change_rates replaces negative species populations with 0s in the rate computation
        # this should produce slopes of 0
        new_species_populations.fill(-1)
        ode_submodel.compute_population_change_rates(time, new_species_populations,
                                                     initial_population_change_rates)
        np.testing.assert_array_equal(initial_population_change_rates, np.zeros(ode_submodel.num_species))

        # create ODE submodel with caching_settings
        ode_submodel_1 = self.make_ode_submodel(self.mdl_1_spec)
        # test rxn: [compt_1]: spec_type_0 => spec_type_1 @ k * spec_type_0 / Avogadro / volume_compt_1
        # doubling population should double volume, leaving the slopes of species populations unchanged
        initial_population_change_rates = np.zeros(ode_submodel_1.num_species)
        new_species_populations = np.full(ode_submodel_1.num_species,
                                          self.default_species_copy_number)
        ode_submodel_1.compute_population_change_rates(time, new_species_populations,
                                                            initial_population_change_rates)
        new_species_populations.fill(2 * self.default_species_copy_number)
        updated_population_change_rates = np.zeros(ode_submodel_1.num_species)
        ode_submodel_1.compute_population_change_rates(time, new_species_populations,
                                                            updated_population_change_rates)
        np.testing.assert_allclose(initial_population_change_rates, updated_population_change_rates)

    def test_compute_population_change_rates_control_caching(self):
        ### test all 3 caching combinations ###
        # NO CACHING
        # EVENT_BASED invalidation
        # REACTION_DEPENDENCY_BASED invalidation
        for caching_settings in ([('expression_caching', 'False')],
                                 [('expression_caching', 'True'),
                                  ('cache_invalidation', 'event_based')],
                                 [('expression_caching', 'True'),
                                  ('cache_invalidation', 'reaction_dependency_based')]):
            self.do_test_compute_population_change_rates_control_caching(caching_settings)

    def test_current_species_populations(self):
        self.ode_submodel_1.current_species_populations()
        for pop in self.ode_submodel_1.populations:
            self.assertAlmostEqual(pop, self.default_species_copy_number, delta=0.5)

    # TODO(Arthur): make this a real test
    def test_run_ode_solver(self):
        print()
        attr = ReactionParticipantAttribute()
        for rxn in self.mdl_1_spec.reactions:
            print('rxn:', attr.serialize(rxn.participants))
            print('rate law:', rxn.rate_laws[0].expression.serialize())
        print(self.ode_submodel_1.local_species_population)
        # print('\t'.join(OdeSubmodel.run_ode_solver_header))
        summary = ['external step',
                   f'{0:.4e}',
                   f'{0:.4e}',
                   f'N/A',
                   f'N/A',
                   f'{self.default_species_copy_number:.1f}',
                   f'{self.default_species_copy_number:.1f}']
        print('\t'.join(summary))
        n = 1
        for i in range(n):
            self.ode_submodel_1.run_ode_solver()
        print('end data')
        print(self.ode_submodel_1.local_species_population)

        # make OdeSubmodel.right_hand_side fail
        self.ode_submodel_1.num_species = None
        with self.assertRaises(DynamicMultialgorithmError):
            self.ode_submodel_1.run_ode_solver()

    # test event scheduling and handling
    @unittest.skip("not good test")
    def test_schedule_next_ode_analysis(self):
        custom_ode_time_step = 4
        custom_ode_submodel = self.make_ode_submodel(self.mdl_1_spec, ode_time_step=custom_ode_time_step)
        # no events are scheduled
        self.assertTrue(custom_ode_submodel.simulator.event_queue.empty())

        # check that the next event is a RunOde message at time expected_time
        def check_next_event(expected_time):
            next_event = custom_ode_submodel.simulator.event_queue.next_events()[0]
            self.assertEqual(next_event.creation_time, 0)
            self.assertEqual(next_event.event_time, expected_time)
            self.assertEqual(next_event.sending_object, custom_ode_submodel)
            self.assertEqual(next_event.receiving_object, custom_ode_submodel)
            self.assertEqual(type(next_event.message), RunOde)
            self.assertTrue(custom_ode_submodel.simulator.event_queue.empty())

        # initial event should be at 0
        custom_ode_submodel.send_initial_events()
        check_next_event(0)

        # next RunOde event should be at custom_ode_time_step
        custom_ode_submodel.schedule_next_ode_analysis()
        check_next_event(custom_ode_time_step)
