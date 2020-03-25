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
from wc_onto import onto
from wc_sim.dynamic_components import DynamicRateLaw
from wc_sim.message_types import RunOde
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.submodels.odes import OdeSubmodel
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import read_model_for_test


class TestOdeSubmodel(unittest.TestCase):

    # todo: install
    # ODE_TEST_CASES = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'verification', 'testing', 'semantic')

    def setUp(self):
        self.default_species_copy_number = 1000000000.123
        self.mdl_1_spec = \
            MakeModel.make_test_model('2 species, 1 reaction, with rates given by reactant population',
                                      init_vol_stds=[0],
                                      default_species_copy_number=self.default_species_copy_number,
                                      default_species_std=0,
                                      submodel_framework='WC:ordinary_differential_equations')
        self.ode_submodel_1 = self.make_ode_submodel(self.mdl_1_spec)
        '''
        # todo: install SBML tests
        test_case = '00001'
        self.sbml_case_00001_file = os.path.join(self.ODE_TEST_CASES, test_case,
                                                 "{}-wc_lang.xlsx".format(test_case))
        self.case_00001_model = Reader().run(self.sbml_case_00001_file, strict=False)
        '''

    def make_ode_submodel(self, model, ode_time_step=1.0, submodel_name='submodel_1'):
        """ Make a MultialgorithmSimulation from a wc lang model """
        # assume a single submodel
        # todo: test concurrent OdeSubmodels, perhaps
        self.ode_time_step = ode_time_step
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, ode_time_step=ode_time_step)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
        simulation_engine.initialize()
        submodel_1 = dynamic_model.dynamic_submodels[submodel_name]
        return submodel_1

    ### test low level methods ###
    def test_ode_submodel_init(self):
        self.assertEqual(self.ode_submodel_1.ode_time_step, self.ode_time_step)

        # test exception
        bad_ode_time_step = -2
        with self.assertRaisesRegexp(MultialgorithmError,
            'ode_time_step must be positive, but is {}'.format(bad_ode_time_step)):
            self.make_ode_submodel(self.mdl_1_spec, ode_time_step=bad_ode_time_step)

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

    def test_compute_population_change_rates(self):
        # test rxn: A[c] ==> B[c] @ 0.04 * B[c]
        # slope of species populations depends linearly on [B[c]]
        # doubling species populations doubles population slopes
        one_rxn_exponential_file = os.path.join(os.path.dirname(__file__), '..', 'fixtures',
                                                 'dynamic_tests', f'one_rxn_exponential.xlsx')
        one_rxn_exp_mdl = read_model_for_test(one_rxn_exponential_file,
                                              integration_framework='WC:ordinary_differential_equations')
        ode_submodel = self.make_ode_submodel(one_rxn_exp_mdl, submodel_name='submodel')
        ode_submodel.current_species_populations()
        new_species_populations = ode_submodel.populations.copy()
        population_change_rates_1 = np.zeros(ode_submodel.num_species)
        time = None
        ode_submodel.compute_population_change_rates(time, new_species_populations,
                                                            population_change_rates_1)
        new_species_populations = 2 * new_species_populations
        population_change_rates_2 = np.zeros(ode_submodel.num_species)
        ode_submodel.compute_population_change_rates(time, new_species_populations,
                                                            population_change_rates_2)
        np.testing.assert_allclose(2 * population_change_rates_1, population_change_rates_2)

        # compute_population_change_rates replaces negative species populations with 0s in the rate computation
        # this produces change rates of 0
        new_species_populations.fill(-1)
        ode_submodel.compute_population_change_rates(time, new_species_populations,
                                                            population_change_rates_2)
        np.testing.assert_array_equal(population_change_rates_2, np.zeros(ode_submodel.num_species))

        # test rxn: [compt_1]: spec_type_0 => spec_type_1 @ k * spec_type_0 / Avogadro / volume_compt_1
        # doubling population also doubles volume, leaving the slopes of species populations unchanged
        population_change_rates_1 = np.zeros(self.ode_submodel_1.num_species)
        new_species_populations = np.full(self.ode_submodel_1.num_species,
                                          self.default_species_copy_number)
        self.ode_submodel_1.compute_population_change_rates(time, new_species_populations,
                                                            population_change_rates_1)
        new_species_populations.fill(2 * self.default_species_copy_number)
        population_change_rates_2 = np.zeros(self.ode_submodel_1.num_species)
        self.ode_submodel_1.compute_population_change_rates(time, new_species_populations,
                                                            population_change_rates_2)
        np.testing.assert_allclose(population_change_rates_1, population_change_rates_2)

    def test_current_species_populations(self):
        self.ode_submodel_1.current_species_populations()
        for pop in self.ode_submodel_1.populations:
            self.assertAlmostEqual(pop, self.default_species_copy_number, delta=0.5)

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
