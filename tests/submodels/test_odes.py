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

from wc_lang.io import Reader
from wc_sim.submodels.odes import OdeSubmodel
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.message_types import RunOde
from wc_sim.testing.make_models import MakeModel
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.dynamic_components import DynamicRateLaw


class TestOdeSubmodel(unittest.TestCase):

    # todo: install
    # ODE_TEST_CASES = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'validation', 'testing', 'semantic')

    def setUp(self):
        self.default_species_copy_number = 100_000_000
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

    def make_ode_submodel(self, model, time_step=1.0):
        """ Make a MultialgorithmSimulation from a wc lang model """
        # assume a single submodel
        # todo: test concurrent OdeSubmodels, perhaps
        self.time_step = time_step
        args = dict(time_step=self.time_step)
        multialgorithm_simulation = MultialgorithmSimulation(model, args)
        simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
        simulation_engine.initialize()
        submodel_1 = dynamic_model.dynamic_submodels['submodel_1']
        return submodel_1

    ### test low level methods ###
    def test_ode_submodel_init(self):
        self.assertEqual(self.ode_submodel_1.time_step, self.time_step)

        # test exception
        bad_time_step = -2
        with self.assertRaisesRegexp(MultialgorithmError,
            'time_step must be positive, but is {}'.format(bad_time_step)):
            self.make_ode_submodel(self.mdl_1_spec, time_step=bad_time_step)

    def test_set_up_optimizations(self):
        ode_submodel = self.ode_submodel_1
        self.assertTrue(set(ode_submodel.ode_species_ids) == ode_submodel.ode_species_ids_set \
            == set(ode_submodel.adjustments.keys()))
        self.assertEqual(ode_submodel.populations.shape, ((len(ode_submodel.ode_species_ids), )))

    # todo: for the next 4 tests, check results against raw properties of self.mdl_1_spec
    def test_solver_lock(self):
        self.ode_submodel_empty = OdeSubmodel('test_1', None, [], [], [], None, 1)
        self.assertTrue(self.ode_submodel_empty.get_solver_lock())
        with self.assertRaisesRegexp(MultialgorithmError, 'OdeSubmodel .*: cannot get_solver_lock'):
            self.ode_submodel_empty.get_solver_lock()
        self.assertTrue(self.ode_submodel_empty.release_solver_lock())

    ### without running the simulator, test solving ###
    def test_set_up_ode_submodel(self):
        self.assertEqual(self.ode_submodel_1, OdeSubmodel.instance)
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
        self.assertTrue(isinstance(self.ode_submodel_1.create_ode_solver(), ode))

    def test_right_hand_side(self):
        pop_change_rates = [0, 0]
        num_reactants = 100
        self.assertEqual(0, self.ode_submodel_1.right_hand_side(0, [num_reactants, 0], pop_change_rates))
        pop_change_rate_spec_1 = pop_change_rates[1]
        self.assertTrue(0 < pop_change_rate_spec_1)

        # test exceptions
        short_list = [1]
        with self.assertRaises(MultialgorithmError):
            self.ode_submodel_1.right_hand_side(0, [num_reactants, 0], short_list)
        self.ode_submodel_1.testing = False
        self.assertEqual(1, self.ode_submodel_1.right_hand_side(0, [num_reactants, 0], short_list))

        '''
        # todo: use if species_populations are copied to the LocalSpeciesPopulation
        # rates decline as reactant converted to product
        for i in range(1, 10):
            self.ode_submodel_1.right_hand_side(0, [num_reactants - i, i], pop_change_rates)
            self.assertTrue(pop_change_rates[1] < pop_change_rate_spec_1)
            pop_change_rate_spec_1 = pop_change_rates[1]
        '''

    def test_current_species_populations(self):
        self.ode_submodel_1.current_species_populations()
        for pop in self.ode_submodel_1.populations:
            self.assertEqual(pop, self.default_species_copy_number)

    def test_run_ode_solver(self):
        # todo: make a test after continuous_adjustment and adjust_continuously are fixed
        self.ode_submodel_1.time += 1
        print(self.ode_submodel_1.local_species_population)
        self.ode_submodel_1.run_ode_solver()

        # make OdeSubmodel.right_hand_side fail
        self.ode_submodel_1.num_species = None
        with self.assertRaises(MultialgorithmError):
            self.ode_submodel_1.run_ode_solver()

    @unittest.skip("todo: needs ODE_TEST_CASES")
    def test_run_ode_solver_2(self):
        '''
        case_00001_ode_submodel = self.make_ode_submodel(self.case_00001_model)
        case_00001_ode_submodel.increment_time_step_count()
        case_00001_ode_submodel.run_ode_solver()
        '''

        case_00001_ode_submodel = self.make_ode_submodel(self.case_00001_model)
        # odes outputs '[CVODE ERROR]  CVode\n  tout too close to t0 to start integration.'
        with self.assertRaisesRegexp(MultialgorithmError, re.escape('solver step() error')):
            case_00001_ode_submodel.run_ode_solver()

    # test event scheduling and handling
    @unittest.skip("not good test")
    def test_schedule_next_ode_analysis(self):
        custom_time_step = 4
        custom_ode_submodel = self.make_ode_submodel(self.mdl_1_spec, time_step=custom_time_step)
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

        # next RunOde event should be at custom_time_step
        custom_ode_submodel.schedule_next_ode_analysis()
        check_next_event(custom_time_step)
