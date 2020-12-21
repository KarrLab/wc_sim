""" Test dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-02-07
:Copyright: 2018, Karr Lab
:License: MIT
"""

from math import log
from scipy.constants import Avogadro
import copy
import math
import numpy
import numpy.testing
import os
import re
import shutil
import tempfile
import timeit
import unittest
import warnings

from de_sim.simulation_config import SimulationConfig
from obj_tables.math.expression import Expression
from wc_lang import (Model, Compartment, Species, Parameter,
                     DistributionInitConcentration,
                     Observable, ObservableExpression, StopCondition,
                     Function, FunctionExpression, InitVolume)
from wc_lang.io import Reader
from wc_onto import onto
from wc_sim.dynamic_components import (SimTokCodes, WcSimToken, DynamicComponent, DynamicExpression,
                                       DynamicModel, DynamicSpecies, DynamicFunction, DynamicParameter,
                                       DynamicCompartment, DynamicStopCondition, DynamicObservable,
                                       DynamicRateLaw,
                                       CacheManager, InvalidationApproaches, CachingEvents)
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.run_results import RunResults
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.simulation import Simulation
from wc_sim.species_populations import LocalSpeciesPopulation, MakeTestLSP
from wc_sim.testing.utils import read_model_for_test, get_expected_dependencies
from wc_utils.util.environ import EnvironUtils, ConfigEnvDict
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.units import unit_registry
import obj_tables
import obj_tables.io
import wc_sim.config
import wc_sim.dynamic_components
import wc_utils.util.ontology


# Almost all machines map Python floats to IEEE-754 64-bit “double precision”, which provides 15 to
# 17 decimal digits. Places for comparing values that should be equal to within the precision of floats
IEEE_64_BIT_FLOATING_POINT_PLACES = 14


class TestInitialDynamicComponentsComprehensively(unittest.TestCase):

    def setUp(self):
        self.model_file = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dynamic_expressions.xlsx')
        self.model = Reader().run(self.model_file)[Model][0]
        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, wc_sim_config)
        _, self.dynamic_model = multialgorithm_simulation.build_simulation(prepare_model=False)

    def test(self):
        # test all DynamicComponents that implement eval()

        ### Test DynamicExpressions ###
        # each one is tested using each of the objects it uses in some instance in self.model_file
        # DynamicFunction
        for id, dynamic_function in self.dynamic_model.dynamic_functions.items():
            expected_value = float(self.model.get_functions(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_function.eval(0), expected_value)
        # test eval_dynamic_functions()
        for func_id, func_val in self.dynamic_model.eval_dynamic_functions(0).items():
            expected_value = float(self.model.get_functions(id=func_id)[0].comments)
            numpy.testing.assert_approx_equal(func_val, expected_value)
        a_func_id = list(self.dynamic_model.dynamic_functions)[0]
        for func_id, func_val in \
            self.dynamic_model.eval_dynamic_functions(0, functions_to_eval=[a_func_id]).items():
            expected_value = float(self.model.get_functions(id=func_id)[0].comments)
            numpy.testing.assert_approx_equal(func_val, expected_value)

        # DynamicStopCondition
        for id, dynamic_stop_condition in self.dynamic_model.dynamic_stop_conditions.items():
            expected_val_in_comment = self.model.get_stop_conditions(id=id)[0].comments
            if expected_val_in_comment == 'True':
                expected_value = True
            elif expected_val_in_comment == 'False':
                expected_value = False
            self.assertEqual(expected_value, dynamic_stop_condition.eval(0))

        # DynamicObservable
        for id, dynamic_observable in self.dynamic_model.dynamic_observables.items():
            expected_value = float(self.model.get_observables(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_observable.eval(0), expected_value)
        # test eval_dynamic_observables()
        for obs_id, obs_val in self.dynamic_model.eval_dynamic_observables(0).items():
            expected_value = float(self.model.get_observables(id=obs_id)[0].comments)
            numpy.testing.assert_approx_equal(obs_val, expected_value)
        an_obs_id = list(self.dynamic_model.dynamic_observables)[0]
        for obs_id, obs_val in \
            self.dynamic_model.eval_dynamic_observables(0, observables_to_eval=[an_obs_id]).items():
            expected_value = float(self.model.get_observables(id=obs_id)[0].comments)
            numpy.testing.assert_approx_equal(obs_val, expected_value)

        # DynamicRateLaw
        for id, dynamic_rate_law in self.dynamic_model.dynamic_rate_laws.items():
            expected_value = float(self.model.get_rate_laws(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_rate_law.eval(0), expected_value)

        ### Test DynamicComponents ###
        # DynamicCompartment
        for id, dynamic_compartment in self.dynamic_model.dynamic_compartments.items():
            expected_value = float(self.model.get_compartments(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_compartment.eval(0), expected_value)

        # DynamicParameter
        for id, dynamic_parameter in self.dynamic_model.dynamic_parameters.items():
            expected_value = float(self.model.get_parameters(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_parameter.eval(0), expected_value)

        # DynamicSpecies
        for id, dynamic_species in self.dynamic_model.dynamic_species.items():
            expected_value = float(self.model.get_species(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_species.eval(0), expected_value)

    def test_get_stop_condition(self):
        all_stop_conditions = self.dynamic_model.get_stop_condition()
        self.assertTrue(callable(all_stop_conditions))
        self.assertTrue(all_stop_conditions(0))

        # teat a dynamic model with no stop conditions
        self.dynamic_model.dynamic_stop_conditions = {}
        self.assertEqual(self.dynamic_model.get_stop_condition(), None)

        # TODO: make a few dynamic models to test all branches of get_stop_condition()
        # a dynamic model with stop conditions that all evaluate false
        # a dynamic model with at least one stop conditions that evaluates true

def make_objects(test_case):
    model = Model()
    objects = {
        Observable: {},
        Parameter: {},
        Function: {},
        StopCondition: {}
    }
    test_case.param_value = 4
    objects[Parameter]['param'] = param = \
        model.parameters.create(id='param', value=test_case.param_value,
                                units=unit_registry.parse_units('dimensionless'))

    test_case.fun_expr = expr = 'param - 2 + max(param, 10)'
    fun1 = Expression.make_obj(model, Function, 'fun1', expr, objects)
    fun2 = Expression.make_obj(model, Function, 'fun2', 'log(2) - param', objects)
    stop_cond = Expression.make_obj(model, StopCondition, 'stop_cond', '0 == 1', objects)

    return model, param, fun1, fun2, stop_cond


def make_dynamic_objects(test_case):
        test_case.init_pop = {}
        test_case.local_species_population = MakeTestLSP(initial_population=test_case.init_pop).local_species_pop
        test_case.model, test_case.parameter, test_case.fun1, test_case.fun2, test_case.stop_cond = make_objects(test_case)

        test_case.dynamic_model = DynamicModel(test_case.model, test_case.local_species_population, {})

        # create a DynamicParameter, two DynamicFunctions, and a DynamicObservable
        test_case.dynamic_objects = dynamic_objects = {}
        dynamic_objects[test_case.parameter] = DynamicParameter(test_case.dynamic_model,
                                                                test_case.local_species_population,
                                                                test_case.parameter, test_case.parameter.value)

        for fun in [test_case.fun1, test_case.fun2]:
            dynamic_objects[fun] = DynamicFunction(test_case.dynamic_model, test_case.local_species_population,
                                                   fun, fun.expression._parsed_expression)

        dynamic_objects[test_case.stop_cond] = DynamicStopCondition(test_case.dynamic_model,
                                                                    test_case.local_species_population,
                                                                    test_case.stop_cond,
                                                                    test_case.stop_cond.expression._parsed_expression)


class TestDynamicComponentAndDynamicExpressions(unittest.TestCase):

    def setUp(self):
        make_dynamic_objects(self)

    def test_get_dynamic_model_type(self):

        self.assertEqual(DynamicComponent.get_dynamic_model_type(Function), DynamicFunction)
        with self.assertRaisesRegex(MultialgorithmError,
                                    "model class of type 'FunctionExpression' not found"):
            DynamicComponent.get_dynamic_model_type(FunctionExpression)

        self.assertEqual(DynamicComponent.get_dynamic_model_type(self.fun1), DynamicFunction)
        expr_model_obj, _ = Expression.make_expression_obj(Function, '11.11', {})
        with self.assertRaisesRegex(MultialgorithmError, "model of type 'FunctionExpression' not found"):
            DynamicComponent.get_dynamic_model_type(expr_model_obj)

        self.assertEqual(DynamicComponent.get_dynamic_model_type('Function'), DynamicFunction)
        with self.assertRaisesRegex(MultialgorithmError, "model type 'NoSuchModel' not defined"):
            DynamicComponent.get_dynamic_model_type('NoSuchModel')
        with self.assertRaisesRegex(MultialgorithmError, "model type '3' has wrong type"):
            DynamicComponent.get_dynamic_model_type(3)
        with self.assertRaisesRegex(MultialgorithmError, "model type 'None' has wrong type"):
            DynamicComponent.get_dynamic_model_type(None)
        with self.assertRaisesRegex(MultialgorithmError, "model of type 'RateLawDirection' not found"):
            DynamicComponent.get_dynamic_model_type('RateLawDirection')

    def test_get_dynamic_component(self):

        self.assertEqual(DynamicComponent.get_dynamic_component(Parameter, 'param'),
                         self.dynamic_objects[self.parameter])
        self.assertEqual(DynamicComponent.get_dynamic_component('Parameter', 'param'),
                         self.dynamic_objects[self.parameter])
        self.assertEqual(DynamicComponent.get_dynamic_component(DynamicParameter, 'param'),
                         self.dynamic_objects[self.parameter])

        class NewDynamicExpression(DynamicComponent): pass
        with self.assertRaisesRegex(MultialgorithmError,
                                    "model type 'NewDynamicExpression' not in DynamicComponent.*"):
            DynamicComponent.get_dynamic_component(NewDynamicExpression, '')
        bad_id = 'no such param'
        with self.assertRaisesRegex(MultialgorithmError,
                                    f"model type '.*' with id='{bad_id}' not in DynamicComponent.*"):
            DynamicComponent.get_dynamic_component(DynamicParameter, bad_id)

    def test_simple_dynamic_expressions(self):
        for dyn_obj in self.dynamic_objects.values():
            cls = dyn_obj.__class__
            self.assertEqual(DynamicComponent.dynamic_components_objs[cls][dyn_obj.id], dyn_obj)

        expected_fun1_wc_sim_tokens = [
            WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
            WcSimToken(SimTokCodes.other, '-2+max('),
            WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
            WcSimToken(SimTokCodes.other, ',10)'),
        ]
        expected_fun1_expr_substring = ['', '-2+max(', '', ',10)']
        expected_fun1_local_ns_key = 'max'
        param_val = str(self.param_value)
        expected_fun1_value = eval(self.fun_expr.replace('param', param_val))

        dynamic_expression = self.dynamic_objects[self.fun1]
        dynamic_expression.prepare()
        self.assertEqual(expected_fun1_wc_sim_tokens, dynamic_expression.wc_sim_tokens)
        self.assertEqual(expected_fun1_expr_substring, dynamic_expression.expr_substrings)
        self.assertTrue(expected_fun1_local_ns_key in dynamic_expression.local_ns)
        self.assertEqual(expected_fun1_value, dynamic_expression.eval(0))
        self.assertIn("id: {}".format(dynamic_expression.id), str(dynamic_expression))
        self.assertIn("type: {}".format(dynamic_expression.__class__.__name__),
                      str(dynamic_expression))
        self.assertIn("expression: {}".format(dynamic_expression.expression), str(dynamic_expression))

        dynamic_expression = self.dynamic_objects[self.fun2]
        dynamic_expression.prepare()
        expected_fun2_wc_sim_tokens = [  # for 'log(2) - param'
            WcSimToken(SimTokCodes.other, 'log(2)-'),
            WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
        ]
        self.assertEqual(expected_fun2_wc_sim_tokens, dynamic_expression.wc_sim_tokens)

    def test_dynamic_expression_errors(self):
        # remove the Function's tokenized result
        self.fun1.expression._parsed_expression._obj_tables_tokens = []
        with self.assertRaisesRegex(MultialgorithmError,
                                    "_obj_tables_tokens cannot be empty - ensure that '.*' is valid"):
            DynamicFunction(self.dynamic_model, self.local_species_population,
                            self.fun1, self.fun1.expression._parsed_expression)

        # remove param from registered dynamic components so fun2 prepare() fails
        del DynamicComponent.dynamic_components_objs[DynamicParameter]['param']
        with self.assertRaisesRegex(MultialgorithmError, 'must be prepared to create'):
            self.dynamic_objects[self.fun2].prepare()

        expr = 'max(1) - 2'
        fun = Expression.make_obj(self.model, Function, 'fun', expr, {}, allow_invalid_objects=True)
        dynamic_function = DynamicFunction(self.dynamic_model, self.local_species_population,
                                           fun, fun.expression._parsed_expression)
        dynamic_function.prepare()
        with self.assertRaisesRegex(MultialgorithmError, re.escape("eval of '{}' raises".format(expr))):
            dynamic_function.eval(1)


class TestDynamics(unittest.TestCase):

    def setUp(self):
        self.objects = objects = {
            Parameter: {},
            Function: {},
            StopCondition: {},
            Observable: {},
            Species: {}
        }

        self.model = model = Model(id='TestDynamics')
        species_types = {}
        st_ids = ['a', 'b']
        for id in st_ids:
            species_types[id] = model.species_types.create(id=id)
        compartments = {}
        comp_ids = ['c1', 'c2']
        for id in comp_ids:
            compartments[id] = model.compartments.create(id=id)
        submodels = {}
        for sm_id, c_id in zip(['submodel1', 'submodel2'], comp_ids):
            submodels[id] = model.submodels.create(id=id)

        for c_id, st_id in zip(comp_ids, st_ids):
            species = model.species.create(species_type=species_types[st_id], compartment=compartments[c_id])
            species.id = species.gen_id()
            objects[Species][species.id] = species
            conc = model.distribution_init_concentrations.create(
                species=species, mean=0, units=unit_registry.parse_units('M'))
            conc.id = conc.gen_id()

        self.init_pop = {
            'a[c1]': 10,
            'b[c2]': 20,
        }

        # map wc_lang object -> expected value
        self.expected_values = expected_values = {}
        objects[Parameter]['param'] = param = \
            model.parameters.create(id='param', value=4,
                                    units=unit_registry.parse_units('dimensionless'))
        objects[Parameter]['molecule_units'] = molecule_units = \
            model.parameters.create(id='molecule_units', value=1.,
            units=unit_registry.parse_units('molecule'))
        expected_values[param] = param.value
        expected_values[molecule_units] = molecule_units.value

        # (wc_lang model type, expression, expected value)
        wc_lang_obj_specs = [
            # just reference param:
            (Function, 'param - 2 + max(param, 10)', 12, 'dimensionless'),
            (StopCondition, '10 < 2 * log10(100) + 2 * param', True, unit_registry.parse_units('dimensionless')),
            # reference other model types:
            (Observable, 'a[c1]', 10, unit_registry.parse_units('molecule')),
            (Observable, '2 * a[c1] - b[c2]', 0, unit_registry.parse_units('molecule')),
            (Function, 'observable_1 + min(observable_2, 10 * molecule_units)', 10,
                unit_registry.parse_units('molecule')),
            (StopCondition, 'observable_1 < param * molecule_units + function_1', True,
                unit_registry.parse_units('dimensionless')),
            # reference same model type:
            (Observable, '3 * observable_1 + b[c2]', 50, unit_registry.parse_units('molecule')),
            (Function, '2 * function_2', 20, unit_registry.parse_units('molecule')),
            (Function, '3 * observable_1 + function_1 * molecule_units', 42,
                unit_registry.parse_units('molecule'))
        ]

        self.expression_models = expression_models = [Function, StopCondition, Observable]
        last_ids = {wc_lang_type: 0 for wc_lang_type in expression_models}

        def make_id(wc_lang_type):
            last_ids[wc_lang_type] += 1
            return "{}_{}".format(wc_lang_type.__name__.lower(), last_ids[wc_lang_type])

        # create wc_lang models
        for model_type, expr, expected_value, units in wc_lang_obj_specs:
            obj_id = make_id(model_type)
            wc_lang_obj = Expression.make_obj(model, model_type, obj_id, expr, objects)
            wc_lang_obj.units = units
            objects[model_type][obj_id] = wc_lang_obj
            expected_values[wc_lang_obj.id] = expected_value

        self.local_species_population = MakeTestLSP(initial_population=self.init_pop).local_species_pop
        self.dynamic_model = DynamicModel(self.model, self.local_species_population, {})

    def test_dynamic_expressions(self):

        # stop caching in this test to measure performance correctly & get correct values for expressions
        self.dynamic_model._stop_caching()

        # check computed value and measure performance of all test Dynamic objects
        executions = 10000
        static_times = []
        for model_cls in (Observable, Function, StopCondition):
            for id, model_obj in self.objects[model_cls].items():
                eval_time = timeit.timeit(stmt='model_obj.expression._parsed_expression.test_eval()',
                    number=executions, globals=locals())
                static_times.append((eval_time * 1e6 / executions, id, model_obj.expression.expression))

        dynamic_times = []
        for dynamic_obj_dict in [self.dynamic_model.dynamic_observables,
                                 self.dynamic_model.dynamic_functions,
                                 self.dynamic_model.dynamic_stop_conditions]:
            for id, dynamic_expression in dynamic_obj_dict.items():
                self.assertEqual(self.expected_values[id], dynamic_expression.eval(0))
                eval_time = timeit.timeit(stmt='dynamic_expression.eval(0)', number=executions,
                                          globals=locals())
                dynamic_times.append((eval_time * 1e6 / executions, dynamic_expression.id,
                    dynamic_expression.expression))

        print("\nExpression performance (usec/eval) for {} evals:".format(executions))
        tab_width_1 = 8
        tab_width_2 = 18
        print("{}\t{}\t".format('Static', 'Dynamic').expandtabs(tab_width_1),
            "{}\t{}".format('id', 'Expression').expandtabs(tab_width_2))
        for (static_time, id, expression), (dynamic_time, _, _) in zip(static_times, dynamic_times):
            print("{:5.1f}\t{:5.1f}\t".format(static_time, dynamic_time).expandtabs(tab_width_1),
                "{}\t{}".format(id, expression).expandtabs(tab_width_2))

    def test_dynamic_compartments(self):
        for dynamic_components in [self.dynamic_model.dynamic_species,
                                  self.dynamic_model.dynamic_parameters]:
            for dynamic_component in dynamic_components.values():
                self.assertIn("id: {}".format(dynamic_component.id), str(dynamic_component))
                self.assertIn("type: {}".format(dynamic_component.__class__.__name__),
                              str(dynamic_component))

        self.assertEqual(DynamicComponent.get_dynamic_model_type(Parameter), DynamicParameter)
        self.assertEqual(DynamicComponent.get_dynamic_model_type(Species), DynamicSpecies)


class TestUninitializedDynamicCompartment(unittest.TestCase):
    """ Test DynamicCompartments that do not have a species population
    """

    def setUp(self):

        # make a Compartment & use it to make a DynamicCompartment
        comp_id = 'comp_id'
        self.mean_init_volume = 1E-17
        self.compartment = Compartment(id=comp_id, name='name',
                                       init_volume=InitVolume(mean=self.mean_init_volume,
                                       std=self.mean_init_volume / 40.))
        self.compartment.init_density = Parameter(id='density_{}'.format(comp_id), value=1100.,
                                                  units=unit_registry.parse_units('g l^-1'))

        self.random_state = RandomStateManager.instance()
        self.dynamic_compartment = DynamicCompartment(None, self.random_state, self.compartment)

        self.abstract_compartment = Compartment(id=comp_id, name='name',
                                                init_volume=InitVolume(mean=self.mean_init_volume, std=0),
                                                physical_type=onto['WC:abstract_compartment'])
        self.abstract_dynamic_compartment = DynamicCompartment(None, self.random_state,
                                                               self.abstract_compartment)

    def test_dynamic_compartment(self):
        volumes = []
        # test mean initial volume
        for i_trial in range(10):
            dynamic_compartment = DynamicCompartment(None, self.random_state, self.compartment)
            volumes.append(dynamic_compartment.init_volume)
        self.assertLess(numpy.abs((numpy.mean(volumes) - self.mean_init_volume) / self.mean_init_volume), 0.1)

    def test_abstract_dynamic_compartment(self):
        self.assertTrue(self.abstract_dynamic_compartment._is_abstract())
        self.assertFalse(hasattr(self.abstract_dynamic_compartment, 'init_density'))

    def test_dynamic_compartment_exceptions(self):
        compartment = Compartment(id='id', name='name', init_volume=InitVolume(mean=0))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.random_state, compartment)

        self.compartment.init_density = Parameter(id='density_{}'.format(self.compartment.id),
                                                  value=float('nan'),
                                                  units=unit_registry.parse_units('g l^-1'))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.random_state, self.compartment)

        self.compartment.init_density = Parameter(id='density_{}'.format(self.compartment.id),
                                                value=0., units=unit_registry.parse_units('g l^-1'))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.random_state, self.compartment)

        self.compartment.init_volume = None
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.random_state, self.compartment)


class TestInitializedDynamicCompartment(unittest.TestCase):
    """ Test DynamicCompartments with species populations
    """

    def setUp(self):
        comp_id = 'comp_id'

        # make a LocalSpeciesPopulation
        self.num_species = 100
        species_nums = list(range(0, self.num_species))
        self.species_ids = list(map(lambda x: "species_{}[{}]".format(x, comp_id), species_nums))
        self.all_pops = 1E6
        self.init_populations = dict(zip(self.species_ids, [self.all_pops]*len(species_nums)))
        self.all_m_weights = 50
        self.molecular_weights = dict(zip(self.species_ids, [self.all_m_weights]*len(species_nums)))
        self.local_species_pop = LocalSpeciesPopulation('test', self.init_populations, self.molecular_weights,
                                                        random_state=RandomStateManager.instance())
        model, _, _, _, _ = make_objects(self)
        self.dynamic_model = DynamicModel(model, self.local_species_pop, {})

        # make Compartments & use them to make a DynamicCompartments
        self.mean_init_volume = 1E-17
        self.compartment = Compartment(id=comp_id, name='name',
                                       init_volume=InitVolume(mean=self.mean_init_volume,
                                       std=self.mean_init_volume / 40.))
        self.compartment.init_density = Parameter(id='density_{}'.format(comp_id), value=1100.,
                                                  units=unit_registry.parse_units('g l^-1'))
        self.random_state = RandomStateManager.instance()
        self.dynamic_compartment = DynamicCompartment(self.dynamic_model, self.random_state, self.compartment,
                                                      self.species_ids)

        self.abstract_compartment = Compartment(id=comp_id, name='name',
                                                init_volume=InitVolume(mean=self.mean_init_volume, std=0),
                                                physical_type=onto['WC:abstract_compartment'])
        self.abstract_dynamic_compartment = DynamicCompartment(self.dynamic_model, self.random_state,
                                                               self.abstract_compartment)

    def specify_init_accounted_fraction(self, desired_init_accounted_fraction):
        # make a DynamicCompartment with accounted_fraction ~= desired_init_accounted_fraction
        # without changing init_accounted_mass or init_volume
        init_density = self.dynamic_compartment.init_accounted_mass / \
            (desired_init_accounted_fraction * self.dynamic_compartment.init_volume)
        self.compartment.init_density = Parameter(id='density_x', value=init_density,
            units=unit_registry.parse_units('g l^-1'))
        return DynamicCompartment(self.dynamic_model, self.random_state, self.compartment)

    def test_initialize_mass_and_density(self):
        self.dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        estimated_accounted_mass = self.num_species * self.all_pops * self.all_m_weights / Avogadro
        self.assertAlmostEqual(self.dynamic_compartment.init_accounted_mass, estimated_accounted_mass,
                               places=IEEE_64_BIT_FLOATING_POINT_PLACES)
        self.assertTrue(0 < self.dynamic_compartment.init_mass)
        self.assertAlmostEqual(self.dynamic_compartment.accounted_fraction,
                               self.dynamic_compartment.init_accounted_mass / \
                                self.dynamic_compartment.init_mass,
                               places=IEEE_64_BIT_FLOATING_POINT_PLACES)

        # test abstract compartment
        self.abstract_dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        self.assertAlmostEqual(self.abstract_dynamic_compartment.init_accounted_mass,
                               estimated_accounted_mass,
                               places=IEEE_64_BIT_FLOATING_POINT_PLACES)
        self.assertEqual(self.abstract_dynamic_compartment.init_accounted_mass,
                         self.abstract_dynamic_compartment.init_mass)
        self.assertFalse(hasattr(self.abstract_dynamic_compartment, 'init_accounted_density'))

        empty_local_species_pop = LocalSpeciesPopulation('test', {}, {},
                                                         random_state=RandomStateManager.instance())
        dynamic_compartment = DynamicCompartment(self.dynamic_model, self.random_state, self.compartment)

        # stop caching to ignore any cached value for DynamicCompartment.accounted_mass()
        self.dynamic_model.cache_manager._stop_caching()
        with self.assertRaisesRegex(MultialgorithmError, "initial accounted ratio is 0"):
            dynamic_compartment.initialize_mass_and_density(empty_local_species_pop)

        config_multialgorithm = wc_sim.config.core.get_config()['wc_sim']['multialgorithm']
        MAX_ALLOWED_INIT_ACCOUNTED_FRACTION = config_multialgorithm['max_allowed_init_accounted_fraction']
        dynamic_compartment = \
            self.specify_init_accounted_fraction((MAX_ALLOWED_INIT_ACCOUNTED_FRACTION + 1)/2)
        with warnings.catch_warnings(record=True) as w:
            dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
            self.assertIn("initial accounted ratio (", str(w[-1].message))
            self.assertIn(") greater than 1.0", str(w[-1].message))

        dynamic_compartment = self.specify_init_accounted_fraction(4)
        with self.assertRaises(MultialgorithmError):
            dynamic_compartment.initialize_mass_and_density(self.local_species_pop)

    def test_fold_changes(self):
        # ensure that initial fold changes == 1
        self.dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        self.assertAlmostEqual(self.dynamic_compartment.fold_change_total_mass(0), 1.0)
        self.assertAlmostEqual(self.dynamic_compartment.fold_change_total_volume(0), 1.0)

    def test_init_volume_and_eval(self):
        self.dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        self.assertAlmostEqual(self.dynamic_compartment.volume(), self.dynamic_compartment.init_volume,
            places=IEEE_64_BIT_FLOATING_POINT_PLACES)
        self.assertAlmostEqual(self.dynamic_compartment.eval(), self.dynamic_compartment.init_mass,
            places=IEEE_64_BIT_FLOATING_POINT_PLACES)

        # test abstract compartment
        self.abstract_dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        self.assertEqual(self.abstract_dynamic_compartment.accounted_mass(time=0),
                         self.abstract_dynamic_compartment.init_mass)
        self.assertEqual(self.abstract_dynamic_compartment.accounted_volume(time=0),
                         self.abstract_dynamic_compartment.init_volume)
        self.assertEqual(self.abstract_dynamic_compartment.mass(time=0),
                         self.abstract_dynamic_compartment.init_mass)
        self.assertEqual(self.abstract_dynamic_compartment.volume(time=0),
                         self.abstract_dynamic_compartment.init_volume)

    def test_str(self):
        dynamic_compartment = DynamicCompartment(self.dynamic_model, self.random_state, self.compartment)
        self.assertIn("has not been initialized", str(dynamic_compartment))
        self.assertIn(self.dynamic_compartment.id, str(dynamic_compartment))

        dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        self.assertIn("has been initialized", str(dynamic_compartment))
        self.assertIn('Fold change total mass: 1.0', str(dynamic_compartment))

        # test abstract compartment
        self.abstract_dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        self.assertNotIn('Fraction of mass accounted for', str(self.abstract_dynamic_compartment))


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
    DEPENDENCIES_MDL_FILE = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dependencies.xlsx')

    @classmethod
    def setUpClass(cls):
        cls.models = {}
        for model_file in [cls.MODEL_FILENAME, cls.DEPENDENCIES_MDL_FILE]:
            cls.models[model_file] = Reader().run(model_file, ignore_extra_models=True)[Model][0]

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def make_dynamic_model(self, model_filename):
        # read and initialize a model
        model = TestDynamicModel.models[model_filename]
        return self._make_dynamic_model(model)

    def _make_dynamic_model(self, model):
        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, ode_time_step=2, dfba_time_step=5)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        _, dynamic_model = multialgorithm_simulation.build_simulation()
        return model, dynamic_model

    def compute_expected_actual_masses(self, model_filename):
        # provide the expected actual masses for the compartments in model_filename, keyed by compartment id
        model = TestDynamicModel.models[model_filename]
        expected_actual_masses = {}
        for compartment in model.get_compartments():
            expected_actual_masses[compartment.id] = \
                compartment.init_volume.mean * compartment.init_density.value
        return expected_actual_masses

    def compare_aggregate_states(self, expected_aggregate_state, computed_aggregate_state, frac_diff=1e-1):
        list_of_nested_keys_to_test = [
            ['cell mass'],
            ['compartments', 'c', 'mass'],
        ]
        for nested_keys_to_test in list_of_nested_keys_to_test:
            expected = expected_aggregate_state
            computed = computed_aggregate_state
            for key in nested_keys_to_test:
                expected = expected[key]
                computed = computed[key]
            numpy.testing.assert_approx_equal(expected, computed, significant=1)

    def test_dynamic_model(self):
        model, dynamic_model = self.make_dynamic_model(self.MODEL_FILENAME)
        self.assertEqual(len(dynamic_model.cellular_dyn_compartments), 1)
        self.assertEqual(dynamic_model.cellular_dyn_compartments[0].id, 'c')
        self.assertEqual(dynamic_model.get_num_submodels(), 2)

        for compartment in model.get_compartments():
            compartment.biological_type = onto['WC:extracellular_compartment']
        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        multialgorithm_simulation.initialize_components()
        with self.assertRaisesRegex(MultialgorithmError, 'must have at least 1 cellular compartment'):
            DynamicModel(model, multialgorithm_simulation.local_species_population,
                         multialgorithm_simulation.temp_dynamic_compartments)

    def test_obtain_dependencies(self):
        def sub_dependencies(dependencies, model_type):
            """ Extract the dependencies for models of type model_type """
            rv = {}
            for rxn, models in dependencies.items():
                rv[rxn.id] = set()
                for model in models:
                    if type(model) == model_type:
                        rv[rxn.id].add(model.id)
            return rv

        _, dynamic_model = self.make_dynamic_model(self.DEPENDENCIES_MDL_FILE)
        dependencies = dynamic_model.obtain_dependencies(self.models[self.DEPENDENCIES_MDL_FILE])
        self.assertEqual(dynamic_model.rxn_expression_dependencies, dependencies)

        # remove rxns with no dependencies from expected_dependencies, like obtain_dependencies() does
        # reaction_10 has no dependencies
        expected_dependencies = get_expected_dependencies()
        for model_type in expected_dependencies:
            del expected_dependencies[model_type]['reaction_10']

        for model_type in (DynamicFunction, DynamicObservable, DynamicRateLaw, DynamicStopCondition):
            self.assertEqual(sub_dependencies(dependencies, model_type), expected_dependencies[model_type.__name__])

    @unittest.skip("requires about 5 GB RAM and takes 0.5 hr.")
    def test_large_obtain_dependencies(self):
        # test obtain_dependencies() on a large model
        H1_MODEL_SCALED_DOWN = os.path.join(os.path.dirname(__file__), 'fixtures', 'h1_scaled_down_model_core.xlsx')
        # would like to set validate = True, and validate_element_charge_balance = False in wc_lang config 'cuz
        # some H1_MODEL_SCALED_DOWN rxns aren't balanced
        # but validate takes a REALLY long time
        h1_model = Reader().run(H1_MODEL_SCALED_DOWN, validate=False)[Model][0]

        # to speed up reading model would like to save a 'compiled' model in a fixture file, like a pickle file, but pickle fails
        # scaled_down_model contains compartments with large initial accounted ratios
        with EnvironUtils.temp_config_env([(['wc_sim', 'multialgorithm', 'max_allowed_init_accounted_fraction'], '1E3')]):
            _, dynamic_model = self._make_dynamic_model(h1_model)
            self.assertTrue(isinstance(dynamic_model.rxn_expression_dependencies, dict))

    def test_continuous_reaction_dependencies(self):
        _, dynamic_model = self.make_dynamic_model(self.DEPENDENCIES_MDL_FILE)

        ode_submodel = dynamic_model.dynamic_submodels['ode_submodel']
        ode_reactions = [rxn.id for rxn in ode_submodel.reactions]
        expected_continous_rxn_dependencies = {'ode_submodel': set()}
        expected_dependencies = get_expected_dependencies()
        for expr_type_name, dependencies in expected_dependencies.items():
            for rxn_id, dependent_ids in dependencies.items():
                if rxn_id in ode_reactions:
                    for dependent_id in dependent_ids:
                        dyn_model_type = getattr(wc_sim.dynamic_components, expr_type_name)
                        dyn_model = DynamicComponent.get_dynamic_component(dyn_model_type, dependent_id)
                        expected_continous_rxn_dependencies['ode_submodel'].add(dyn_model)
        # compare sets
        continuous_rxn_dependencies = dynamic_model.continuous_reaction_dependencies()
        continuous_rxn_dependencies['ode_submodel'] = set(continuous_rxn_dependencies['ode_submodel'])
        self.assertEqual(continuous_rxn_dependencies, expected_continous_rxn_dependencies)

    def test_flush_operations(self):
        NOT_FOUND = 'not found'
        def get_cached_masses(dynamic_model):
            values = {}
            for compt in dynamic_model.dynamic_compartments.values():
                try:
                    values[compt] = dynamic_model.cache_manager.get(compt)
                except MultialgorithmError as e:
                    values[compt] = NOT_FOUND
            return values

        def eval_rate_laws_in_submodel(dynamic_model, submodel_id):
            """ Evaluate some expressions """
            dsa_submodel = dynamic_model.dynamic_submodels[submodel_id]
            dsa_submodel.calc_reaction_rates()

        dsa_submodel_id = 'dsa_submodel'
        ode_submodel_id = 'ode_submodel'


        ### test NO CACHING ###
        # all flush methods do nothing
        with EnvironUtils.temp_config_env([(['wc_sim', 'multialgorithm', 'expression_caching'], 'False')]):
            model, dynamic_model = self.make_dynamic_model(self.DEPENDENCIES_MDL_FILE)
            cached_masses = get_cached_masses(dynamic_model)
            dynamic_model.flush_compartment_masses()
            self.assertEqual(cached_masses, get_cached_masses(dynamic_model))

            reaction = model.get_reactions(id='reaction_1')[0]
            dynamic_model.flush_after_reaction(reaction)
            self.assertTrue(dynamic_model.cache_manager.empty())
            self.assertTrue(dynamic_model.cache_manager.empty())

            dynamic_model.continuous_submodel_flush_after_populations_change(ode_submodel_id)
            self.assertTrue(dynamic_model.cache_manager.empty())


        ### test EVENT_BASED invalidation ###
        with EnvironUtils.temp_config_env(((['wc_sim', 'multialgorithm', 'expression_caching'], 'True'),
                                           (['wc_sim', 'multialgorithm', 'cache_invalidation'], 'event_based'))):
            model, dynamic_model = self.make_dynamic_model(self.DEPENDENCIES_MDL_FILE)
            eval_rate_laws_in_submodel(dynamic_model, dsa_submodel_id)

            # when using EVENT_BASED invalidation, flush_compartment_masses does nothing
            cached_masses = get_cached_masses(dynamic_model)
            dynamic_model.flush_compartment_masses()
            self.assertEqual(cached_masses, get_cached_masses(dynamic_model))

            # when using EVENT_BASED invalidation, flush_after_reaction empties cache
            reaction = model.get_reactions(id='reaction_1')[0]
            dynamic_model.flush_after_reaction(reaction)
            self.assertTrue(dynamic_model.cache_manager.empty())

            # when using EVENT_BASED invalidation, continuous_submodel_flush_after_populations_change empties cache
            eval_rate_laws_in_submodel(dynamic_model, ode_submodel_id)
            dynamic_model.continuous_submodel_flush_after_populations_change(ode_submodel_id)
            self.assertTrue(dynamic_model.cache_manager.empty())


        ### test REACTION_DEPENDENCY_BASED invalidation ###
        with EnvironUtils.temp_config_env(((['wc_sim', 'multialgorithm', 'expression_caching'], 'True'),
                                           (['wc_sim', 'multialgorithm', 'cache_invalidation'],
                                            'reaction_dependency_based'))):
            model, dynamic_model = self.make_dynamic_model(self.DEPENDENCIES_MDL_FILE)
            eval_rate_laws_in_submodel(dynamic_model, dsa_submodel_id)

            # when using REACTION_DEPENDENCY_BASED invalidation,
            # flush_compartment_masses deletes all compartment mass cache entries
            cached_masses = get_cached_masses(dynamic_model)
            for key in cached_masses:
                cached_masses[key] = NOT_FOUND
            dynamic_model.flush_compartment_masses()
            self.assertEqual(cached_masses, get_cached_masses(dynamic_model))

            # when using REACTION_DEPENDENCY_BASED invalidation, flush_after_reaction flushes dependent expressions
            # reaction_10 has no dependencies
            reaction = model.get_reactions(id='reaction_10')[0]
            cache_copy = copy.copy(dynamic_model.cache_manager._cache)
            dynamic_model.flush_after_reaction(reaction)
            self.assertEqual(dynamic_model.cache_manager._cache, cache_copy)

        def get_expected_rxn_dependencies(reaction_ids):
            """ Get dynamic expressions that are known to depend on reactions
            """
            expected_dependencies = get_expected_dependencies()
            dependent_exprs = set()
            for expr_type_name, dependencies in expected_dependencies.items():
                for rxn_id in reaction_ids:
                    for dependent_id in dependencies[rxn_id]:
                        dyn_model_type = getattr(wc_sim.dynamic_components, expr_type_name)
                        dyn_model = DynamicComponent.get_dynamic_component(dyn_model_type, dependent_id)
                        dependent_exprs.add(dyn_model)
            return dependent_exprs

        rxn_id = 'reaction_1'
        self.assertLess(0, len([expr for expr in get_expected_rxn_dependencies([rxn_id])\
                                if expr in dynamic_model.cache_manager]))
        reaction = model.get_reactions(id=rxn_id)[0]
        dynamic_model.flush_after_reaction(reaction)
        for expression in get_expected_rxn_dependencies([rxn_id]):
            self.assertNotIn(expression, dynamic_model.cache_manager)

        # when using REACTION_DEPENDENCY_BASED invalidation, continuous_submodel_flush_after_populations_change
        # flushes expressions that depend on the continuous submodel calling it
        eval_rate_laws_in_submodel(dynamic_model, ode_submodel_id)
        reaction_ids = ['reaction_3', 'reaction_8']
        self.assertLess(0, len([expr for expr in get_expected_rxn_dependencies(reaction_ids)\
                                if expr in dynamic_model.cache_manager]))
        expected_rxn_dependencies = get_expected_rxn_dependencies(reaction_ids)
        dynamic_model.continuous_submodel_flush_after_populations_change(ode_submodel_id)
        for expression in expected_rxn_dependencies:
            self.assertNotIn(expression, dynamic_model.cache_manager)

    def do_test_expression_dependency_dynamics(self, model_file, framework, max_time,
                                               alternative_caching_settings, seed=17):

        ### test with caching specified by alternative_caching_settings ###

        # must suspend rounding by DynamicSpeciesState.get_population() because DynamicSpeciesStates
        # and submodels share a RandomState, so if rounding is used it changes stochastic algorithms
        # todo: remove this suspention of rounding when submodels and the Local Species Populattion
        # use different RandomStates
        with EnvironUtils.temp_config_env([(['wc_sim', 'multialgorithm', 'default_rounding'], 'False')]):
            model = Reader().run(model_file)[Model][0]
            # change DSA submodel to another framework
            for submodel in model.submodels:
                if wc_utils.util.ontology.are_terms_equivalent(submodel.framework,
                                                               onto['WC:deterministic_simulation_algorithm']):
                    submodel.framework = onto[framework]
            simulation = Simulation(model)
            results_dir = tempfile.mkdtemp(dir=self.test_dir)
            kwargs = dict(max_time=max_time, results_dir=results_dir, checkpoint_period=1, seed=seed,
                          ode_time_step=1, progress_bar=False, verbose=False)
            with EnvironUtils.temp_config_env([(['wc_sim', 'multialgorithm', 'expression_caching'], 'False')]):
                run_results_no_caching = RunResults(simulation.run(**kwargs).results_dir)

            for caching_settings in alternative_caching_settings:
                config_env_dict = ConfigEnvDict()
                for caching_attr, caching_setting in caching_settings:
                    config_var_path = ['wc_sim', 'multialgorithm']
                    config_var_path.append(caching_attr)
                    config_env_dict.add_config_value(config_var_path, caching_setting)
                with EnvironUtils.make_temp_environ(**config_env_dict.get_env_dict()):
                    kwargs['results_dir'] = tempfile.mkdtemp(dir=self.test_dir)
                    run_results_caching = RunResults(simulation.run(**kwargs).results_dir)
                    self.assertTrue(run_results_no_caching.semantically_equal(run_results_caching, debug=True))

    def test_expression_dependency_dynamics(self):
        # ensure that models with expressions that depend on reactions generates
        # the same results with and without caching
        alternative_caching_settings = ([('expression_caching', 'True'),
                                         ('cache_invalidation', 'event_based')],
                                        [('expression_caching', 'True'),
                                         ('cache_invalidation', 'reaction_dependency_based')])

        # test ODE with DSA, SSA and NRM
        frameworks = ('WC:deterministic_simulation_algorithm', 'WC:stochastic_simulation_algorithm',
                      'WC:next_reaction_method',)
        for framework in frameworks:
            self.do_test_expression_dependency_dynamics(self.DEPENDENCIES_MDL_FILE, framework,
                                                        2, alternative_caching_settings)

    def test_agregate_properties(self):
        # test aggregate properties like mass and volume against independent calculations of their values
        # calculations made in the model's spreadsheet

        # read model while ignoring missing models
        model = read_model_for_test(self.MODEL_FILENAME)
        # create dynamic model
        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, submodels_to_skip=['submodel_1'])
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        _, dynamic_model = multialgorithm_simulation.build_simulation()

        # a Model to store expected initial values
        class ExpectedInitialValue(obj_tables.Model):
            component = obj_tables.StringAttribute()
            attribute = obj_tables.StringAttribute()
            expected_initial_value = obj_tables.FloatAttribute()
            comment = obj_tables.StringAttribute()
            class Meta(obj_tables.Model.Meta):
                attribute_order = ('component', 'attribute', 'expected_initial_value', 'comment')

        # get calculations of expected initial values from the workbook
        expected_initial_values = \
            obj_tables.io.Reader().run(self.MODEL_FILENAME, models=[ExpectedInitialValue],
                                       ignore_extra_models=True)[ExpectedInitialValue]
        for cellular_compartment in dynamic_model.cellular_dyn_compartments:
            compartment = dynamic_model.dynamic_compartments[cellular_compartment.id]
            actual_values = {
                'mass': compartment.mass(),
                'volume': compartment.volume(),
                'accounted mass': compartment.accounted_mass(),
                'accounted volume': compartment.accounted_volume()}
            for expected_initial_value in expected_initial_values:
                if expected_initial_value.component == cellular_compartment.id:
                    expected_value = expected_initial_value.expected_initial_value
                    actual_value = actual_values[expected_initial_value.attribute]
                    numpy.testing.assert_approx_equal(actual_value, expected_value)

        # cell mass, cell volume, etc.
        actual_values = {
            'cell mass': dynamic_model.cell_mass(),
            'cell volume': dynamic_model.cell_volume(),
            'cell accounted mass': dynamic_model.cell_accounted_mass(),
            'cell accounted volume': dynamic_model.cell_accounted_volume()}
        for expected_initial_value in expected_initial_values:
            if expected_initial_value.component == 'whole_cell':
                expected_value = expected_initial_value.expected_initial_value
                actual_value = actual_values[f"cell {expected_initial_value.attribute}"]
                numpy.testing.assert_approx_equal(actual_value, expected_value)

        # test dynamic_model.get_aggregate_state()
        aggregate_state = dynamic_model.get_aggregate_state()
        for eiv_record in expected_initial_values:
            expected_value = eiv_record.expected_initial_value
            if eiv_record.component == 'whole_cell':
                actual_value = aggregate_state[f"cell {eiv_record.attribute}"]
                numpy.testing.assert_approx_equal(actual_value, expected_value)
            else:
                actual_value = aggregate_state['compartments'][eiv_record.component][eiv_record.attribute]
                numpy.testing.assert_approx_equal(actual_value, expected_value)

    def test_cache_settings(self):
        _, dynamic_model = self.make_dynamic_model(self.MODEL_FILENAME)
        dynamic_model._stop_caching()
        self.assertEqual(dynamic_model.cache_manager.caching(), False)
        dynamic_model._start_caching()
        self.assertEqual(dynamic_model.cache_manager.caching(), True)


class TestCacheManager(unittest.TestCase):

    def setUp(self):
        make_dynamic_objects(self)
        self.dyn_function = self.dynamic_objects[self.fun1]
        self.dyn_function_2 = self.dynamic_objects[self.fun2]
        self.dyn_stop_cond = self.dynamic_objects[self.stop_cond]

    def test(self):
        ### test arguments
        cache_manager = CacheManager(caching_active=False)
        self.assertEqual(cache_manager.caching(), False)

        for cache_invalidation in ('reaction_dependency_based', 'event_based'):
            cache_manager = CacheManager(caching_active=True, cache_invalidation=cache_invalidation)
            self.assertEqual(cache_manager.caching(), True)
            self.assertTrue(isinstance(cache_manager.cache_invalidation, InvalidationApproaches))
            self.assertEqual(cache_manager.invalidation_approach(), cache_manager.cache_invalidation)

        ### test no caching
        cache_manager = CacheManager(caching_active=False)
        self.assertEqual(cache_manager.caching(), False)
        self.assertEqual(cache_manager.get(self.dyn_stop_cond), None)
        self.assertEqual(cache_manager.set(self.dyn_stop_cond, 0), None)
        self.assertNotIn(self.dyn_stop_cond, cache_manager)
        self.assertEqual(cache_manager.flush([]), None)
        self.assertEqual(cache_manager.clear_cache(), None)
        self.assertEqual(cache_manager.invalidate(), None)

        ### test utilities
        cache_manager = CacheManager(caching_active=True)
        self.assertEqual(cache_manager.size(), 0)
        self.assertTrue(cache_manager.empty())
        cache_manager.set(self.dyn_stop_cond, 0)
        self.assertIn(self.dyn_stop_cond, cache_manager)
        self.assertEqual(cache_manager.size(), 1)
        self.assertFalse(cache_manager.empty())

        def test_get_set(test_case, cache_manager):
            # test 'set' and 'get', which don't depend on the invalidation approach
            expressions = []
            v = 3
            cache_manager.set(test_case.dyn_function, v)
            expressions.append(test_case.dyn_function)
            test_case.assertEqual(cache_manager.get(test_case.dyn_function), v)
            v = 7
            cache_manager.set(test_case.dyn_function, v)
            expressions.append(test_case.dyn_function)
            test_case.assertEqual(cache_manager.get(test_case.dyn_function), v)
            v = 44
            cache_manager.set(self.dyn_stop_cond, v)
            expressions.append(self.dyn_stop_cond)
            test_case.assertEqual(cache_manager.get(self.dyn_stop_cond), v)

            with test_case.assertRaisesRegex(MultialgorithmError, 'dynamic expression .* not in cache'):
                cache_manager.get(self.dyn_function_2)

            return expressions

        def test_cache_settings(test_case, cache_manager):
            for b in [False, True]:
                cache_manager.set_caching(b)
                test_case.assertEqual(cache_manager.caching(), b)
            cache_manager._stop_caching()
            test_case.assertEqual(cache_manager.caching(), False)
            cache_manager._start_caching()
            test_case.assertEqual(cache_manager.caching(), True)

        ### test flush
        cache_manager = CacheManager(caching_active=True, cache_invalidation='event_based')
        self.assertEqual(cache_manager.caching(), True)
        self.assertEqual(cache_manager.cache_invalidation, InvalidationApproaches.EVENT_BASED)
        expressions = test_get_set(self, cache_manager)
        # flush should empty the cache
        cache_manager.flush(expressions)
        self.assertTrue(cache_manager.empty())

        # flush should do nothing with keys not in the cache
        cache_manager = CacheManager(caching_active=True, cache_invalidation='event_based')
        cache_manager.set(self.dyn_function, 1)
        cache_copy = copy.copy(cache_manager._cache)
        expressions_not_in_cache = [self.dyn_stop_cond]
        cache_manager.flush(expressions_not_in_cache)
        self.assertEqual(cache_manager._cache, cache_copy)

        # clear_cache should empty the cache
        cache_manager.clear_cache()
        self.assertTrue(cache_manager.empty())
        test_cache_settings(self, cache_manager)

        ### test event_based caching
        cache_manager = CacheManager(caching_active=True, cache_invalidation='event_based')
        test_get_set(self, cache_manager)
        # invalidate should empty the cache
        cache_manager.invalidate()
        self.assertTrue(cache_manager.empty())

        # test reaction_dependency_based caching
        cache_manager = CacheManager(caching_active=True, cache_invalidation='reaction_dependency_based')
        self.assertEqual(cache_manager.cache_invalidation, InvalidationApproaches.REACTION_DEPENDENCY_BASED)
        expressions = test_get_set(self, cache_manager)

        # with no keys, invalidate should do nothing
        cache_copy = copy.copy(cache_manager._cache)
        cache_manager.invalidate()
        self.assertEqual(cache_manager._cache, cache_copy)

        # with keys in the cache, invalidate should empty it
        cache_manager.invalidate(expressions)
        self.assertTrue(cache_manager.empty())

        # test caching configured from config file
        cache_manager = CacheManager()
        self.assertEqual(cache_manager._cache, dict())
        self.assertTrue(isinstance(cache_manager.caching(), bool))

        # no caching
        with EnvironUtils.temp_config_env([(['wc_sim', 'multialgorithm', 'expression_caching'], 'False')]):
            cache_manager = CacheManager()
            self.assertEqual(cache_manager.caching(), False)

        # bad cache_invalidation
        with EnvironUtils.temp_config_env(((['wc_sim', 'multialgorithm', 'expression_caching'], 'True'),
                                           (['wc_sim', 'multialgorithm', 'cache_invalidation'], "'invalid'"))):
            with self.assertRaisesRegex(MultialgorithmError, "cache_invalidation .* not in"):
                CacheManager()

    def test_caching_stats(self):
        cache_manager = CacheManager(caching_active=True, cache_invalidation='reaction_dependency_based')
        with self.assertRaises(MultialgorithmError):
            # MISS
            cache_manager.get(self.dyn_function)
        cache_manager.set(self.dyn_function, 1)
        # HIT
        cache_manager.get(self.dyn_function)
        # FLUSH_HIT
        cache_manager.flush([self.dyn_function])
        # FLUSH_MISS
        cache_manager.flush([self.dyn_function])

        cache_manager._add_hit_ratios()
        for expression_name, stats in cache_manager._cache_stats.items():
            if expression_name == DynamicFunction.__name__:
                self.assertEqual(stats[CachingEvents.HIT], 1)
                self.assertEqual(stats[CachingEvents.MISS], 1)
                self.assertEqual(stats[CachingEvents.FLUSH_HIT], 1)
                self.assertEqual(stats[CachingEvents.FLUSH_MISS], 1)
                self.assertEqual(stats[CachingEvents.HIT_RATIO], 1/2)
                self.assertEqual(stats[CachingEvents.FLUSH_HIT_RATIO], 1/2)
            else:
                self.assertEqual(stats[CachingEvents.HIT], 0)
                self.assertEqual(stats[CachingEvents.MISS], 0)
                self.assertEqual(stats[CachingEvents.FLUSH_HIT], 0)
                self.assertEqual(stats[CachingEvents.FLUSH_MISS], 0)
                self.assertTrue(math.isnan(stats[CachingEvents.HIT_RATIO]))
                self.assertTrue(math.isnan(stats[CachingEvents.FLUSH_HIT_RATIO]))

        self.assertTrue(isinstance(cache_manager.cache_stats_table(), str))
        cache_stats_table = cache_manager.cache_stats_table()
        expected = ('DynamicFunction', 'HIT\tMISS', '0', '1')   # DynamicFunction HIT 1 time
        for e in expected:
            self.assertIn(e, cache_stats_table)

        self.assertIn('caching_active:', str(cache_manager))
        self.assertIn('cache_invalidation:', str(cache_manager))
        self.assertIn('cache:', str(cache_manager))
