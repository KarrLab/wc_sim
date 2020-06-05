""" Test dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-02-07
:Copyright: 2018, Karr Lab
:License: MIT
"""

from math import log
from scipy.constants import Avogadro
import numpy
import numpy.testing
import os
import re
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
                                       DynamicCompartment, DynamicStopCondition, CacheManager)
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.species_populations import LocalSpeciesPopulation, MakeTestLSP
from wc_sim.testing.utils import read_model_for_test
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.units import unit_registry
import obj_tables
import obj_tables.io


# Almost all machines map Python floats to IEEE-754 64-bit “double precision”, which provides 15 to
# 17 decimal digits. Places for comparing values that should be equal to within the precision of floats
IEEE_64_BIT_FLOATING_POINT_PLACES = 14


class TestInitialDynamicComponentsComprehensively(unittest.TestCase):

    def setUp(self):
        self.model_file = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dynamic_expressions.xlsx')
        self.model = Reader().run(self.model_file)[Model][0]
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, wc_sim_config)
        _, self.dynamic_model = multialgorithm_simulation.build_simulation()

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

    # TODO: FIX FOR DE-SIM CHANGES: make a few dynamic models to test all branches of get_stop_condition()
    # a dynamic model with no stop conditions
    # a dynamic model with stop conditions that all evaluate false
    # a dynamic model with at least one stop conditions that evaluates true
    def test_get_stop_condition(self):
        all_stop_conditions = self.dynamic_model.get_stop_condition()
        self.assertTrue(callable(all_stop_conditions))
        self.assertTrue(all_stop_conditions(0))


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

    return model, param, fun1, fun2


class TestDynamicComponentAndDynamicExpressions(unittest.TestCase):

    def setUp(self):
        self.init_pop = {}
        self.local_species_population = MakeTestLSP(initial_population=self.init_pop).local_species_pop
        self.model, self.parameter, self.fun1, self.fun2 = make_objects(self)

        self.dynamic_model = DynamicModel(self.model, self.local_species_population, {})

        # create a DynamicParameter and some DynamicFunctions
        self.dynamic_objects = dynamic_objects = {}
        dynamic_objects[self.parameter] = DynamicParameter(self.dynamic_model, self.local_species_population,
                                                           self.parameter, self.parameter.value)

        for fun in [self.fun1, self.fun2]:
            dynamic_objects[fun] = DynamicFunction(self.dynamic_model, self.local_species_population,
                                                   fun, fun.expression._parsed_expression)

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
        self.dynamic_model.cache_manager.stop_caching()

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
        model, _, _, _ = make_objects(self)
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

        # clear cache to remove cached value of DynamicCompartment.accounted_mass()
        self.dynamic_model.cache_manager.clear_cache()
        with self.assertRaisesRegex(MultialgorithmError, "initial accounted ratio is 0"):
            dynamic_compartment.initialize_mass_and_density(empty_local_species_pop)

        # clear cache to remove cached value of DynamicCompartment.accounted_mass()
        self.dynamic_model.cache_manager.clear_cache()
        dynamic_compartment = self.specify_init_accounted_fraction(
            (DynamicCompartment.MAX_ALLOWED_INIT_ACCOUNTED_FRACTION + 1)/2)
        with warnings.catch_warnings(record=True) as w:
            # FIX ====
            dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
            self.assertIn("initial accounted ratio (", str(w[-1].message))
            self.assertIn(") greater than 1.0", str(w[-1].message))

        dynamic_compartment = self.specify_init_accounted_fraction(2)
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
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')
    DEPENDENCIES_MDL_FILE = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dependencies.xlsx')

    @classmethod
    def setUpClass(cls):
        cls.models = {}
        for model_file in [cls.MODEL_FILENAME, cls.DRY_MODEL_FILENAME, cls.DEPENDENCIES_MDL_FILE]:
            cls.models[model_file] = Reader().run(model_file, ignore_extra_models=True)[Model][0]

    def make_dynamic_model(self, model_filename):
        # read and initialize a model
        model = TestDynamicModel.models[model_filename]
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        multialgorithm_simulation.initialize_components()
        dynamic_model = DynamicModel(model, multialgorithm_simulation.local_species_population,
                                     multialgorithm_simulation.temp_dynamic_compartments)
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
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        multialgorithm_simulation.initialize_components()
        with self.assertRaisesRegex(MultialgorithmError, 'must have at least 1 cellular compartment'):
            DynamicModel(model, multialgorithm_simulation.local_species_population,
                         multialgorithm_simulation.temp_dynamic_compartments)

    def test_obtain_dependencies(self):
        def sub_dependencies(dependencies, model_type):
            """ Extract the dependencies for models of type model_type
            """
            rv = {}
            for rxn_id, models in dependencies.items():
                rv[rxn_id] = set()
                for model in models:
                    mdl_type, id = model
                    if mdl_type == model_type:
                        rv[rxn_id].add(id)
            return rv

        _, dynamic_model = self.make_dynamic_model(self.DEPENDENCIES_MDL_FILE)
        dependencies = dynamic_model.obtain_dependencies(self.models[self.DEPENDENCIES_MDL_FILE])

        expected_dependencies = {}
        expected_dependencies['Function'] = \
            {'reaction_1': {'function_4', 'function_9'},
             'reaction_2': {'function_5', 'function_9'},
             'reaction_3': set(),
             'reaction_4': {'function_4', 'function_9'},
             'reaction_5': {'function_4', 'function_9'},
             'reaction_6': {'function_4', 'function_5', 'function_9'},
             'reaction_7': {'function_4', 'function_9'},
             'reaction_8': {'function_4', 'function_9'}}

        expected_dependencies['Observable'] = \
            {'reaction_1': {'observable_1', 'observable_2', 'observable_6'},
             'reaction_2': {'observable_3', 'observable_4', 'observable_5', 'observable_7'},
             'reaction_3': {'observable_5', 'observable_7'},
             'reaction_4': {'observable_1', 'observable_2', 'observable_6'},
             'reaction_5': {'observable_2', 'observable_6'},
             'reaction_6': {'observable_1', 'observable_2', 'observable_6', 'observable_3', 'observable_4',
                            'observable_5', 'observable_7'},
             'reaction_7': {'observable_1', 'observable_2', 'observable_6'},
             'reaction_8': {'observable_1', 'observable_2', 'observable_6'}}

        expected_dependencies['RateLaw'] = \
            {'reaction_1': {'reaction_3-forward', 'reaction_5-forward', 'reaction_7-forward'},
             'reaction_2': {'reaction_4-forward', 'reaction_7-forward'},
             'reaction_3': set(),
             'reaction_4': {'reaction_3-forward', 'reaction_5-forward', 'reaction_7-forward'},
             'reaction_5': {'reaction_5-forward', 'reaction_7-forward'},
             'reaction_6': {'reaction_3-forward', 'reaction_4-forward', 'reaction_5-forward', 'reaction_7-forward'},
             'reaction_7': {'reaction_3-forward', 'reaction_5-forward', 'reaction_7-forward'},
             'reaction_8': {'reaction_3-forward', 'reaction_5-forward', 'reaction_7-forward'}}

        expected_dependencies['StopCondition'] = \
            {'reaction_1': {'stop_condition_4', 'stop_condition_5'},
             'reaction_2': {'stop_condition_5'},
             'reaction_3': set(),
             'reaction_4': {'stop_condition_4', 'stop_condition_5'},
             'reaction_5': set(),
             'reaction_6': {'stop_condition_4', 'stop_condition_5'},
             'reaction_7': {'stop_condition_4', 'stop_condition_5'},
             'reaction_8': {'stop_condition_4', 'stop_condition_5'}}

        for model_type in ('Function', 'Observable', 'RateLaw', 'StopCondition'):
            self.assertEqual(sub_dependencies(dependencies, model_type), expected_dependencies[model_type])

    def test_dynamic_components(self):
        # test agregate properties like mass and volume against independent calculations of their values
        # calculations made in the model's spreadsheet

        # read model while ignoring missing models
        model = read_model_for_test(self.MODEL_FILENAME)
        # create dynamic model
        de_simulation_config = SimulationConfig(time_max=10)
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

    @unittest.skip("todo: fix or toss test_dry_dynamic_model()")
    def test_dry_dynamic_model(self):
        cell_masses = []
        computed_aggregate_states = []
        for i_trial in range(10):
            model, dynamic_model = self.make_dynamic_model(self.DRY_MODEL_FILENAME)
            cell_masses.append(dynamic_model.cell_mass())
            computed_aggregate_states.append(dynamic_model.get_aggregate_state())

        # expected values computed in tests/fixtures/test_dry_model_with_mass_computation.xlsx
        numpy.testing.assert_approx_equal(numpy.mean(cell_masses), 9.160E-19)
        aggregate_state = self.dynamic_model.get_aggregate_state()
        expected_aggregate_state = {
            'cell mass': 9.160E-19,
            'compartments': {'c':
                             {'mass': 9.160E-19,
                              'name': 'Cell'}}
        }
        computed_aggregate_state = {
            'cell mass': numpy.mean([s['cell mass'] for s in computed_aggregate_states]),
            'compartments': {'c':
                             {'mass': numpy.mean([s['compartments']['c']['mass'] for s in computed_aggregate_states]),
                              'name': 'Cell'}}
        }
        self.compare_aggregate_states(expected_aggregate_state, computed_aggregate_state, frac_diff=2.5e-1)


class TestCacheManager(unittest.TestCase):

    def test(self):
        cache_manager = CacheManager()
        self.assertEqual(cache_manager._cache, dict())
        self.assertTrue(isinstance(cache_manager.caching(), bool))

        for b in [False, True]:
            cache_manager.set_caching(b)
            self.assertEqual(cache_manager.caching(), b)
        cache_manager.stop_caching()
        self.assertEqual(cache_manager.caching(), False)
        cache_manager.start_caching()
        self.assertEqual(cache_manager.caching(), True)

        v = 3
        id = 'f1'
        cache_manager.set(DynamicFunction, id, v)
        self.assertEqual(cache_manager.get(DynamicFunction, id), v)
        v = 7
        cache_manager.set(DynamicFunction, id, v)
        self.assertEqual(cache_manager.get(DynamicFunction, id), v)
        v = 5
        id = 'f6'
        cache_manager.set(DynamicFunction, id, v)
        self.assertEqual(cache_manager.get(DynamicFunction, id), v)

        v = 44
        id = 's0'
        cache_manager.set(DynamicStopCondition, id, v)
        self.assertEqual(cache_manager.get(DynamicStopCondition, id), v)

        with self.assertRaisesRegex(ValueError, 'key not in cache'):
            cache_manager.get(DynamicStopCondition, 'not_id')
        cache_manager.stop_caching()
        with self.assertRaisesRegex(ValueError, 'caching not enabled'):
            cache_manager.get(DynamicStopCondition, id)
        self.assertEqual(cache_manager.set(DynamicFunction, id, v), None)

        caching_stats = cache_manager.cache_stats_table()
        expected = ('DynamicFunction', 'HIT\tMISS', '0', '3')   # DynamicFunction HIT 3 times
        for e in expected:
            self.assertIn(e, caching_stats)

        cache_manager.clear_cache()
        self.assertEqual(cache_manager._cache, dict())
