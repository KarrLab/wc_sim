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

from obj_tables.expression import Expression
from wc_lang import (Model, Compartment, Species, Parameter,
                     DistributionInitConcentration,
                     Observable, ObservableExpression, StopCondition,
                     Function, FunctionExpression, InitVolume)
from wc_lang.io import Reader
from wc_sim.dynamic_components import (SimTokCodes, WcSimToken, DynamicComponent, DynamicExpression,
                                     DynamicModel, DynamicSpecies, DynamicFunction, DynamicParameter,
                                     DynamicCompartment, DynamicStopCondition)
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.species_populations import LocalSpeciesPopulation, MakeTestLSP
from wc_sim.testing.utils import read_model_and_set_all_std_devs_to_0
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.units import unit_registry
import obj_tables

# Almost all machines map Python floats to IEEE-754 64-bit “double precision”, which provides 15 to
# 17 decimal digits. Places for comparing values that should be equal to within the precision of floats
IEEE_64_BIT_FLOATING_POINT_PLACES = 14


# todo: DfbaObjectiveExpression: expression_term_models = ('Reaction', 'DfbaObjReaction')
class TestDynamicExpressionsComprehensively(unittest.TestCase):

    def setUp(self):
        self.model_file = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dynamic_expressions.xlsx')
        self.model = Reader().run(self.model_file)[Model][0]
        multialgorithm_simulation = MultialgorithmSimulation(self.model, None)
        _, self.dynamic_model = multialgorithm_simulation.build_simulation()

    def test(self):
        # test all DynamicExpressions
        # each one is tested using each of the objects it uses in some instance in self.model_file
        # DynamicFunction
        for id, dynamic_function in self.dynamic_model.dynamic_functions.items():
            expected_value = float(self.model.get_functions(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_function.eval(0), expected_value)
        # test eval_dynamic_functions()
        for func_id, func_val in self.dynamic_model.eval_dynamic_functions(0).items():
            expected_value = float(self.model.get_functions(id= func_id)[0].comments)
            numpy.testing.assert_approx_equal(func_val, expected_value)
        a_func_id = list(self.dynamic_model.dynamic_functions)[0]
        for func_id, func_val in \
            self.dynamic_model.eval_dynamic_functions(0, functions_to_eval=[a_func_id]).items():
            expected_value = float(self.model.get_functions(id= func_id)[0].comments)
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

        # Test DynamicCompartment & DynamicParameter which have eval() methods, although they're not Expressions
        # DynamicCompartment
        for id, dynamic_compartment in self.dynamic_model.dynamic_compartments.items():
            expected_value = float(self.model.get_compartments(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_compartment.eval(0), expected_value)

        # DynamicParameter
        for id, dynamic_parameter in self.dynamic_model.dynamic_parameters.items():
            expected_value = float(self.model.get_parameters(id=id)[0].comments)
            numpy.testing.assert_approx_equal(dynamic_parameter.eval(0), expected_value)


class TestDynamicComponentAndDynamicExpressions(unittest.TestCase):

    def make_objects(self):
        model = Model()
        objects = {
            Observable: {},
            Parameter: {},
            Function: {},
            StopCondition: {}
        }
        self.param_value = 4
        objects[Parameter]['param'] = param = \
            model.parameters.create(id='param', value=self.param_value,
                                    units=unit_registry.parse_units('dimensionless'))

        self.fun_expr = expr = 'param - 2 + max(param, 10)'
        fun1 = Expression.make_obj(model, Function, 'fun1', expr, objects)
        fun2 = Expression.make_obj(model, Function, 'fun2', 'log(2) - param', objects)

        return model, param, fun1, fun2

    def setUp(self):
        self.init_pop = {}
        self.local_species_population = MakeTestLSP(initial_population=self.init_pop).local_species_pop
        self.model, self.parameter, self.fun1, self.fun2 = self.make_objects()

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

        self.model = model = Model()
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
            specie = model.species.create(species_type=species_types[st_id], compartment=compartments[c_id])
            specie.id = specie.gen_id()
            objects[Species][specie.id] = specie
            conc = model.distribution_init_concentrations.create(
                species=specie, mean=0, units=unit_registry.parse_units('M'))
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
    """ Test DynamicCompartment without a species population
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

    def test_dynamic_compartment(self):
        volumes = []
        # test mean initial volume
        for i_trial in range(10):
            dynamic_compartment = DynamicCompartment(None, self.random_state, self.compartment)
            volumes.append(dynamic_compartment.init_volume)
        self.assertLess(numpy.abs((numpy.mean(volumes) - self.mean_init_volume) / self.mean_init_volume), 0.1)

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
    """ Test DynamicCompartment with a species population
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

        # make a Compartment & use it to make a DynamicCompartment
        self.mean_init_volume = 1E-17
        self.compartment = Compartment(id=comp_id, name='name',
                                       init_volume=InitVolume(mean=self.mean_init_volume,
                                       std=self.mean_init_volume / 40.))
        self.compartment.init_density = Parameter(id='density_{}'.format(comp_id), value=1100.,
            units=unit_registry.parse_units('g l^-1'))

        self.random_state = RandomStateManager.instance()
        self.dynamic_compartment = DynamicCompartment(None, self.random_state, self.compartment,
            self.species_ids)

    def specify_init_accounted_fraction(self, desired_init_accounted_fraction):
        # make a DynamicCompartment with accounted_fraction ~= desired_init_accounted_fraction
        # without changing init_accounted_mass or init_volume
        init_density = self.dynamic_compartment.init_accounted_mass / \
            (desired_init_accounted_fraction * self.dynamic_compartment.init_volume)
        self.compartment.init_density = Parameter(id='density_x', value=init_density,
            units=unit_registry.parse_units('g l^-1'))
        return DynamicCompartment(None, self.random_state, self.compartment)

    def test_initialize_mass_and_density(self):
        self.dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        estimated_accounted_mass = self.num_species * self.all_pops * self.all_m_weights / Avogadro
        self.assertAlmostEqual(self.dynamic_compartment.init_accounted_mass, estimated_accounted_mass,
            places=IEEE_64_BIT_FLOATING_POINT_PLACES)
        self.assertTrue(0 < self.dynamic_compartment.init_mass)
        self.assertAlmostEqual(self.dynamic_compartment.accounted_fraction,
            self.dynamic_compartment.init_accounted_mass / self.dynamic_compartment.init_mass,
            places=IEEE_64_BIT_FLOATING_POINT_PLACES)

        empty_local_species_pop = LocalSpeciesPopulation('test', {}, {},
            random_state=RandomStateManager.instance())
        dynamic_compartment = DynamicCompartment(None, self.random_state, self.compartment)
        with self.assertRaisesRegex(MultialgorithmError, "initial accounted ratio is 0"):
            dynamic_compartment.initialize_mass_and_density(empty_local_species_pop)

        dynamic_compartment = self.specify_init_accounted_fraction(
            (DynamicCompartment.MAX_ALLOWED_INIT_ACCOUNTED_FRACTION + 1)/2)
        with warnings.catch_warnings(record=True) as w:
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

    def test_str(self):
        dynamic_compartment = DynamicCompartment(None, self.random_state, self.compartment)
        self.assertIn("has not been initialized", str(dynamic_compartment))
        self.assertIn(self.dynamic_compartment.id, str(dynamic_compartment))

        dynamic_compartment.initialize_mass_and_density(self.local_species_pop)
        self.assertIn("has been initialized", str(dynamic_compartment))
        self.assertIn('Fold change total mass: 1.0', str(dynamic_compartment))


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')

    @classmethod
    def setUpClass(cls):
        cls.models = {}
        for model_file in [cls.MODEL_FILENAME, cls.DRY_MODEL_FILENAME]:
            cls.models[model_file] = Reader().run(model_file, ignore_extra_models=True)[Model][0]

    def make_dynamic_model(self, model_filename):
        # read and initialize a model
        self.model = TestDynamicModel.models[model_filename]
        multialgorithm_simulation = MultialgorithmSimulation(self.model, None)
        multialgorithm_simulation.initialize_components()
        self.dynamic_model = DynamicModel(self.model, multialgorithm_simulation.local_species_population,
            multialgorithm_simulation.dynamic_compartments)

    def compute_expected_actual_masses(self, model_filename):
        # provide the expected actual masses for the compartments in model_filename, keyed by compartment id
        model = TestDynamicModel.models[model_filename]
        expected_actual_masses = {}
        for compartment in model.get_compartments():
            expected_actual_masses[compartment.id] = compartment.init_volume.mean * compartment.init_density.value
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

    # TODO(Arthur): test with multiple compartments that have exchange reactions
    def test_dynamic_model(self):
        self.make_dynamic_model(self.MODEL_FILENAME)
        self.assertEqual(len(self.dynamic_model.cellular_dyn_compartments), 1)
        self.assertEqual(self.dynamic_model.cellular_dyn_compartments[0].name, 'Cell')

        self.assertEqual(self.dynamic_model.get_num_submodels(), 2)

    def test_dynamic_components(self):
        # test agregate properties like mass and volume against independent calculations of their values
        # calculations made in the model's spreadsheet

        # read model while ignoring missing models
        model = read_model_and_set_all_std_devs_to_0(self.MODEL_FILENAME)
        # create dynamic model
        multialgorithm_simulation = MultialgorithmSimulation(model, None)
        multialgorithm_simulation.initialize_components()
        dynamic_model = DynamicModel(model, multialgorithm_simulation.local_species_population,
            multialgorithm_simulation.dynamic_compartments)

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
            'mass': dynamic_model.cell_mass(),
            'volume': dynamic_model.cell_volume(),
            'accounted mass': dynamic_model.cell_accounted_mass(),
            'accounted volume': dynamic_model.cell_accounted_volume()}
        for expected_initial_value in expected_initial_values:
            if expected_initial_value.component == 'whole_cell':
                expected_value = expected_initial_value.expected_initial_value
                actual_value = actual_values[expected_initial_value.attribute]
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

    @unittest.skip("todo: fix or toss test_dry_dynamic_model")
    def test_dry_dynamic_model(self):
        cell_masses = []
        computed_aggregate_states = []
        for i_trial in range(10):
            self.make_dynamic_model(self.DRY_MODEL_FILENAME)
            cell_masses.append(self.dynamic_model.cell_mass())
            computed_aggregate_states.append(self.dynamic_model.get_aggregate_state())

        # expected values computed in tests/fixtures/test_dry_model_with_mass_computation.xlsx
        # todo: fix tests/fixtures/test_dry_model_with_mass_computation.xlsx
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
