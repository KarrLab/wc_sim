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
                                     DynamicCompartment)
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.species_populations import LocalSpeciesPopulation, MakeTestLSP
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.units import unit_registry


class TestDynamicExpression(unittest.TestCase):

    def make_objects(self):
        model = Model()
        objects = {
            Observable: {},
            Parameter: {},
            Function: {},
            StopCondition: {}
        }
        self.param_value = 4
        objects[Parameter]['param'] = param = model.parameters.create(id='param', value=self.param_value,
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
        dynamic_objects = {}
        dynamic_objects[self.parameter] = DynamicParameter(self.dynamic_model, self.local_species_population,
                                                           self.parameter, self.parameter.value)

        for fun in [self.fun1, self.fun2]:
            dynamic_objects[fun] = DynamicFunction(self.dynamic_model, self.local_species_population,
                                                   fun, fun.expression._parsed_expression)
        self.dynamic_objects = dynamic_objects
        # todo: test other DynamicComponents

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
        with self.assertRaisesRegex(MultialgorithmError, "_obj_tables_tokens cannot be empty - ensure that '.*' is valid"):
            DynamicFunction(self.dynamic_model, self.local_species_population,
                            self.fun1, self.fun1.expression._parsed_expression)

        expr = 'max(1) - 2'
        fun = Expression.make_obj(self.model, Function, 'fun', expr, {}, allow_invalid_objects=True)
        dynamic_function = DynamicFunction(self.dynamic_model, self.local_species_population,
                                           fun, fun.expression._parsed_expression)
        dynamic_function.prepare()
        with self.assertRaisesRegex(MultialgorithmError, re.escape("eval of '{}' raises".format(expr))):
            dynamic_function.eval(1)

    def test_get_dynamic_model_type(self):

        self.assertEqual(DynamicComponent.get_dynamic_model_type(Function), DynamicFunction)
        with self.assertRaisesRegex(MultialgorithmError, "model class of type 'FunctionExpression' not found"):
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
        objects[Parameter]['param'] = param = model.parameters.create(id='param', value=4,
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
            (Function, 'observable_1 + min(observable_2, 10 * molecule_units)', 10, unit_registry.parse_units('molecule')),
            (StopCondition, 'observable_1 < param * molecule_units + function_1', True,
                unit_registry.parse_units('dimensionless')),
            # reference same model type:
            (Observable, '3 * observable_1 + b[c2]', 50, unit_registry.parse_units('molecule')),
            (Function, '2 * function_2', 20, unit_registry.parse_units('molecule')),
            (Function, '3 * observable_1 + function_1 * molecule_units', 42, unit_registry.parse_units('molecule'))
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


class TestDynamicCompartment(unittest.TestCase):

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

        # make a DynamicCompartment
        self.mean_init_volume = 1E-17
        self.compartment = Compartment(id=comp_id, name='name',
                                       init_volume=InitVolume(mean=self.mean_init_volume,
                                       std=self.mean_init_volume / 40.))
        self.compartment.init_density = Parameter(id='density_{}'.format(comp_id), value=1100.,
            units=unit_registry.parse_units('g l^-1'))

        self.dynamic_compartment = DynamicCompartment(None, self.local_species_pop, self.compartment,
            self.species_ids)

    def test_dynamic_compartment(self):
        volumes = []
        # test mean initial volume
        for i_trial in range(10):
            dynamic_compartment = DynamicCompartment(None, self.local_species_pop, self.compartment)
            volumes.append(dynamic_compartment.init_volume)
        self.assertLess(numpy.abs((numpy.mean(volumes) - self.mean_init_volume) / self.mean_init_volume), 0.1)

    def specify_init_accounted_ratio(self, desired_init_accounted_ratio):
        # make a DynamicCompartment with init_accounted_ratio ~= desired_init_accounted_ratio
        # without changing init_accounted_mass or init_volume
        init_density = self.dynamic_compartment.init_accounted_mass / \
            (desired_init_accounted_ratio * self.dynamic_compartment.init_volume)
        self.compartment.init_density = Parameter(id='density_x', value=init_density,
            units=unit_registry.parse_units('g l^-1'))
        return DynamicCompartment(None, self.local_species_pop, self.compartment)

    def test_initialize_mass_and_density(self):
        self.dynamic_compartment.initialize_mass_and_density()
        estimated_accounted_mass = self.num_species * self.all_pops * self.all_m_weights / Avogadro
        # IEEE 64-bit floating point has 53 bits in the mantissa, providing 15 to 17 decimal digits
        self.assertAlmostEqual(self.dynamic_compartment.init_accounted_mass, estimated_accounted_mass,
            places=14)
        self.assertTrue(0 < self.dynamic_compartment.init_mass)
        self.assertAlmostEqual(self.dynamic_compartment.init_accounted_ratio,
            self.dynamic_compartment.init_accounted_mass / self.dynamic_compartment.init_mass, places=14)

        empty_local_species_pop = LocalSpeciesPopulation('test', {}, {},
            random_state=RandomStateManager.instance())
        dynamic_compartment = DynamicCompartment(None, empty_local_species_pop, self.compartment)
        with warnings.catch_warnings(record=True) as w:
            dynamic_compartment.initialize_mass_and_density()
            self.assertIn("initial accounted ratio is 0", str(w[-1].message))

        dynamic_compartment = self.specify_init_accounted_ratio(
            (DynamicCompartment.MAX_ALLOWED_INIT_ACCOUNTED_RATIO + 1)/2)
        with warnings.catch_warnings(record=True) as w:
            dynamic_compartment.initialize_mass_and_density()
            self.assertIn("initial accounted ratio (", str(w[-1].message))
            self.assertIn(") greater than 1.0", str(w[-1].message))

        dynamic_compartment = self.specify_init_accounted_ratio(2)
        with self.assertRaises(MultialgorithmError):
            dynamic_compartment.initialize_mass_and_density()

    def test_fold_changes(self):
        # ensure that initial fold changes == 1
        self.dynamic_compartment.initialize_mass_and_density()
        self.assertAlmostEqual(self.dynamic_compartment.fold_change_total_mass(0), 1.0)
        self.assertAlmostEqual(self.dynamic_compartment.fold_change_total_volume(0), 1.0)

    def test_init_volume_and_eval(self):
        self.dynamic_compartment.initialize_mass_and_density()
        self.assertAlmostEqual(self.dynamic_compartment.volume(), self.dynamic_compartment.init_volume,
            places=14)
        self.assertAlmostEqual(self.dynamic_compartment.eval(), self.dynamic_compartment.init_volume,
            places=14)

    def test_str(self):
        dynamic_compartment = DynamicCompartment(None, self.local_species_pop, self.compartment)
        dynamic_compartment.initialize_mass_and_density()
        self.assertIn(self.dynamic_compartment.id, str(dynamic_compartment))
        self.assertIn(str(self.dynamic_compartment.init_density), str(dynamic_compartment))

    @unittest.skip("todo: refactor into new tests")
    def test_simple_dynamic_compartment_old(self):

        # test DynamicCompartment
        masses = []
        for i_trial in range(10):
            self.dynamic_compartment = DynamicCompartment(None, self.local_species_pop, self.compartment,
                self.species_ids)
            masses.append(self.dynamic_compartment.mass())
        self.assertIn(self.dynamic_compartment.id, str(self.dynamic_compartment))
        estimated_mass = self.num_species * self.all_pops * self.all_m_weights / Avogadro
        self.assertLess(numpy.abs((numpy.mean(masses) - estimated_mass) / estimated_mass), 0.1)

        # self.compartment containing just the first element of self.species_ids
        self.dynamic_compartment = DynamicCompartment(None, self.local_species_pop, self.compartment,
            self.species_ids[:1])
        estimated_mass = self.all_pops*self.all_m_weights / Avogadro
        self.assertAlmostEqual(self.dynamic_compartment.mass(), estimated_mass)

        # set population of species to 0
        init_populations = dict(zip(self.species_ids, [0] * len(self.species_ids)))
        local_species_pop = LocalSpeciesPopulation('test2', init_populations, self.molecular_weights,
                                                   random_state=RandomStateManager.instance())
        for i_trial in range(10):
            with warnings.catch_warnings(record=True) as w:
                dynamic_compartment = DynamicCompartment(None, local_species_pop, self.compartment,
                    self.species_ids)
                self.assertIn("initial mass is 0", str(w[-1].message))
            self.assertEqual(dynamic_compartment.init_mass, 0.)
            self.assertEqual(dynamic_compartment.mass(), 0.)

        # check that 'mass increases'
        self.assertEqual(dynamic_compartment.mass(), 0.)
        local_species_pop.adjust_discretely(0, {self.species_ids[0]: 5})
        self.assertEqual(dynamic_compartment.mass(), 5 * self.all_m_weights / Avogadro)

    def test_dynamic_compartment_exceptions(self):

        compartment = Compartment(id='id', name='name', init_volume=InitVolume(mean=0))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.local_species_pop, compartment, self.species_ids)

        self.compartment.init_density = Parameter(id='density_{}'.format(self.compartment.id),
                                                value=float('nan'),
                                                units=unit_registry.parse_units('g l^-1'))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.local_species_pop, self.compartment)

        self.compartment.init_density = Parameter(id='density_{}'.format(self.compartment.id),
                                                value=0., units=unit_registry.parse_units('g l^-1'))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.local_species_pop, self.compartment)

        self.compartment.init_volume = None
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.local_species_pop, self.compartment)


@unittest.skip("todo: refactor into new tests")
class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')

    def read_model(self, model_filename):
        # read and initialize a model
        self.model = Reader().run(model_filename)[Model][0]
        multialgorithm_simulation = MultialgorithmSimulation(self.model, None)
        dynamic_compartments = multialgorithm_simulation.dynamic_compartments
        self.dynamic_model = DynamicModel(self.model, multialgorithm_simulation.local_species_population,
            dynamic_compartments)

    # TODO(Arthur): move this proportional test to a utility & use it instead of assertAlmostEqual everywhere
    def almost_equal_test(self, a, b, frac_diff=1/100):
        delta = min(a, b) * frac_diff
        self.assertAlmostEqual(a, b, delta=delta)

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
            self.almost_equal_test(expected, computed, frac_diff=frac_diff)

    # TODO(Arthur): test with multiple compartments
    def test_dynamic_model(self):
        cell_masses = []
        computed_aggregate_states = []
        for i_trial in range(10):
            self.read_model(self.MODEL_FILENAME)
            cell_masses.append(self.dynamic_model.cell_mass())
            computed_aggregate_states.append(self.dynamic_model.get_aggregate_state())

        # expected values computed in tests/fixtures/test_model_with_mass_computation.xlsx
        self.almost_equal_test(numpy.mean(cell_masses), 8.260E-16, frac_diff=1e-1)
        expected_aggregate_state = {
            'cell mass': 8.260E-16,
            'compartments': {'c':
                             {'mass': 8.260E-16,
                              'name': 'Cell'}}
        }
        computed_aggregate_state = {
            'cell mass': numpy.mean([s['cell mass'] for s in computed_aggregate_states]),
            'compartments': {'c':
                             {'mass': numpy.mean([s['compartments']['c']['mass'] for s in computed_aggregate_states]),
                              'name': 'Cell'}}
        }
        self.compare_aggregate_states(expected_aggregate_state, computed_aggregate_state, frac_diff=2.5e-1)

    def test_dry_dynamic_model(self):
        cell_masses = []
        computed_aggregate_states = []
        for i_trial in range(10):
            self.read_model(self.DRY_MODEL_FILENAME)
            cell_masses.append(self.dynamic_model.cell_mass())
            computed_aggregate_states.append(self.dynamic_model.get_aggregate_state())

        # expected values computed in tests/fixtures/test_dry_model_with_mass_computation.xlsx
        self.almost_equal_test(numpy.mean(cell_masses), 9.160E-19, frac_diff=1e-1)
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

    def test_eval_dynamic_observables(self):
        # make a Model
        model = Model()
        comp = model.compartments.create(id='comp_0')
        submodel = model.submodels.create(id='submodel')

        num_species_types = 10
        species_types = []
        for i in range(num_species_types):
            st = model.species_types.create(id='st_{}'.format(i))
            species_types.append(st)

        species = []
        for st_idx in range(num_species_types):
            specie = model.species.create(species_type=species_types[st_idx], compartment=comp)
            specie.id = specie.gen_id()
            conc = model.distribution_init_concentrations.create(
                species=specie, mean=0, units=unit_registry.parse_units('M'))
            conc.id = conc.gen_id()
            species.append(specie)

        # create some observables
        objects = {
            Species: {},
            Observable: {}
        }
        num_non_dependent_observables = 5
        non_dependent_observables = []
        for i in range(num_non_dependent_observables):
            expr_parts = []
            for j in range(i+1):
                expr_parts.append("{}*{}".format(j, species[j].id))
                objects[Species][species[j].id] = species[j]
            expr = ' + '.join(expr_parts)
            obj = ObservableExpression.make_obj(model, Observable, 'obs_nd_{}'.format(i), expr, objects)
            self.assertTrue(obj.expression.validate() is None)
            non_dependent_observables.append(obj)

        num_dependent_observables = 4
        dependent_observables = []
        for i in range(num_dependent_observables):
            expr_parts = []
            for j in range(i+1):
                nd_obs_id = 'obs_nd_{}'.format(j)
                expr_parts.append("{}*{}".format(j, nd_obs_id))
                objects[Observable][nd_obs_id] = non_dependent_observables[j]
            expr = ' + '.join(expr_parts)
            obj = ObservableExpression.make_obj(model, Observable, 'obs_d_{}'.format(i), expr, objects)
            self.assertTrue(obj.expression.validate() is None)
            dependent_observables.append(obj)

        # make a LocalSpeciesPopulation
        init_pop = dict(zip([s.id for s in species], list(range(num_species_types))))
        lsp = MakeTestLSP(initial_population=init_pop).local_species_pop

        # make a DynamicModel
        dyn_mdl = DynamicModel(model, lsp, {})
        # check that dynamic observables have the right values
        for obs_id, obs_val in dyn_mdl.eval_dynamic_observables(0).items():
            index = int(obs_id.split('_')[-1])
            if 'obs_nd_' in obs_id:
                expected_val = float(sum([i*i for i in range(index+1)]))
                self.assertEqual(expected_val, obs_val)
            elif 'obs_d_' in obs_id:
                expected_val = 0
                for d_index in range(index+1):
                    expected_val += d_index * sum([i*i for i in range(d_index+1)])
                self.assertEqual(expected_val, obs_val)

    def test_eval_dynamic_functions(self):
        # make a Model
        model = Model()
        comp = model.compartments.create(id='comp_0')
        submodel = model.submodels.create(id='submodel')

        num_species_types = 10
        species_types = []
        for i in range(num_species_types):
            st = model.species_types.create(id='st_{}'.format(i))
            species_types.append(st)

        species = []
        for st_idx in range(num_species_types):
            specie = model.species.create(species_type=species_types[st_idx], compartment=comp)
            specie.id = specie.gen_id()
            conc = model.distribution_init_concentrations.create(
                species=specie, mean=0, units=unit_registry.parse_units('M'))
            conc.id = conc.gen_id()
            species.append(specie)

        # create some functions
        objects = {
            Species: {},
            Function: {}
        }
        num_non_dependent_functions = 5
        non_dependent_functions = []
        for i in range(num_non_dependent_functions):
            expr_parts = []
            for j in range(i + 1):
                expr_parts.append("{} * {}".format(j, species[j].id))
                objects[Species][species[j].id] = species[j]
            expr = ' + '.join(expr_parts)
            obj = FunctionExpression.make_obj(model, Function, 'func_nd_{}'.format(i), expr, objects)
            self.assertTrue(obj.expression.validate() is None)
            non_dependent_functions.append(obj)

        num_dependent_functions = 4
        dependent_functions = []
        for i in range(num_dependent_functions):
            expr_parts = []
            for j in range(i+1):
                nd_func_id = 'func_nd_{}'.format(j)
                expr_parts.append("{}*{}".format(j, nd_func_id))
                objects[Function][nd_func_id] = non_dependent_functions[j]
            expr = ' + '.join(expr_parts)
            obj = FunctionExpression.make_obj(model, Function, 'func_d_{}'.format(i), expr, objects)
            self.assertTrue(obj.expression.validate() is None)
            dependent_functions.append(obj)

        # make a LocalSpeciesPopulation
        init_pop = dict(zip([s.id for s in species], list(range(num_species_types))))
        lsp = MakeTestLSP(initial_population=init_pop).local_species_pop

        # make a DynamicModel
        dyn_mdl = DynamicModel(model, lsp, {})
        # check that dynamic functions have the right values
        for func_id, func_val in dyn_mdl.eval_dynamic_functions(0).items():
            index = int(func_id.split('_')[-1])
            if 'func_nd_' in func_id:
                expected_val = float(sum([i*i for i in range(index+1)]))
                self.assertEqual(expected_val, func_val)
            elif 'func_d_' in func_id:
                expected_val = 0
                for d_index in range(index+1):
                    expected_val += d_index * sum([i * i for i in range(d_index + 1)])
                self.assertEqual(expected_val, func_val)
