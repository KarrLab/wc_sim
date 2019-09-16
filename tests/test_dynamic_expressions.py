"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

from math import log
from obj_model.expression import Expression
from wc_lang import (Model, Species, DistributionInitConcentration,
                     Observable, ObservableExpression,
                     Function, FunctionExpression,
                     StopCondition,
                     Parameter)
from wc_sim.dynamic_components import DynamicModel
from wc_sim.dynamic_expressions import (SimTokCodes, WcSimToken, DynamicComponent, DynamicExpression,
                                        DynamicSpecies, DynamicFunction, DynamicParameter)
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.species_populations import MakeTestLSP
from wc_utils.util.units import unit_registry
import re
import timeit
import unittest
import wc_lang


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
        self.fun1.expression._parsed_expression._obj_model_tokens = []
        with self.assertRaisesRegex(MultialgorithmError, "_obj_model_tokens cannot be empty - ensure that '.*' is valid"):
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
