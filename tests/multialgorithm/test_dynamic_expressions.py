"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import warnings
from math import log
import re
import timeit

from wc_lang.expression_utils import TokCodes
import wc_lang
from wc_lang import (Model, SpeciesType, Compartment, Species, Parameter, Function, StopCondition,
    FunctionExpression, StopConditionExpression, Observable, ObjectiveFunction, RateLawEquation,
    ExpressionMethods, Concentration, ConcentrationUnit)
from wc_sim.multialgorithm.dynamic_expressions import (DynamicComponent, SimTokCodes, WcSimToken,
    DynamicExpression, DynamicParameter, DynamicFunction, DynamicStopCondition, DynamicObservable,
    DynamicSpecies, WC_LANG_MODEL_TO_DYNAMIC_MODEL)
from wc_sim.multialgorithm.species_populations import MakeTestLSP
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.dynamic_components import DynamicModel
from wc_sim.multialgorithm.make_models import MakeModels


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
            units='dimensionless')
        model.parameters.create(id='fractionDryWeight', value=0.3, units='dimensionless')

        self.fun_expr = expr = 'param - 2 + max(param, 10)'
        fun1 = ExpressionMethods.make_obj(model, Function, 'fun1', expr, objects)
        fun2 = ExpressionMethods.make_obj(model, Function, 'fun2', 'log(2) - param', objects)

        return model, param, fun1, fun2

    def setUp(self):
        self.init_pop = {}
        self.local_species_population = MakeTestLSP(initial_population=self.init_pop).local_species_pop
        self.model, self.parameter, self.fun1, self.fun2 = self.make_objects()

        self.dynamic_model = DynamicModel(self.model, self.local_species_population, {})

        # create a DynamicParameter and a DynamicFunction
        dynamic_objects = {}
        dynamic_objects[self.parameter] = DynamicParameter(self.dynamic_model, self.local_species_population,
            self.parameter, self.parameter.value)

        for fun in [self.fun1, self.fun2]:
            dynamic_objects[fun] = DynamicFunction(self.dynamic_model, self.local_species_population,
                fun, fun.expression.analyzed_expr)
        self.dynamic_objects = dynamic_objects

    def test_simple_dynamic_expressions(self):
        for dyn_obj in self.dynamic_objects.values():
            cls = dyn_obj.__class__
            self.assertEqual(DynamicExpression.dynamic_components[cls][dyn_obj.id], dyn_obj)

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
        self.assertIn( "id: {}".format(dynamic_expression.id), str(dynamic_expression))
        self.assertIn( "type: {}".format(dynamic_expression.__class__.__name__),
            str(dynamic_expression))
        self.assertIn( "expression: {}".format(dynamic_expression.expression), str(dynamic_expression))

        dynamic_expression = self.dynamic_objects[self.fun2]
        dynamic_expression.prepare()
        expected_fun2_wc_sim_tokens = [ # for 'log(2) - param'
            WcSimToken(SimTokCodes.other, 'log(2)-'),
            WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
        ]
        self.assertEqual(expected_fun2_wc_sim_tokens, dynamic_expression.wc_sim_tokens)

    def test_dynamic_expression_errors(self):
        # remove the Function's tokenized result
        self.fun1.expression.analyzed_expr.wc_tokens = []
        with self.assertRaisesRegexp(MultialgorithmError, "wc_tokens cannot be empty - ensure that '.*' is valid"):
            DynamicFunction(self.dynamic_model, self.local_species_population,
                self.fun1, self.fun1.expression.analyzed_expr)

        expr = 'max(1) - 2'
        fun = ExpressionMethods.make_obj(self.model, Function, 'fun', expr, {})
        dynamic_function = DynamicFunction(self.dynamic_model, self.local_species_population,
            fun, fun.expression.analyzed_expr)
        dynamic_function.prepare()
        with self.assertRaisesRegexp(MultialgorithmError, re.escape("eval of '{}' raises".format(expr))):
            dynamic_function.eval(1)

    def test_get_dynamic_model_type(self):

        self.assertEqual(DynamicExpression.get_dynamic_model_type(Function), DynamicFunction)
        with self.assertRaisesRegexp(MultialgorithmError, "model class of type 'FunctionExpression' not found"):
            DynamicExpression.get_dynamic_model_type(FunctionExpression)

        self.assertEqual(DynamicExpression.get_dynamic_model_type(self.fun1), DynamicFunction)
        expr_model_obj, _ = ExpressionMethods.make_expression_obj(Function, '', {})
        with self.assertRaisesRegexp(MultialgorithmError, "model of type 'FunctionExpression' not found"):
            DynamicExpression.get_dynamic_model_type(expr_model_obj)

        self.assertEqual(DynamicExpression.get_dynamic_model_type('Function'), DynamicFunction)
        with self.assertRaisesRegexp(MultialgorithmError, "model type 'NoSuchModel' not defined"):
            DynamicExpression.get_dynamic_model_type('NoSuchModel')
        with self.assertRaisesRegexp(MultialgorithmError, "model type '3' has wrong type"):
            DynamicExpression.get_dynamic_model_type(3)


class TestAllDynamicExpressionTypes(unittest.TestCase):

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
            submodels[id] = model.submodels.create(id=id, compartment=compartments[c_id])

        for c_id, st_id in zip(comp_ids, st_ids):
            specie = compartments[c_id].species.create(species_type=species_types[st_id])
            objects[Species][specie.get_id()] = specie
            Concentration(species=specie, value=0, units=ConcentrationUnit.M)

        self.init_pop = {'a[c1]': 10, 'b[c2]': 20}

        # map wc_lang object -> expected value
        self.expected_values = expected_values = {}
        param_value = 4
        objects[Parameter]['param'] = param = model.parameters.create(id='param', value=param_value,
            units='dimensionless')
        expected_values[param] = param_value

        # (wc_lang model type, expression, expected value)
        wc_lang_obj_specs = [
            # just reference param:
            (Function, 'param - 2 + max(param, 10)', 12),
            (StopCondition, '10 < 2*log10(100) + 2*param', True),
            # reference other model types:
            (Observable, 'a[c1]', 10),
            (Observable, '2*a[c1] - b[c2]', 0),
            (Function, 'observable_1 + min(observable_2, 10)' , 10),
            (StopCondition, 'observable_1 < param + function_1()', True),
            # reference same model type:
            (Observable, '3*observable_1 + b[c2]', 50),
            (Function, '2*function_2()', 20),
            (Function, '3*observable_1 + function_1()', 42)
        ]

        self.expression_models = expression_models = [Function, StopCondition, Observable]
        last_ids = {wc_lang_type:0 for wc_lang_type in expression_models}
        def make_id(wc_lang_type):
            last_ids[wc_lang_type] += 1
            return "{}_{}".format(wc_lang_type.__name__.lower(), last_ids[wc_lang_type])

        # create wc_lang models
        for wc_lang_model_type, expr, expected_value in wc_lang_obj_specs:
            obj_id = make_id(wc_lang_model_type)
            wc_lang_obj = ExpressionMethods.make_obj(model, wc_lang_model_type, obj_id, expr, objects)
            objects[wc_lang_model_type][obj_id] = wc_lang_obj
            expected_values[wc_lang_obj.id] = expected_value

        # needed for simulation:
        model.parameters.create(id='fractionDryWeight', value=0.3, units='dimensionless')

        self.local_species_population = MakeTestLSP(initial_population=self.init_pop).local_species_pop
        self.dynamic_model = DynamicModel(self.model, self.local_species_population, {})

    def test_all_dynamic_expressions(self):

        # check computed value and measure performance of all test Dynamic objects
        number = 10000
        print()
        print("Measure {} evals of each Dynamic expression:".format(number))
        for dynamic_obj_dict in [self.dynamic_model.dynamic_observables,
            self.dynamic_model.dynamic_functions, self.dynamic_model.dynamic_stop_conditions]:
            for id, dynamic_expression in dynamic_obj_dict.items():
                self.assertEqual(self.expected_values[id], dynamic_expression.eval(0))
                eval_time = timeit.timeit(stmt='dynamic_expression.eval(0)', number=number,
                    globals=locals())
                print("{:.2f} usec/eval of {} {} '{}'".format(eval_time*1E6/number,
                    dynamic_expression.__class__.__name__, dynamic_expression.id, dynamic_expression.expression))
