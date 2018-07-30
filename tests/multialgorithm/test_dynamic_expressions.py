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
    FunctionExpression, StopConditionExpression, Observable, ObjectiveFunction, RateLawEquation)
from wc_sim.multialgorithm.dynamic_expressions import (DynamicComponent, SimTokCodes, WcSimToken,
    DynamicExpression, DynamicParameter, DynamicFunction, DynamicStopCondition, DynamicObservable)
from wc_sim.multialgorithm.species_populations import MakeTestLSP
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.dynamic_components import DynamicModel
from wc_sim.multialgorithm.make_models import MakeModels


class TestDynamicExpressionModule(unittest.TestCase):

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
        fun_expr, _ = FunctionExpression.deserialize(Function.Meta.attributes['expression'], expr, objects)
        objects[Function]['fun'] = fun = model.functions.create(id='fun', expression=fun_expr)

        self.cond_expr = expr = '10 < 2 * log(10) + 2*param'
        cond_expr, _ = StopConditionExpression.deserialize(StopCondition.Meta.attributes['expression'],
            expr, objects)
        objects[StopCondition]['sc'] = stop_cond = model.stop_conditions.create(id='sc', expression=cond_expr)
        self.objects = objects

        return model, param, fun, stop_cond

    def setUp(self):
        self.st_a = st_a = SpeciesType(id='a')
        self.st_b = st_b = SpeciesType(id='b')
        self.c1 = c1 = Compartment(id='c1')
        self.c2 = c2 = Compartment(id='c2')
        self.s_a_1 = s_a_1 = Species(species_type=st_a, compartment=c1)
        self.s_b_2 = s_b_2 = Species(species_type=st_b, compartment=c2)
        self.init_pop = {'a[c1]': 10, 'b[c2]': 20}
        self.local_species_population = MakeTestLSP(initial_population=self.init_pop).local_species_pop
        self.model, self.parameter, self.function, self.stop_cond = self.make_objects()

        self.dynamic_model = DynamicModel(self.model, self.local_species_population, {})

        # create all the Dynamic* objects
        dynamic_objects = {}
        dynamic_objects[self.parameter] = DynamicParameter(self.dynamic_model, self.local_species_population,
            self.parameter, self.parameter.value)
        dynamic_objects[self.function] = DynamicFunction(self.dynamic_model, self.local_species_population,
            self.function, self.function.expression.analyzed_expr)
        dynamic_objects[self.stop_cond] = DynamicStopCondition(self.dynamic_model, self.local_species_population,
            self.stop_cond, self.stop_cond.expression.analyzed_expr)
        self.dynamic_objects = dynamic_objects

    def test_simple_dynamic_expressions(self):
        for dyn_obj in self.dynamic_objects.values():
            cls = dyn_obj.__class__
            self.assertEqual(DynamicExpression.dynamic_components[cls][dyn_obj.id], dyn_obj)

        expected_wc_sim_tokens = {
            self.function: [
                WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
                WcSimToken(SimTokCodes.other, '-2+max('),
                WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter]),
                WcSimToken(SimTokCodes.other, ',10)'),
            ],
            self.stop_cond: [
                WcSimToken(SimTokCodes.other, '10<2*log(10)+2*'),
                WcSimToken(SimTokCodes.dynamic_expression, 'param', self.dynamic_objects[self.parameter])
            ]
        }
        expected_expr_substrings = {
            self.function: ['', '-2+max(', '', ',10)'],
            self.stop_cond: ['10<2*log(10)+2*', '']
        }
        expected_local_ns_keys = {self.function: 'max', self.stop_cond: 'log'}
        param_val = str(self.param_value)
        expected_values = {
            self.function: eval(self.fun_expr.replace('param', param_val)),
            self.stop_cond: eval(self.cond_expr.replace('param', param_val))
        }
        for expression_model in [self.function, self.stop_cond]:
            dynamic_expression = self.dynamic_objects[expression_model]
            dynamic_expression.prepare()
            self.assertEqual(expected_wc_sim_tokens[expression_model], dynamic_expression.wc_sim_tokens)
            self.assertEqual(expected_expr_substrings[expression_model], dynamic_expression.expr_substrings)
            self.assertTrue(expected_local_ns_keys[expression_model] in dynamic_expression.local_ns)
            self.assertEqual(expected_values[expression_model], dynamic_expression.eval(0))
            self.assertIn( "id: {}".format(dynamic_expression.id), str(dynamic_expression))
            self.assertIn( "type: {}".format(dynamic_expression.__class__.__name__),
                str(dynamic_expression))
            self.assertIn( "expression: {}".format(dynamic_expression.expression), str(dynamic_expression))

    def test_dynamic_expression_perf(self):
        number = 10000
        print()
        for expression_model in [self.function, self.stop_cond]:
            dynamic_expression = self.dynamic_objects[expression_model]
            dynamic_expression.prepare()
            eval_time = timeit.timeit(stmt='dynamic_expression.eval(0)', number=number, globals=locals())
            print("{:.2f} usec per eval for {} '{}'".format(eval_time*1E6/number,
                dynamic_expression.__class__.__name__, dynamic_expression.expression))

    def test_dynamic_expression_errors(self):
        # remove the Function's tokenized result
        self.function.expression.analyzed_expr.wc_tokens = []
        with self.assertRaisesRegexp(MultialgorithmError, "wc_tokens cannot be empty"):
            DynamicFunction(self.dynamic_model, self.local_species_population,
                self.function, self.function.expression.analyzed_expr)

        expr = 'max(1) - 2'
        fun_expr, _ = FunctionExpression.deserialize(Function.Meta.attributes['expression'], expr, self.objects)
        fun = self.model.functions.create(id='fun', expression=fun_expr)
        dynamic_function = DynamicFunction(self.dynamic_model, self.local_species_population,
            fun, fun.expression.analyzed_expr)
        dynamic_function.prepare()
        with self.assertRaisesRegexp(MultialgorithmError, re.escape("eval of '{}' raises".format(expr))):
            dynamic_function.eval(1)


class TestDynamicFunction(unittest.TestCase):

    def setUp(self):
        pass


class TestDynamicStopCondition(unittest.TestCase):

    def setUp(self):
        pass


class TestDynamicObservable(unittest.TestCase):

    def setUp(self):
        pass

'''
class TestDynamicObservables(unittest.TestCase):

    def setUp(self):
        self.st_a = st_a = SpeciesType(id='a')
        self.st_b = st_b = SpeciesType(id='bb')
        self.c_a = c_a = Compartment(id='a')
        self.c_b = c_b = Compartment(id='bb')
        self.s_a = s_a = Species(species_type=st_a, compartment=c_a)
        self.s_b = s_b = Species(species_type=st_b, compartment=c_b)
        self.sc_a = sc_a = SpeciesCoefficient(species=s_a, coefficient=2.)
        self.sc_b = sc_b = SpeciesCoefficient(species=s_b, coefficient=3.)
        self.obs_1 = obs_1 = Observable(id='obs_1')
        obs_1.species.append(sc_a)
        obs_1.species.append(sc_b)
        self.obs_coeff_1 = obs_coeff_1 = ObservableCoefficient(observable=obs_1, coefficient=5.)
        self.obs_2 = obs_2 = Observable(id='obs_2')
        obs_2.species.append(sc_a)
        obs_2.observables.append(obs_coeff_1)

        self.init_pop = {'a[a]': 10, 'bb[bb]': 20}
        self.lsp = MakeTestLSP(initial_population=self.init_pop).local_species_pop

        self.tokens = [('3', TokCodes.other), ('*', TokCodes.other), ('log', TokCodes.math_fun_id),
            ('(', TokCodes.other), ('obs_1', TokCodes.wc_lang_obj_id), (')', TokCodes.other)]
        self.pseudo_function = PseudoFunction('fun_1', self.tokens, [self.obs_1])

        self.model = MakeModels.make_test_model('1 species, 1 reaction')
        self.dyn_mdl = DynamicModel(self.model, self.lsp, {})

    def test_dynamic_observable(self):

        dynamic_observable_1 = DynamicObservable(self.dyn_mdl, self.lsp, self.obs_1)
        self.assertEqual(set(dynamic_observable_1.weighted_species),
            set([(self.sc_a.coefficient, self.s_a.id()), (self.sc_b.coefficient, self.s_b.id())]))
        self.assertEqual(dynamic_observable_1.weighted_observables, [])
        self.assertEqual(dynamic_observable_1.eval(0),
            2 * self.init_pop['a[a]'] + 3 * self.init_pop['bb[bb]'])

        dynamic_observable_2 = DynamicObservable(self.dyn_mdl, self.lsp, self.obs_2)
        self.assertEqual(set(dynamic_observable_2.weighted_species),
            set([(self.sc_a.coefficient, self.s_a.id())]))
        self.assertEqual(set(dynamic_observable_2.weighted_observables),
            set([(self.obs_coeff_1.coefficient, dynamic_observable_1)]))
        self.assertEqual(dynamic_observable_2.eval(0),
            2 * self.init_pop['a[a]'] + 5 * dynamic_observable_1.eval(0))
        self.assertIn('weighted_species', str(dynamic_observable_2))
        self.assertIn('weighted_observables', str(dynamic_observable_2))

    def test_dynamic_observable_memoize_perf(self):
        """
            implmentations to compare:
                no memoization w recomputation
                no memoization and avoid recomputation for each eval by tracking dependencies
                memoization *
            workloads to test:
                small observables with few dependencies
                large observables with many dependencies
            * the current implementation
        """
        # large observables with many dependencies
        # TODO: make performance test
        num_dependencies = 20

    def test_dynamic_observable_exceptions_and_warnings(self):
        obs = Observable(id='')
        with self.assertRaises(MultialgorithmError):
            DynamicObservable(self.dyn_mdl, None, obs)
        with self.assertRaises(MultialgorithmError):
            DynamicObservable(self.dyn_mdl, None, self.obs_2)
        DynamicObservable(self.dyn_mdl, None, self.obs_1)
        with warnings.catch_warnings(record=True) as w:
            DynamicObservable(self.dyn_mdl, None, self.obs_1)
            self.assertRegex(str(w[-1].message), "Replacing observable '.*' with a new instance")

    @unittest.skip("fixed when DynamicExpressions work")
    def test_dynamic_function(self):
        expression = '3 * log ( obs_1 )'
        self.assertEqual(expression, ' '.join([val for val, _ in self.tokens]))
        dyn_obs_1 = DynamicObservable(self.dyn_mdl, self.lsp, self.obs_1)
        dynamic_function_1 = DynamicFunction(self.dyn_mdl, self.pseudo_function)
        self.assertEqual(dynamic_function_1.dynamic_observables,
            {dyn_obs_1.id: dyn_obs_1})
        # SB: 3 * log( dyn_obs_1(0) ) = 3 * log( 80 )
        self.assertAlmostEqual(dynamic_function_1.eval(0), 3 * math.log( 80 ))

    def test_dynamic_function_exceptions_and_warnings(self):
        with self.assertRaises(MultialgorithmError):
            DynamicFunction(self.dyn_mdl, self.pseudo_function)

    @unittest.skip("fixed when DynamicExpressions work")
    def test_dynamic_stop_condition(self):
        dyn_obs_1 = DynamicObservable(self.dyn_mdl, self.lsp, self.obs_1)
        # expression: obs_1 < 90
        tokens1 = [('obs_1', TokCodes.wc_lang_obj_id), ('<', TokCodes.other), ('90', TokCodes.other)]
        tokens2 = [('obs_1', TokCodes.wc_lang_obj_id), ('>', TokCodes.other), ('90', TokCodes.other)]
        for tokens,expected_val in zip([tokens1, tokens2], [True, False]):
            pseudo_function = PseudoFunction('fun_1', tokens, [self.obs_1])
            dynamic_stop_cond = DynamicStopCondition(self.dyn_mdl, pseudo_function)
            self.assertEqual(dynamic_stop_cond.eval(0), expected_val)

    @unittest.skip("fixed when DynamicExpressions work")
    def test_dynamic_stop_condition_exceptions_and_warnings(self):
        with self.assertRaises(MultialgorithmError):
            DynamicStopCondition(self.dyn_mdl, self.pseudo_function)

        tokens = [('9', TokCodes.other), ('-', TokCodes.other), ('5', TokCodes.other)]
        pseudo_function = PseudoFunction('fun_1', tokens, [])
        dynamic_stop_cond = DynamicStopCondition(self.dyn_mdl, pseudo_function)
        with self.assertRaises(MultialgorithmError):
            dynamic_stop_cond.eval(0)
'''
