"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import warnings
import math

from wc_lang import (StopCondition, Function, Observable, SpeciesType, Compartment, Species,
    SpeciesCoefficient, ObservableCoefficient)
from wc_sim.multialgorithm.observables import (DynamicObservable, DynamicFunction,
    DynamicStopCondition, TokCodes)
from wc_sim.multialgorithm.species_populations import MakeTestLSP
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.dynamic_components import DynamicModel
from wc_sim.multialgorithm.make_models import MakeModels


# a temporary hack, until wc_lang.Function correctly parses expressions
class PseudoFunction(object):
    def __init__(self, id, tokens, observables):
        self.id = id
        self.tokens = tokens
        self.observables = observables
# e.g.: 3 * log( obs_1 )
# would tokenize as [('3', other), ('*', other), ('log', math_function), 
# ('(', other), ('observable_1', python_id), (')', other)]


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

        self.tokens = [('3', TokCodes.other), ('*', TokCodes.other), ('log', TokCodes.math_function),
            ('(', TokCodes.other), ('obs_1', TokCodes.python_id), (')', TokCodes.other)]
        self.pseudo_function = PseudoFunction('fun_1', self.tokens, [self.obs_1])

        self.model = MakeModels.make_test_model('1 species, 1 reaction')
        self.dyn_mdl = DynamicModel(self.model, {})

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
        '''
            implmentations to compare:
                no memoization w recomputation
                no memoization and avoid recomputation for each eval by tracking dependencies
                memoization *
            workloads to test:
                small observables with few dependencies
                large observables with many dependencies
            * the current implementation
        '''
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

    def test_dynamic_stop_condition(self):
        dyn_obs_1 = DynamicObservable(self.dyn_mdl, self.lsp, self.obs_1)
        # expression: obs_1 < 90
        tokens1 = [('obs_1', TokCodes.python_id), ('<', TokCodes.other), ('90', TokCodes.other)]
        tokens2 = [('obs_1', TokCodes.python_id), ('>', TokCodes.other), ('90', TokCodes.other)]
        for tokens,expected_val in zip([tokens1, tokens2], [True, False]):
            pseudo_function = PseudoFunction('fun_1', tokens, [self.obs_1])
            dynamic_stop_cond = DynamicStopCondition(self.dyn_mdl, pseudo_function)
            self.assertEqual(dynamic_stop_cond.eval(0), expected_val)

    def test_dynamic_stop_condition_exceptions_and_warnings(self):
        with self.assertRaises(MultialgorithmError):
            DynamicStopCondition(self.dyn_mdl, self.pseudo_function)

        tokens = [('9', TokCodes.other), ('-', TokCodes.other), ('5', TokCodes.other)]
        pseudo_function = PseudoFunction('fun_1', tokens, [])
        dynamic_stop_cond = DynamicStopCondition(self.dyn_mdl, pseudo_function)
        with self.assertRaises(MultialgorithmError):
            dynamic_stop_cond.eval(0)
