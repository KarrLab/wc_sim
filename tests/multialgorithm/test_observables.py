"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import warnings

from wc_lang import (StopCondition, Function, Observable, SpeciesType, Compartment, Species,
    SpeciesCoefficient, ObservableCoefficient)
from wc_sim.multialgorithm.observables import DynamicObservable, DynamicFunction, DynamicStopCondition
from wc_sim.multialgorithm.species_populations import MakeTestLSP
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.dynamic_components import DynamicModel
from wc_sim.multialgorithm.make_models import MakeModels


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
        pass

    def test_dynamic_stop_condition(self):
        pass
