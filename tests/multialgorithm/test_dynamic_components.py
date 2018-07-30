""" Test dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-02-07
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import unittest
import warnings
from argparse import Namespace
from scipy.constants import Avogadro
from itertools import chain

from wc_lang.io import Reader
from wc_lang.core import (Submodel, Compartment, Reaction, SpeciesType, Species, SpeciesCoefficient,
    Observable, ObservableCoefficient)
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.dynamic_components import DynamicModel, DynamicCompartment
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.dynamic_expressions import DynamicObservable
from wc_sim.multialgorithm.species_populations import MakeTestLSP
from wc_sim.multialgorithm.make_models import MakeModels


class TestDynamicCompartment(unittest.TestCase):

    def setUp(self):
        comp_id = 'comp_id'

        # make a LocalSpeciesPopulation
        self.num_species = 100
        species_nums = list(range(0, self.num_species))
        self.species_ids = list(map(lambda x: "specie_{}[{}]".format(x, comp_id), species_nums))
        self.all_pops = 1E6
        self.init_populations = dict(zip(self.species_ids, [self.all_pops]*len(species_nums)))
        self.all_m_weights = 50
        self.molecular_weights = dict(zip(self.species_ids, [self.all_m_weights]*len(species_nums)))
        self.local_species_pop = LocalSpeciesPopulation('test', self.init_populations, self.molecular_weights)

        # make a DynamicCompartment
        self.initial_volume=1E-17
        self.compartment = Compartment(id=comp_id, name='name', initial_volume=self.initial_volume)
        self.dynamic_compartment = DynamicCompartment(self.compartment, self.local_species_pop, self.species_ids)

    def test_simple_dynamic_compartment(self):

        # test DynamicCompartment
        self.assertEqual(self.dynamic_compartment.volume(), self.compartment.initial_volume)
        self.assertIn(self.dynamic_compartment.id, str(self.dynamic_compartment))
        self.assertIn("Fold change volume: 1.0", str(self.dynamic_compartment))
        estimated_mass = self.num_species*self.all_pops*self.all_m_weights/Avogadro
        self.assertAlmostEqual(self.dynamic_compartment.mass(), estimated_mass)
        estimated_density = estimated_mass/self.initial_volume
        self.assertAlmostEqual(self.dynamic_compartment.density(), estimated_density)

        # self.compartment containing just the first element of self.species_ids
        self.dynamic_compartment = DynamicCompartment(self.compartment, self.local_species_pop, self.species_ids[:1])
        estimated_mass = self.all_pops*self.all_m_weights/Avogadro
        self.assertAlmostEqual(self.dynamic_compartment.mass(), estimated_mass)

        # set population of species to 0
        init_populations = dict(zip(self.species_ids, [0]*len(self.species_ids)))
        local_species_pop = LocalSpeciesPopulation('test2', init_populations, self.molecular_weights)
        with warnings.catch_warnings(record=True) as w:
            dynamic_compartment = DynamicCompartment(self.compartment, local_species_pop, self.species_ids)
            self.assertIn("initial mass is 0, so constant_density is 0, and volume will remain constant", str(w[-1].message))

        # check that 'volume remains constant'
        self.assertEqual(dynamic_compartment.volume(), self.compartment.initial_volume)
        local_species_pop.adjust_discretely(0, {self.species_ids[0]:5})
        self.assertTrue(0 < dynamic_compartment.mass())
        self.assertEqual(dynamic_compartment.volume(), self.compartment.initial_volume)

    def test_dynamic_compartment_exceptions(self):

        compartment = Compartment(id='id', name='name', initial_volume=0)
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(compartment, self.local_species_pop, self.species_ids)

        compartment = Compartment(id='id', name='name', initial_volume=float('nan'))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(compartment, self.local_species_pop, self.species_ids)


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')

    def read_model(self, model_filename):
        # read and initialize a model
        self.model = Reader().run(model_filename, strict=False)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, None)
        dynamic_compartments = multialgorithm_simulation.dynamic_compartments
        self.dynamic_model = DynamicModel(self.model, multialgorithm_simulation.local_species_population, dynamic_compartments)

    # TODO(Arthur): move this proportional test to a utility & use it instead of assertAlmostEqual everywhere
    def almost_equal_test(self, a, b, frac_diff=1/100):
        delta = min(a, b) * frac_diff
        self.assertAlmostEqual(a, b, delta=delta)

    def compare_aggregate_states(self, expected_aggregate_state, computed_aggregate_state):
        list_of_nested_keys_to_test = [
            ['cell mass'],
            ['cell volume'],
            ['compartments', 'c', 'mass'],
            ['compartments', 'c', 'volume']
        ]
        for nested_keys_to_test in list_of_nested_keys_to_test:
            expected = expected_aggregate_state
            computed = computed_aggregate_state
            for key in nested_keys_to_test:
                expected = expected[key]
                computed = computed[key]
            self.almost_equal_test(expected, computed)

    # TODO(Arthur): test with multiple compartments
    def test_dynamic_model(self):
        self.read_model(self.MODEL_FILENAME)
        self.assertEqual(self.dynamic_model.fraction_dry_weight, 0.3)

        # expected values computed in tests/multialgorithm/fixtures/test_model_with_mass_computation.xlsx
        self.almost_equal_test(self.dynamic_model.cell_mass(), 8.260E-16)
        self.almost_equal_test(self.dynamic_model.cell_dry_weight(), 2.48E-16)
        expected_aggregate_state = {
            'cell mass': 8.260E-16,
            'cell volume': 4.58E-17,
            'compartments': {'c':
                {'mass': 8.260E-16,
                'name': 'Cell',
                'volume': 4.58E-17}}
        }
        computed_aggregate_state = self.dynamic_model.get_aggregate_state()
        self.compare_aggregate_states(expected_aggregate_state, computed_aggregate_state)

    def test_dry_dynamic_model(self):
        self.read_model(self.DRY_MODEL_FILENAME)
        self.assertEqual(self.dynamic_model.fraction_dry_weight, 0)

        # expected values computed in tests/multialgorithm/fixtures/test_dry_model_with_mass_computation.xlsx
        self.almost_equal_test(self.dynamic_model.cell_mass(), 9.160E-19)
        self.almost_equal_test(self.dynamic_model.cell_dry_weight(), 9.160E-19)
        aggregate_state = self.dynamic_model.get_aggregate_state()
        computed_aggregate_state = self.dynamic_model.get_aggregate_state()
        expected_aggregate_state = {
            'cell mass': 9.160E-19,
            'cell volume': 4.58E-17,
            'compartments': {'c':
                {'mass': 9.160E-19,
                'name': 'Cell',
                'volume': 4.58E-17}}
        }
        self.compare_aggregate_states(expected_aggregate_state, computed_aggregate_state)

    def test_eval_dynamic_observables(self):
        # create some dynamic observables
        num_species_types = 10
        species_types = []
        for i in range(num_species_types):
            species_types.append(SpeciesType(id='st_{}'.format(i)))

        comp = Compartment(id='comp_0')

        species = []
        species_coefficients = []
        for st_idx in range(num_species_types):
            species.append(Species(species_type=species_types[st_idx], compartment=comp))
            species_coefficients.append(SpeciesCoefficient(species=species[-1], coefficient=st_idx))

        num_non_dependent_observables = 10
        non_dependent_observables = []
        for i in range(num_non_dependent_observables):
            non_dependent_observables.append(Observable(id='obs_nd_{}'.format(i)))
            for j in range(i):
                non_dependent_observables[-1].species.append(species_coefficients[j])

        num_dependent_observables = 5
        dependent_observables = []
        for i in range(num_dependent_observables):
            dependent_observables.append(Observable(id='obs_d_{}'.format(i)))
            for j in range(i):
                oc = ObservableCoefficient(observable=non_dependent_observables[j], coefficient=j)
                dependent_observables[-1].observables.append(oc)

        # make a LocalSpeciesPopulation
        init_pop = dict(zip([s.id() for s in species], list(range(num_species_types))))
        lsp = MakeTestLSP(initial_population=init_pop).local_species_pop

        # make a Model
        model = MakeModels.make_test_model('no reactions')

        # make a DynamicModel
        dyn_mdl = DynamicModel(model, lsp, {})

        '''
        # activate when implemented
        non_dependent_dynamic_observables = []
        for non_dependent_observable in non_dependent_observables:
            non_dependent_dynamic_observables.append(DynamicObservable(dyn_mdl, lsp, non_dependent_observable))

        dependent_dynamic_observables = []
        for dependent_observable in dependent_observables:
            dependent_dynamic_observables.append(DynamicObservable(dyn_mdl, lsp, dependent_observable))

        # test them
        expected_non_dependent = []
        for idx, dynamic_observable in enumerate(non_dependent_dynamic_observables):
            expected_non_dependent.append(sum([i*i for i in range(idx)]))
            self.assertEqual(dynamic_observable.eval(0), expected_non_dependent[-1])

        expected_dependent = []
        for idx, dynamic_observable in enumerate(dependent_dynamic_observables):
            expected_dependent.append(sum([i*expected_non_dependent[i] for i in range(idx)]))
            self.assertEqual(dynamic_observable.eval(0), expected_dependent[-1])

        ids_of_non_dependent_dynamic_observables = [do.id for do in non_dependent_dynamic_observables]
        self.assertEqual(dyn_mdl.eval_dynamic_observables(0, ids_of_non_dependent_dynamic_observables),
            dict(zip(ids_of_non_dependent_dynamic_observables, expected_non_dependent)))

        ids_of_dependent_dynamic_observables = [do.id for do in dependent_dynamic_observables]
        self.assertEqual(dyn_mdl.eval_dynamic_observables(0, ids_of_dependent_dynamic_observables),
            dict(zip(ids_of_dependent_dynamic_observables, expected_dependent)))

        expected_eval_dynamic_observables = dict(
            zip(chain(ids_of_non_dependent_dynamic_observables, ids_of_dependent_dynamic_observables),
                chain(expected_non_dependent, expected_dependent)))
        self.assertEqual(dyn_mdl.eval_dynamic_observables(0), expected_eval_dynamic_observables)
        '''
