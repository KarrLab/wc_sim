""" Test dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-02-07
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import unittest
from argparse import Namespace
from scipy.constants import Avogadro

from wc_lang.io import Reader
from wc_lang.core import (Submodel, Compartment, Reaction, SpeciesType)
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.dynamic_components import DynamicModel, DynamicCompartment
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError


class TestDynamicCompartment(unittest.TestCase):

    def test_simple_dynamic_compartment(self):

        comp_id = 'comp_id'

        # make a LocalSpeciesPopulation
        num_species = 100
        species_nums = list(range(0, num_species))
        species_ids = list(map(lambda x: "specie_{}[{}]".format(x, comp_id), species_nums))
        all_pops = 1E6
        init_populations = dict(zip(species_ids, [all_pops]*len(species_nums)))
        all_m_weights = 50
        molecular_weights = dict(zip(species_ids, [all_m_weights]*len(species_nums)))
        local_species_pop = LocalSpeciesPopulation('test', init_populations, molecular_weights)

        # make a DynamicCompartment
        initial_volume=1E-17
        compartment = Compartment(id=comp_id, name='name', initial_volume=initial_volume)
        dynamic_compartment = DynamicCompartment(compartment, local_species_pop, species_ids)

        # test DynamicCompartment
        self.assertEqual(dynamic_compartment.volume(), compartment.initial_volume)
        self.assertIn(dynamic_compartment.id, str(dynamic_compartment))
        self.assertIn("Fold change volume: 1.0", str(dynamic_compartment))
        estimated_mass = num_species*all_pops*all_m_weights/Avogadro
        self.assertAlmostEqual(dynamic_compartment.mass(), estimated_mass)
        estimated_density = estimated_mass/initial_volume
        self.assertAlmostEqual(dynamic_compartment.density(), estimated_density)

        # compartment containing just the first element of species_ids
        dynamic_compartment = DynamicCompartment(compartment, local_species_pop, species_ids[:1])
        estimated_mass = all_pops*all_m_weights/Avogadro
        self.assertAlmostEqual(dynamic_compartment.mass(), estimated_mass)

        compartment = Compartment(id='id', name='name', initial_volume=0)
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(compartment, local_species_pop, species_ids)


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')

    def setUp(self):
        for model in [Submodel, Reaction, SpeciesType]:
            model.objects.reset()

    def read_model(self, model_filename):
        # read and initialize a model
        self.model = Reader().run(model_filename, strict=False)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, None)
        dynamic_compartments = multialgorithm_simulation.dynamic_compartments
        self.dynamic_model = DynamicModel(self.model, dynamic_compartments)

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
