""" Test dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-02-07
:Copyright: 2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
from wc_lang import (Model, Compartment, Species,
                     DistributionInitConcentration, ConcentrationUnit, Observable)
from wc_lang.expression import Expression
from wc_lang.io import Reader
from wc_sim.multialgorithm.dynamic_components import DynamicModel, DynamicCompartment
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation, MakeTestLSP
import os
import unittest
import warnings


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
        self.local_species_pop = LocalSpeciesPopulation('test', self.init_populations, self.molecular_weights)

        # make a DynamicCompartment
        self.mean_init_volume = 1E-17
        self.compartment = Compartment(id=comp_id, name='name', mean_init_volume=self.mean_init_volume)
        self.dynamic_compartment = DynamicCompartment(self.compartment, self.local_species_pop, self.species_ids)

    def test_simple_dynamic_compartment(self):

        # test DynamicCompartment
        self.assertEqual(self.dynamic_compartment.volume(), self.compartment.mean_init_volume)
        self.assertIn(self.dynamic_compartment.id, str(self.dynamic_compartment))
        self.assertIn("Fold change volume: 1.0", str(self.dynamic_compartment))
        estimated_mass = self.num_species*self.all_pops*self.all_m_weights/Avogadro
        self.assertAlmostEqual(self.dynamic_compartment.mass(), estimated_mass)
        estimated_density = estimated_mass/self.mean_init_volume
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
        self.assertEqual(dynamic_compartment.volume(), self.compartment.mean_init_volume)
        local_species_pop.adjust_discretely(0, {self.species_ids[0]: 5})
        self.assertTrue(0 < dynamic_compartment.mass())
        self.assertEqual(dynamic_compartment.volume(), self.compartment.mean_init_volume)

    def test_dynamic_compartment_exceptions(self):

        compartment = Compartment(id='id', name='name', mean_init_volume=0)
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(compartment, self.local_species_pop, self.species_ids)

        compartment = Compartment(id='id', name='name', mean_init_volume=float('nan'))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(compartment, self.local_species_pop, self.species_ids)


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')

    def read_model(self, model_filename):
        # read and initialize a model
        self.model = Reader().run(model_filename)
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
        # make a Model
        model = Model()
        comp = model.compartments.create(id='comp_0')
        submodel = model.submodels.create(id='submodel')
        model.parameters.create(id='fractionDryWeight', value=0.3, units='dimensionless')

        num_species_types = 10
        species_types = []
        for i in range(num_species_types):
            st = model.species_types.create(id='st_{}'.format(i))
            species_types.append(st)

        species = []
        for st_idx in range(num_species_types):
            specie = model.species.create(species_type=species_types[st_idx], compartment=comp)
            specie.id = specie.gen_id(specie.species_type.id, specie.compartment.id)
            conc = model.distribution_init_concentrations.create(
                id=DistributionInitConcentration.gen_id(specie.id),
                species=specie, mean=0, units=ConcentrationUnit.M)
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
            obj = Expression.make_obj(model, Observable, 'obs_nd_{}'.format(i), expr, objects)
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
            obj = Expression.make_obj(model, Observable, 'obs_d_{}'.format(i), expr, objects)
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
