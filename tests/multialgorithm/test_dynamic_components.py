""" Test dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-02-07
:Copyright: 2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
from wc_lang import (Model, Compartment, Species, Parameter,
                     DistributionInitConcentration, ConcentrationUnit,
                     Observable, ObservableExpression,
                     Function, FunctionExpression)
from wc_lang.io import Reader
from wc_sim.multialgorithm.dynamic_components import DynamicModel, DynamicCompartment
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation, MakeTestLSP
from wc_utils.util.rand import RandomStateManager
import numpy
import numpy.testing
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
        self.local_species_pop = LocalSpeciesPopulation('test', self.init_populations, self.molecular_weights,
                                                        random_state=RandomStateManager.instance())

        # make a DynamicCompartment
        self.mean_init_volume = 1E-17

        self.compartment = Compartment(id=comp_id, name='name', mean_init_volume=self.mean_init_volume,
                                       std_init_volume=self.mean_init_volume / 10.)
        self.compartment.init_density = Parameter(id='density_{}'.format(comp_id), value=1100., units='g l^-1')

        self.dynamic_compartment = DynamicCompartment(None, self.local_species_pop, self.compartment, self.species_ids)

    def test_simple_dynamic_compartment(self):

        # test DynamicCompartment
        masses = []
        for i_trial in range(10):
            self.dynamic_compartment = DynamicCompartment(None, self.local_species_pop, self.compartment, self.species_ids)
            masses.append(self.dynamic_compartment.mass())
        self.assertIn(self.dynamic_compartment.id, str(self.dynamic_compartment))
        estimated_mass = self.num_species * self.all_pops * self.all_m_weights / Avogadro
        self.assertLess(numpy.abs((numpy.mean(masses) - estimated_mass) / estimated_mass), 0.25)

        # self.compartment containing just the first element of self.species_ids
        self.dynamic_compartment = DynamicCompartment(None, self.local_species_pop, self.compartment, self.species_ids[:1])
        estimated_mass = self.all_pops*self.all_m_weights / Avogadro
        self.assertAlmostEqual(self.dynamic_compartment.mass(), estimated_mass)

        # set population of species to 0
        init_populations = dict(zip(self.species_ids, [0] * len(self.species_ids)))
        local_species_pop = LocalSpeciesPopulation('test2', init_populations, self.molecular_weights,
                                                   random_state=RandomStateManager.instance())
        for i_trial in range(10):
            with warnings.catch_warnings(record=True) as w:
                dynamic_compartment = DynamicCompartment(None, local_species_pop, self.compartment, self.species_ids)
                self.assertIn("initial mass is 0", str(w[-1].message))
            self.assertEqual(dynamic_compartment.init_mass, 0.)
            self.assertEqual(dynamic_compartment.mass(), 0.)

        # check that 'mass increases'
        self.assertEqual(dynamic_compartment.mass(), 0.)
        local_species_pop.adjust_discretely(0, {self.species_ids[0]: 5})
        self.assertEqual(dynamic_compartment.mass(), 5 * self.all_m_weights / Avogadro)

    def test_dynamic_compartment_exceptions(self):

        compartment = Compartment(id='id', name='name', mean_init_volume=0)
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.local_species_pop, compartment, self.species_ids)

        compartment = Compartment(id='id', name='name', mean_init_volume=float('nan'))
        with self.assertRaises(MultialgorithmError):
            DynamicCompartment(None, self.local_species_pop, compartment, self.species_ids)


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')
    DRY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_dry_model.xlsx')

    def read_model(self, model_filename):
        # read and initialize a model
        self.model = Reader().run(model_filename)[Model][0]
        multialgorithm_simulation = MultialgorithmSimulation(self.model, None)
        dynamic_compartments = multialgorithm_simulation.dynamic_compartments
        self.dynamic_model = DynamicModel(self.model, multialgorithm_simulation.local_species_population, dynamic_compartments)

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

        # expected values computed in tests/multialgorithm/fixtures/test_model_with_mass_computation.xlsx
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

        # expected values computed in tests/multialgorithm/fixtures/test_dry_model_with_mass_computation.xlsx
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
                species=specie, mean=0, units=ConcentrationUnit.M)
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
                species=specie, mean=0, units=ConcentrationUnit.M)
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
