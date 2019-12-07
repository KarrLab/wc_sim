""" Test RunResults

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018, Karr Lab
:License: MIT
"""

from capturer import CaptureOutput
from scipy.constants import Avogadro
import math
import numpy
import os
import pandas
import shutil
import tempfile
import timeit
import unittest

from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.simulation import Simulation
from wc_sim.run_results import RunResults
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import read_model_for_test


class TestRunResults(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # run each simulation only once & copy their results in setUp
        cls.temp_dir = tempfile.mkdtemp()
        # create and run simulation
        model = MakeModel.make_test_model('2 species, 1 reaction')
        simulation = Simulation(model)
        cls.checkpoint_period = 5
        cls.max_time = 30

        with CaptureOutput(relay=False):
            _, cls.results_dir_1_cmpt = simulation.run(end_time=cls.max_time,
                                                        results_dir=tempfile.mkdtemp(dir=cls.temp_dir),
                                                        checkpoint_period=cls.checkpoint_period)

        # run a simulation whose aggregate states vary over time
        exchange_rxn_model = os.path.join(os.path.dirname(__file__), 'fixtures', 'dynamic_tests',
                                          'one_exchange_rxn_compt_growth.xlsx')
        model = read_model_for_test(exchange_rxn_model)
        simulation = Simulation(model)
        with CaptureOutput(relay=False):
            _, cls.results_dir_dyn_aggr = simulation.run(end_time=cls.max_time,
                                                        results_dir=tempfile.mkdtemp(dir=cls.temp_dir),
                                                        checkpoint_period=cls.checkpoint_period)

        # todo: test with model that has multiple compartments

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
mass        new_tmp_dir = os.path.join(tempfile.mkdtemp(dir=self.temp_dir), 'empty_dir')
        self.results_dir_1_cmpt = shutil.copytree(self.results_dir_1_cmpt, new_tmp_dir)
        self.run_results_1_cmpt = RunResults(self.results_dir_1_cmpt)
mass        new_tmp_dir = os.path.join(tempfile.mkdtemp(dir=self.temp_dir), 'empty_dir')
        self.results_dir_dyn_aggr = shutil.copytree(self.results_dir_dyn_aggr, new_tmp_dir)
        self.run_results_dyn_aggr = RunResults(self.results_dir_dyn_aggr)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get(self):
        run_results_2 = RunResults(self.results_dir_1_cmpt)
        for component in RunResults.COMPONENTS:
            self.assertTrue(self.run_results_1_cmpt.get(component).equals(run_results_2.get(component)))

        expected_times = pandas.Float64Index(numpy.linspace(0, self.max_time, 1 + self.max_time/self.checkpoint_period))
        for component in ['populations', 'observables', 'functions', 'aggregate_states', 'random_states']:
            component_data = self.run_results_1_cmpt.get(component)
            self.assertFalse(component_data.empty)
            self.assertTrue(component_data.index.equals(expected_times))

        # total population is invariant
        populations = self.run_results_1_cmpt.get('populations')
        pop_sum = populations.sum(axis='columns')
        for time in expected_times:
            self.assertEqual(pop_sum[time], pop_sum[0.])

        metadata = self.run_results_1_cmpt.get('metadata')
        self.assertEqual(metadata['simulation']['time_max'], self.max_time)

        volumes = self.run_results_1_cmpt.get('volumes')
        numpy.testing.assert_array_equal(volumes, self.run_results_1_cmpt.get_volumes())

        with self.assertRaisesRegex(MultialgorithmError, "component '.*' is not an element of "):
            self.run_results_1_cmpt.get('not_a_component')

    def test_prepare_computed_components(self):
        saved_COMPUTED_COMPONENTS = RunResults.COMPUTED_COMPONENTS

        self.assertEqual(RunResults.COMPUTED_COMPONENTS['volumes'], RunResults.get_volumes)
        # for testing, reset COMPUTED_COMPONENTS to a possible original value
        RunResults.COMPUTED_COMPONENTS = {
            'volumes': 'get_volumes'
        }
        RunResults.prepare_computed_components()
        self.assertEqual(RunResults.COMPUTED_COMPONENTS['volumes'], RunResults.get_volumes)

        BAD_COMPUTED_COMPONENTS = {
            'volumes': 'UNKNOWN',
        }
        RunResults.COMPUTED_COMPONENTS = BAD_COMPUTED_COMPONENTS
        with self.assertRaisesRegex(MultialgorithmError, 'in COMPUTED_COMPONENTS is not a method'):
            RunResults.prepare_computed_components()

        # restore COMPUTED_COMPONENTS
        RunResults.COMPUTED_COMPONENTS = saved_COMPUTED_COMPONENTS

    def test__load_hdf_file(self):
        self.run_results_1_cmpt.run_results = {}
        self.run_results_1_cmpt._load_hdf_file()
        for component in self.run_results_1_cmpt.run_results:
            self.assertTrue(isinstance(self.run_results_1_cmpt.run_results[component],
                                       (pandas.DataFrame, pandas.Series)))

    def test_get_concentrations(self):
        concentration_in_compt_1 = self.run_results_1_cmpt.get_concentrations('compt_1')
        conc_spec_type_0__compt_1__at_0 = concentration_in_compt_1['spec_type_0[compt_1]'][0.0]
        self.assertTrue(math.isclose(conc_spec_type_0__compt_1__at_0,
                                     self.run_results_1_cmpt.get('populations')['spec_type_0[compt_1]'][0.0] /
                                        (self.run_results_1_cmpt.get_volumes('compt_1')[0.0] * Avogadro),
                                     rel_tol=1e-9))

    def test_get_times(self):
        expected_times = numpy.arange(0., float(self.max_time), self.checkpoint_period, dtype='float64')
        expected_times = numpy.append(expected_times, float(self.max_time))
        numpy.testing.assert_array_equal(self.run_results_1_cmpt.get_times(), expected_times)

    def test__get_properties(self):
        numpy.testing.assert_array_equal(self.run_results_1_cmpt._get_properties('compt_1', 'mass'),
                                         self.run_results_1_cmpt._get_properties('compt_1')['mass'])

    def test_get_volumes(self):
        numpy.testing.assert_array_equal(self.run_results_1_cmpt.get_volumes('compt_1'),
                                         self.run_results_1_cmpt._get_properties('compt_1')['volume'])

        # when a model has 1 compartment, obtain same result requesting it
        # or all compartments and then squeezing the df into a Series
        numpy.testing.assert_array_equal(self.run_results_1_cmpt.get_volumes('compt_1'),
                                         self.run_results_1_cmpt.get_volumes().squeeze())

        # todo: test with multiple compartments

    def test_get_masses(self):
        numpy.testing.assert_array_equal(self.run_results_1_cmpt.get_masses('compt_1'),
                                         self.run_results_1_cmpt._get_properties('compt_1')['mass'])

        numpy.testing.assert_array_equal(self.run_results_1_cmpt.get_masses('compt_1'),
                                         self.run_results_1_cmpt.get_masses().squeeze())

        # todo: test with multiple compartments

    def test_performance(self):

        # make RunResults local
        from wc_sim.run_results import RunResults

        # remove HDF5_FILENAME, so cost of making it can be measured
        os.remove(os.path.join(self.results_dir_1_cmpt, RunResults.HDF5_FILENAME))

        print()
        iterations = 5
        results_dirs = []
        new_tmp_dir = tempfile.mkdtemp()
        for i in range(iterations):
            new_results_dir = shutil.copytree(self.results_dir_1_cmpt, os.path.join(new_tmp_dir, f'results_dir_{i}'))
            results_dirs.append(new_results_dir)
        total_time = timeit.timeit('[RunResults(results_dir) for results_dir in results_dirs]',
                                   globals=locals(), number=iterations)
        mean_time = total_time / iterations
        print(f"mean time of {iterations} runs of 'RunResults(results_dir)': {mean_time:.2g} (s)")
        shutil.rmtree(new_tmp_dir)
        iterations = 100
        run_results = RunResults(self.results_dir_1_cmpt)
        total_time = timeit.timeit('run_results._load_hdf_file()', globals=locals(), number=iterations)
        mean_time = total_time / iterations
        print(f"mean time of {iterations} runs of '_load_hdf_file()': {mean_time:.2g} (s)")
