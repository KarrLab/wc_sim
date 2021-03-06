""" Test RunResults

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018, Karr Lab
:License: MIT
"""

from capturer import CaptureOutput
from pandas import DataFrame
from scipy.constants import Avogadro
import cProfile
import h5py
import math
import numpy as np
import os
import pandas
import pstats
import shutil
import tempfile
import timeit
import unittest

from de_sim.simulation_config import SimulationConfig
from wc_lang import Species
from wc_sim.metadata import WCSimulationMetadata
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.run_results import RunResults, MakeDataFrame
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.simulation import Simulation
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

        with CaptureOutput(relay=True):
            cls.results_dir_1_cmpt = simulation.run(max_time=cls.max_time,
                                                    results_dir=tempfile.mkdtemp(dir=cls.temp_dir),
                                                    checkpoint_period=cls.checkpoint_period,
                                                    verbose=False).results_dir

        # run a simulation whose aggregate states vary over time
        exchange_rxn_model = os.path.join(os.path.dirname(__file__), 'fixtures', 'dynamic_tests',
                                          'one_exchange_rxn_compt_growth.xlsx')
        model = read_model_for_test(exchange_rxn_model)
        # make both compartments in model cellular, so results are created for both of them
        comp_c = model.get_compartments(id='c')[0]
        comp_e = model.get_compartments(id='e')[0]
        comp_e.biological_type = comp_c.biological_type
        simulation = Simulation(model)
        with CaptureOutput(relay=False):
            cls.results_dir_dyn_aggr = simulation.run(max_time=cls.max_time,
                                                      results_dir=tempfile.mkdtemp(dir=cls.temp_dir),
                                                      checkpoint_period=cls.checkpoint_period).results_dir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        new_tmp_dir = os.path.join(tempfile.mkdtemp(dir=self.temp_dir), 'empty_dir')
        self.results_dir_1_cmpt = shutil.copytree(self.results_dir_1_cmpt, new_tmp_dir)
        self.run_results_1_cmpt = RunResults(self.results_dir_1_cmpt)
        new_tmp_dir = os.path.join(tempfile.mkdtemp(dir=self.temp_dir), 'empty_dir')
        self.results_dir_dyn_aggr = shutil.copytree(self.results_dir_dyn_aggr, new_tmp_dir)
        self.run_results_dyn_aggr = RunResults(self.results_dir_dyn_aggr)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_errors(self):
        with self.assertRaises(MultialgorithmError):
            RunResults(None)
        with self.assertRaises(MultialgorithmError):
            RunResults('not a dir')

    def test__check_component(self):
        for component in RunResults.COMPONENTS:
            if component != 'dfba_reaction_fluxes':
                self.assertEqual(self.run_results_1_cmpt._check_component(component), None)
        self.run_results_1_cmpt.run_results['populations'] = DataFrame()
        with self.assertRaisesRegex(MultialgorithmError, "component is empty"):
            self.run_results_1_cmpt._check_component('populations')

    def test_get(self):
        run_results = RunResults(self.results_dir_1_cmpt)
        for component in RunResults.COMPONENTS:
            self.assertTrue(self.run_results_1_cmpt.get(component).equals(run_results.get(component)))

        expected_times = pandas.Float64Index(np.linspace(0, self.max_time,
                                                            int(1 + self.max_time/self.checkpoint_period)))
        for component in ['populations', 'observables', 'functions', 'rate_laws', 'aggregate_states', 'random_states']:
            component_data = self.run_results_1_cmpt.get(component)
            self.assertFalse(component_data.empty)
            self.assertTrue(component_data.index.equals(expected_times))
        # total population is invariant
        populations = self.run_results_1_cmpt.get('populations')
        pop_sum = populations.sum(axis='columns')
        for time in expected_times:
            self.assertEqual(pop_sum[time], pop_sum[0.])

        volumes = self.run_results_1_cmpt.get('volumes')
        np.testing.assert_array_equal(volumes, self.run_results_1_cmpt.get_volumes())

        with self.assertRaisesRegex(MultialgorithmError, "component '.*' is not an element of "):
            self.run_results_1_cmpt.get('not_a_component')

    def test_prepare_computed_components(self):
        saved_COMPUTED_COMPONENTS = RunResults.COMPUTED_COMPONENTS

        self.assertEqual(RunResults.COMPUTED_COMPONENTS['volumes'], RunResults.get_volumes)
        # for testing, reset COMPUTED_COMPONENTS to a possible original value
        RunResults.COMPUTED_COMPONENTS = {
            'volumes': 'get_volumes'
        }
        RunResults._prepare_computed_components()
        self.assertEqual(RunResults.COMPUTED_COMPONENTS['volumes'], RunResults.get_volumes)

        BAD_COMPUTED_COMPONENTS = {
            'volumes': 'UNKNOWN',
        }
        RunResults.COMPUTED_COMPONENTS = BAD_COMPUTED_COMPONENTS
        with self.assertRaisesRegex(MultialgorithmError, 'in COMPUTED_COMPONENTS is not a method'):
            RunResults._prepare_computed_components()

        # restore COMPUTED_COMPONENTS
        RunResults.COMPUTED_COMPONENTS = saved_COMPUTED_COMPONENTS

    def test__load_hdf_file(self):
        self.run_results_1_cmpt.run_results = {}
        self.run_results_1_cmpt._load_hdf_file()
        for component in self.run_results_1_cmpt.run_results:
            self.assertTrue(isinstance(self.run_results_1_cmpt.run_results[component],
                                       (DataFrame, pandas.Series)))

    def test_get_concentrations(self):
        concentration_in_compt_1 = self.run_results_1_cmpt.get_concentrations('compt_1')
        conc_spec_type_0__compt_1__at_0 = concentration_in_compt_1['spec_type_0[compt_1]'][0.0]
        self.assertTrue(math.isclose(conc_spec_type_0__compt_1__at_0,
                                     self.run_results_1_cmpt.get('populations')['spec_type_0[compt_1]'][0.0] /
                                        (self.run_results_1_cmpt.get_volumes('compt_1')[0.0] * Avogadro),
                                     rel_tol=1e-9))

        concentrations_in_c = self.run_results_dyn_aggr.get_concentrations('c')
        self.assertTrue(concentrations_in_c.columns.values, ['A[c]'])

        concentrations_in_two_compts = self.run_results_dyn_aggr.get_concentrations()
        conc_spec_type_A__compt_c__at_0 = concentrations_in_two_compts['A[c]'][0.0]
        self.assertTrue(math.isclose(conc_spec_type_A__compt_c__at_0,
                                     self.run_results_dyn_aggr.get('populations')['A[c]'][0.0] /
                                        (self.run_results_dyn_aggr.get_volumes('c')[0.0] * Avogadro),
                                     rel_tol=1e-9))

    def test_get_times(self):
        expected_times = np.arange(0., float(self.max_time), self.checkpoint_period, dtype='float64')
        expected_times = np.append(expected_times, float(self.max_time))
        np.testing.assert_array_equal(self.run_results_1_cmpt.get_times(), expected_times)

    def test_aggregate_state_properties(self):
        expected_properties = set(['mass', 'volume', 'accounted mass', 'accounted volume'])
        self.assertEqual(self.run_results_1_cmpt.aggregate_state_properties(), expected_properties)
        self.assertEqual(self.run_results_dyn_aggr.aggregate_state_properties(), expected_properties)

    def test_get_properties(self):
        np.testing.assert_array_equal(self.run_results_1_cmpt.get_properties('compt_1', 'mass'),
                                         self.run_results_1_cmpt.get_properties('compt_1')['mass'])

    def test_get_volumes(self):
        np.testing.assert_array_equal(self.run_results_1_cmpt.get_volumes('compt_1'),
                                         self.run_results_1_cmpt.get_properties('compt_1')['volume'])
        np.testing.assert_array_equal(self.run_results_dyn_aggr.get_volumes('c'),
                                         self.run_results_dyn_aggr.get_properties('c')['volume'])

        # when a model has 1 compartment, obtain same result requesting it
        # or all compartments and then squeezing the df into a Series
        np.testing.assert_array_equal(self.run_results_1_cmpt.get_volumes('compt_1'),
                                         self.run_results_1_cmpt.get_volumes().squeeze())

    def test_get_masses(self):
        np.testing.assert_array_equal(self.run_results_1_cmpt.get_masses('compt_1'),
                                         self.run_results_1_cmpt.get_properties('compt_1')['mass'])
        np.testing.assert_array_equal(self.run_results_dyn_aggr.get_masses('c'),
                                         self.run_results_dyn_aggr.get_properties('c')['mass'])

        np.testing.assert_array_equal(self.run_results_1_cmpt.get_masses('compt_1'),
                                         self.run_results_1_cmpt.get_masses().squeeze())

    def test_convert_metadata(self):
        metadata_file = self.run_results_1_cmpt._hdf_file()
        hdf5_file = h5py.File(metadata_file, 'r')
        metadata_attrs = hdf5_file[RunResults.METADATA_GROUP].attrs
        self.assertEqual(metadata_attrs['wc_sim_metadata.wc_sim_config.checkpoint_period'],
                         self.checkpoint_period)
        self.assertEqual(metadata_attrs['de_sim_metadata.simulation_config.max_time'], self.max_time)

    def test_get_metadata(self):
        sim_metadata = self.run_results_1_cmpt.get_metadata()
        self.assertEqual(sim_metadata['wc_sim_metadata']['wc_sim_config']['checkpoint_period'],
                         self.checkpoint_period)
        self.assertEqual(sim_metadata['de_sim_metadata']['simulation_config']['max_time'], self.max_time)

    def test_semantically_equal(self):
        self.assertTrue(self.run_results_1_cmpt.semantically_equal(self.run_results_1_cmpt))
        self.assertTrue(self.run_results_dyn_aggr.semantically_equal(self.run_results_dyn_aggr))
        self.assertFalse(self.run_results_dyn_aggr.semantically_equal(self.run_results_1_cmpt))
        self.assertFalse(self.run_results_dyn_aggr.semantically_equal(1))
        with CaptureOutput(relay=False) as capturer:
            self.assertFalse(self.run_results_dyn_aggr.semantically_equal(self.run_results_1_cmpt, debug=True))
            self.assertIn('components are not close', capturer.get_text())

        # create and run simulations with different metadata, by adding seed
        model = MakeModel.make_test_model('2 species, 1 reaction',
                                          init_vol_stds=[0],
                                          default_species_std=0,
                                          submodel_framework='WC:deterministic_simulation_algorithm')
        simulation = Simulation(model)
        new_tmp_dir = os.path.join(tempfile.mkdtemp(dir=self.temp_dir), 'empty_dir')
        run_kwargs = dict(max_time=5,
                          results_dir=new_tmp_dir,
                          checkpoint_period=self.checkpoint_period,
                          verbose=False,
                          progress_bar=True)
        run_results_dsa_1 = RunResults(simulation.run(**run_kwargs).results_dir)

        simulation = Simulation(model)
        new_tmp_dir = os.path.join(tempfile.mkdtemp(dir=self.temp_dir), 'empty_dir')
        run_kwargs['results_dir'] = new_tmp_dir
        run_kwargs['seed'] = 2
        run_results_dsa_2 = RunResults(simulation.run(**run_kwargs).results_dir)
        with CaptureOutput(relay=False) as capturer:
            self.assertFalse(run_results_dsa_1.semantically_equal(run_results_dsa_2, debug=True))
            self.assertRegex(capturer.get_text(), "metadata .* instances are not semantically equal")
        self.assertFalse(run_results_dsa_1.semantically_equal(run_results_dsa_2))

    def test_dataframes_are_close(self):
        # test columns and index differ
        data = np.array([1e10, 1e-8])
        index = [1, 2]
        columns = ['A']
        df1 = DataFrame(data, index=index, columns=columns)
        df_diff = DataFrame(data, index=index, columns=['B'])
        self.assertFalse(RunResults.dataframes_are_close(df1, df_diff))
        df_diff = DataFrame(data, index=[2, 2], columns=columns)
        self.assertFalse(RunResults.dataframes_are_close(df1, df_diff))

        # test close and different arrays from https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        kwargs_1 = dict(index=index, columns=columns)
        df1 = DataFrame(np.array([1e10, 1e-8]), **kwargs_1)
        self.assertTrue(RunResults.dataframes_are_close(df1, df1.copy()))
        df_close = DataFrame(np.array([1.00001e10, 1e-9]), **kwargs_1)
        self.assertTrue(RunResults.dataframes_are_close(df1, df_close))
        df_not_close = DataFrame(np.array([1.0001e10, 1e-9]), **kwargs_1)
        self.assertFalse(RunResults.dataframes_are_close(df1, df_not_close))

        # test rtol and atol
        index_2 = [1]
        kwargs_2 = dict(index=index_2, columns=columns)
        df_a = DataFrame(np.array([1_000_000]), **kwargs_2)
        df_b = DataFrame(np.array([1_000_001]), **kwargs_2)
        self.assertTrue(RunResults.dataframes_are_close(df_a, df_b))
        self.assertFalse(RunResults.dataframes_are_close(df_a, df_b, rtol=0))
        self.assertTrue(RunResults.dataframes_are_close(df_a, df_b, atol=0))
        self.assertFalse(RunResults.dataframes_are_close(df_a, df_b, rtol=0, atol=0))

        with self.assertRaises(ValueError):
            RunResults.dataframes_are_close(df1, 3)
        with self.assertRaises(ValueError):
            RunResults.dataframes_are_close(3, df1)

    def test_performance(self):
        # make RunResults local
        from wc_sim.run_results import RunResults

        # remove HDF5_FILENAME, so cost of making it can be measured
        os.remove(self.run_results_1_cmpt._hdf_file())

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
        iterations = 20
        run_results = RunResults(self.results_dir_1_cmpt)
        total_time = timeit.timeit('run_results._load_hdf_file()', globals=locals(), number=iterations)
        mean_time = total_time / iterations
        print(f"mean time of {iterations} runs of '_load_hdf_file()': {mean_time:.2g} (s)")


class TestMakeDataFrame(unittest.TestCase):

    def test(self):
        n_times = 1000
        times = np.arange(n_times)
        n_cols = 1000
        cols = [f"col_{i}" for i in range(n_cols)]
        array = 10. * np.random.rand(n_times, n_cols)
        array = np.rint(array)

        make_df = MakeDataFrame(times, cols)
        for row_num, time in enumerate(times):
            iterator = dict(zip(cols, array[row_num][:]))
            make_df.add(time, iterator)

        self.assertTrue(np.array_equal(array, make_df.ndarray))
        df = make_df.finish()
        self.assertTrue(np.array_equal(df.values, array))
        self.assertEqual(list(df.index), list(times))
        self.assertEqual(list(df.columns), list(cols))


class TestProfileRunResults(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def run_performance_profile(self, num_species, species_pop, species_mw, num_checkpoints):
        """ Run a performance profile of `RunResults()`

        Args:
            num_species (:obj:`int`): number species in the model
            species_pop (:obj:`int`): default species population
            species_mw (:obj:`int`): default species molecular weight
            num_checkpoints (:obj:`int`): number checkpoints in the simulation
        """
        # make RunResults local
        from wc_sim.run_results import RunResults

        print()
        print(f"# species: {num_species}\n# checkpoints: {num_checkpoints}")
        run_results_dir = os.path.join(tempfile.mkdtemp(dir=self.temp_dir), 'run_results_dir')
        os.mkdir(run_results_dir)

        de_simulation_config = SimulationConfig(max_time=num_checkpoints-1, output_dir=run_results_dir)
        de_simulation_config.validate()
        wc_sim_config = WCSimulationConfig(de_simulation_config, checkpoint_period=1)
        wc_sim_config.validate()

        model = MakeModel.make_test_model('1 species, 1 reaction')
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        simulator, _ = multialgorithm_simulation.build_simulation()
        # add remaining additional species
        comp_id = model.compartments[0].id
        local_species_population = multialgorithm_simulation.local_species_population
        for i in range(num_species-1):
            new_species_id = Species._gen_id(f"extra_species_{i}", comp_id)
            local_species_population.init_cell_state_species(new_species_id,
                                                             species_pop,
                                                             species_mw)
        wc_simulation_metadata = WCSimulationMetadata(wc_sim_config)
        simulator.initialize()
        num_events = simulator.simulate(sim_config=de_simulation_config).num_events
        print(simulator.provide_event_counts())
        WCSimulationMetadata.write_dataclass(wc_simulation_metadata, run_results_dir)

        out_file = os.path.join(tempfile.mkdtemp(dir=self.temp_dir), 'profile.out')
        # profile RunResults__init__() & RunResults.convert_checkpoints()
        cProfile.runctx('RunResults(run_results_dir)', locals(), {}, filename=out_file)
        profile = pstats.Stats(out_file)
        print(f"Profile for RunResults() of {num_species} species and {num_checkpoints} checkpoints")
        profile.sort_stats('cumulative').print_stats(20)

    def test_performance_profile(self):

        # test arbitrarily many species and checkpoints
        MAX_SPECIES = 1000
        MAX_CHECKPOINTS = 300
        DEFAULT_POPULATION = 1000
        DEFAULT_MOLECULAR_WEIGHT = 100
        self.run_performance_profile(MAX_SPECIES, DEFAULT_POPULATION, DEFAULT_MOLECULAR_WEIGHT, MAX_CHECKPOINTS)
