""" Test utilities for testing

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-10-31
:Copyright: 2019, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
import os
import shutil
import tempfile
import unittest

from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.simulation import Simulation
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import (read_model_and_set_all_std_devs_to_0, check_simul_results,
                                  plot_expected_vs_actual)


class TestTestingUtils(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.results_dir = tempfile.mkdtemp(dir=self.tmp_dir)
        self.args = dict(results_dir=tempfile.mkdtemp(dir=self.tmp_dir),
                         checkpoint_period=1)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_check_simul_results(self):
        init_volume = 1E-16
        init_density = 1000
        molecular_weight = 100.
        default_species_copy_number = 10_000
        init_accounted_mass = molecular_weight * default_species_copy_number / Avogadro
        init_accounted_density = init_accounted_mass / init_volume
        expected_initial_values_compt_1 = dict(
            init_volume=init_volume,
            init_accounted_mass=init_accounted_mass,
            init_mass= init_volume * init_density,
            init_density=init_density,
            init_accounted_density=init_accounted_density,
            accounted_fraction = init_accounted_density / init_density
        )
        expected_initial_values = {'compt_1': expected_initial_values_compt_1}
        model = MakeModel.make_test_model('1 species, 1 reaction',
                                        init_vols=[expected_initial_values_compt_1['init_volume']],
                                        init_vol_stds=[0],
                                        density=init_density,
                                        molecular_weight=molecular_weight,
                                        default_species_copy_number=default_species_copy_number,
                                        default_species_std=0,
                                        submodel_framework='WC:deterministic_simulation_algorithm')
        multialgorithm_simulation = MultialgorithmSimulation(model, self.args)
        _, dynamic_model = multialgorithm_simulation.build_simulation()
        check_simul_results(self, dynamic_model, None, expected_initial_values=expected_initial_values)

        # test dynamics
        simulation = Simulation(model)
        _, results_dir = simulation.run(end_time=2, **self.args)
        check_simul_results(self, dynamic_model, results_dir,
                                 expected_initial_values=expected_initial_values,
                                 expected_species_trajectories=\
                                     {'spec_type_0[compt_1]':[10000., 9999., 9998.]})
        with self.assertRaises(AssertionError):
            check_simul_results(self, dynamic_model, results_dir,
                                     expected_initial_values=expected_initial_values,
                                     expected_species_trajectories=\
                                         {'spec_type_0[compt_1]':[10000., 10000., 9998.]})
        with self.assertRaises(AssertionError):
            check_simul_results(self, dynamic_model, results_dir,
                                     expected_initial_values=expected_initial_values,
                                     expected_species_trajectories=\
                                         {'spec_type_0[compt_1]':[10000., 10000.]})
        check_simul_results(self, dynamic_model, results_dir,
                                 expected_initial_values=expected_initial_values,
                                 expected_species_trajectories=\
                                    {'spec_type_0[compt_1]':[10000., 9999., 9998.]},
                                    delta=1E-5)
        check_simul_results(self, dynamic_model, results_dir,
                                 expected_property_trajectories={'compt_1':
                                    {'mass':[1.000e-13, 1.000e-13, 9.999e-14]}})
        check_simul_results(self, dynamic_model, results_dir,
                                 expected_property_trajectories={'compt_1':
                                    {'mass':[1.000e-13, 1.000e-13, 9.999e-14]}},
                                    delta=1E-7)
        plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
        os.makedirs(plots_dir, exist_ok=True)
        plot_expected_vs_actual(dynamic_model,
                                results_dir,
                                trajectory_times=[0, 1, 2],
                                plots_dir=plots_dir,
                                expected_species_trajectories=\
                                    {'spec_type_0[compt_1]':[10000., 10000., 9998.]},
                                expected_property_trajectories=\
                                    {'compt_1':
                                        {'mass':[1.000e-13, 1.000e-13, 9.999e-14]}})