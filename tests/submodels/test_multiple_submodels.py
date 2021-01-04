"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-12-03
:Copyright: 2020, Karr Lab
:License: MIT
"""

from capturer import CaptureOutput
import os
import pandas
import shutil
import tempfile
import unittest

from wc_lang import util
from wc_lang.io import Reader
from wc_sim.submodels.dfba import DfbaSubmodel
from wc_sim.testing.transformations import SetStdDevsToZero
from wc_sim.testing.utils import create_run_directory
from wc_utils.util.environ import EnvironUtils
import wc_lang
import wc_sim


class TestMultipleSubmodels(unittest.TestCase):

    TEST_MODEL = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'test_model.xlsx')
    EXAMPLE_WC_MODEL = os.path.join(os.path.dirname(__file__), 'fixtures', '4_submodel_MP_model.xlsx')

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with EnvironUtils.temp_config_env([(['wc_lang', 'validation', 'validate_element_charge_balance'], 'False')]):
            self.model = Reader().run(self.EXAMPLE_WC_MODEL, validate=True)[wc_lang.Model][0]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def run_simulation(self, simulation, max_time=10, seed=None, options=None):
        checkpoint_period = min(10, max_time)
        with CaptureOutput(relay=True):
            simulation_rv = simulation.run(max_time=max_time,
                                           results_dir=create_run_directory(),
                                           checkpoint_period=checkpoint_period,
                                           dfba_time_step=2,
                                           seed=seed,
                                           options=options,
                                           progress_bar=False,
                                           verbose=False)
        self.assertTrue(0 < simulation_rv.num_events)
        self.assertTrue(os.path.isdir(simulation_rv.results_dir))
        run_results = wc_sim.RunResults(simulation_rv.results_dir)

        for component in wc_sim.RunResults.COMPONENTS:
            self.assertTrue(isinstance(run_results.get(component), (pandas.DataFrame, pandas.Series)))

        return simulation_rv

    def test_run_submodel(self):
        print()

        # species with positive initial concentrations
        initialized_species = set()
        for dist_init_conc in self.model.get_distribution_init_concentrations():
            if 0 < dist_init_conc.mean:
                initialized_species.add(dist_init_conc.species)

        # species that lack initial concentrations
        un_initialized_species = set()
        for species in self.model.get_species():
            if species not in initialized_species:
                un_initialized_species.add(species)

        print(f'Species that lack initial concentrations:')
        print('\n'.join(sorted([s.id for s in un_initialized_species])))

        # list reactants w uninitialized concs.
        reactants = set()
        for rxn in self.model.get_reactions():
            for species, stoichiometry in DfbaSubmodel._get_species_and_stoichiometry(rxn).items():
                if stoichiometry < 0:
                    reactants.add(species)
        sorted_reactant_ids = sorted([r.id for r in reactants - initialized_species])
        print(f'Reactants in reactions that lack initial concentrations:')
        print('\n'.join(sorted_reactant_ids))

        reactants = set()
        for dfba_obj_reaction in self.model.get_dfba_obj_reactions():
            for species, stoichiometry in DfbaSubmodel._get_species_and_stoichiometry(dfba_obj_reaction).items():
                if stoichiometry < 0:
                    reactants.add(species)
        sorted_reactant_ids = sorted([r.id for r in reactants - initialized_species])
        print(f'Reactants in dfba_obj_reactions that lack initial concentrations:')
        print('\n'.join(sorted_reactant_ids))

        N = 1
        max_time = 50    # 3 * 60 * 60
        seeds = [17, 19, 23, 29, 31]
        dfba_options = dict(options=dict(presolve='on',
                                         verbosity='status'))
        options = {'DfbaSubmodel': dfba_options}
        for i in range(N):
            with EnvironUtils.temp_config_env([(['wc_lang', 'validation', 'validate_element_charge_balance'], 'False')]):
                model = Reader().run(self.EXAMPLE_WC_MODEL, validate=True)[wc_lang.Model][0]
                print(util.get_model_summary(model))
                print(f"\n*** RUN {i} with seed {seeds[i]} ***")
                try:
                    # TODO: remove when 4_submodel_MP_model.xlsx has an accounted mass fraction below the default
                    with EnvironUtils.temp_config_env([(['wc_sim', 'multialgorithm',
                                                         'max_allowed_init_accounted_fraction'],
                                                        str(3.0))]):
                        simulation_rv = self.run_simulation(wc_sim.Simulation(model), max_time=max_time, seed=seeds[i],
                                                            options=options)
                    print(f"Simulated {simulation_rv.num_events} events")
                    print(f"Results in '{simulation_rv.results_dir}'")
                except wc_sim.multialgorithm_errors.MultialgorithmError as e:
                    print(f"MultialgorithmError {e}")
