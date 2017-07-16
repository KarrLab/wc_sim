'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-01
:Copyright: 2016, Karr Lab
:License: MIT
'''
import os
import unittest
import warnings

import pytest
'''
OLD code; to be discarded:
from argparse import Namespace

from wc_sim.multialgorithm.config import paths as config_paths
from wc_sim.multialgorithm.multi_algorithm import MultiAlgorithm
from wc_sim.multialgorithm.multialgorithm_errors import NegativePopulationError
from wc_utils.config.core import ConfigManager

config = ConfigManager(config_paths.core).get_config()['wc_sim']['multialgorithm']
'''

@pytest.mark.skip(reason="no longer works")
class TestMultiAlgorithm(unittest.TestCase):
    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'Mycoplasma pneumoniae.xlsx')

    def setUp(self):
        warnings.simplefilter("ignore")        

    @unittest.skip("skip while negative population predictions are being fixed")
    def test_reproducibility(self):
        # model predictions should be equal because they use the same seeds
        num_fba_time_steps = 100
        args = Namespace(FBA_time_step=config['default_fba_time_step'],
            end_time=num_fba_time_steps*config['default_fba_time_step'], 
            model_filename=self.MODEL_FILENAME,
            output_directory=config['default_output_directory'],
            seed=123)
        history1 = MultiAlgorithm.main( args ).local_species_population.report_history()
        history2 = MultiAlgorithm.main( args ).local_species_population.report_history()
        self.assertEqual( history1, history2 )

    @unittest.skip("skip while negative population predictions are being fixed")
    def test_not_reproducible(self):
        # model predictions should not be equal because they use different seeds
        num_fba_time_steps = 10
        args = Namespace(FBA_time_step=config['default_fba_time_step'],
            end_time=num_fba_time_steps*config['default_fba_time_step'], 
            model_filename=self.MODEL_FILENAME,
            output_directory=config['default_output_directory'],
            seed=None)
        history1 = MultiAlgorithm.main( args ).local_species_population.report_history()
        history2 = MultiAlgorithm.main( args ).local_species_population.report_history()
        self.assertNotEqual( history1, history2 )
    
    @unittest.skip("skip while refactoring")
    def test_loads_model_and_initialize_simulation(self):
        """ Test model loading and simulation
        
        Not a test; just try to load and initialize the simulation, while refactoring
        """
        #TODO: delete this test once above tests are working

        num_fba_time_steps = 1
        args = Namespace(FBA_time_step=config['default_fba_time_step'],
                         end_time=num_fba_time_steps * config['default_fba_time_step'], 
                         model_filename=self.MODEL_FILENAME,
                         output_directory=config['default_output_directory'],
                         seed=123)
        MultiAlgorithm.initialize_simulation(args)
