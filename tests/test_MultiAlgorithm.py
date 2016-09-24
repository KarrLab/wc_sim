import unittest
import warnings

from argparse import Namespace

from Sequential_WC_Simulator.multialgorithm.config import WC_SimulatorConfig
from Sequential_WC_Simulator.multialgorithm.multi_algorithm import MultiAlgorithm

class TestMultiAlgorithm(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        

    def test_reproducibility(self):
        # model predictions should be equal because they use the same seeds
        num_FBA_time_steps = 100
        args = Namespace(FBA_time_step=WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP, debug=False, 
            end_time=num_FBA_time_steps*WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP, 
            model_filename='./test_data/Model.xlsx',
            output_directory=WC_SimulatorConfig.DEFAULT_OUTPUT_DIRECTORY,
            seed=123)
        history1 = MultiAlgorithm.main( args ).the_SharedMemoryCellState.report_history()
        history2 = MultiAlgorithm.main( args ).the_SharedMemoryCellState.report_history()
        self.assertEqual( history1, history2 )

    def test_not_reproducible(self):
        # model predictions should not be equal because they use different seeds
        num_FBA_time_steps = 10
        args = Namespace(FBA_time_step=WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP, debug=False, 
            end_time=num_FBA_time_steps*WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP, 
            model_filename='./test_data/Model.xlsx',
            output_directory=WC_SimulatorConfig.DEFAULT_OUTPUT_DIRECTORY,
            seed=None)
        history1 = MultiAlgorithm.main( args ).the_SharedMemoryCellState.report_history()
        history2 = MultiAlgorithm.main( args ).the_SharedMemoryCellState.report_history()
        self.assertNotEqual( history1, history2 )


