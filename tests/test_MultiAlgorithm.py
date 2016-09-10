import unittest
'''
Run a brief simulation, to ensure that it runs
'''

import sys
print(sys.version)
from argparse import Namespace

from Sequential_WC_Simulator.multialgorithm.config import WC_SimulatorConfig
from Sequential_WC_Simulator.multialgorithm.multi_algorithm import MultiAlgorithm

class TestMultiAlgorithm(unittest.TestCase):

    def test_run(self):
    	# TODO: make model_filename more portable
        args = Namespace(FBA_time_step=WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP, debug=True, 
            end_time=3*WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP, 
            model_filename='./test_data/Model.xlsx',
            output_directory=WC_SimulatorConfig.DEFAULT_OUTPUT_DIRECTORY,
            plot=False, seed=123)
        MultiAlgorithm.main( args ) 


