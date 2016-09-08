import unittest

class TestMultiAlgorithm(unittest.TestCase):

    def setUp(self):
        from Sequential_WC_Simulator.multialgorithm.config import WC_SimulatorConfig
        from Sequential_WC_Simulator.multialgorithm.multi_algorithm import MultiAlgorithm

    def test_run(self):
        args = Namespace(FBA_time_step=WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP, debug=True, 
            end_time=None, model_filename='data',
            output_directory=WC_SimulatorConfig.DEFAULT_OUTPUT_DIRECTORY,
            plot=False, seed=None)
        MultiAlgorithm.main( args ) 


