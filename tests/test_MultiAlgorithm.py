import unittest

class TestMultiAlgorithm(unittest.TestCase):

    def setUp(self):
        from Sequential_WC_Simulator.multialgorithm.multi_algorithm import MultiAlgorithm

    def test_run(self):
        args = "foo"
        # TODO: RIGHT args
        MultiAlgorithm.main( args ) 

