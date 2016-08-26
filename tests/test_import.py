import unittest

class TestImport(unittest.TestCase):

    def test_import(self):
        # test whether importing scipy works
        try:
            from scipy.misc import doccer
        except ImportError as e:
            self.assertTrue(False, msg='ImportError: ' + str(e) )
