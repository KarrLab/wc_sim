import unittest
import sys

# test some imports, which have been causing installation problems

class TestImports(unittest.TestCase):

    def test_imports(self):
        imports = ["from matplotlib import pyplot, ticker", 
            # "import no_such_package",  
            "from cobra.test import test_all",
            ]
        failed = False
        for cmd in imports:
            try:
                exec( cmd )
            except ImportError as e:
                failed = True
                sys.stderr.write( "ImportError while executing '{}': {}\n\n".format( cmd, e ) )
                
        if failed:
            self.fail( msg="at least one import failed" )
