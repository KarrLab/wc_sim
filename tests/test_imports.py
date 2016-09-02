import unittest
import sys
import traceback

# test some imports which have been causing installation problems; feel free to add statements to 'imports'

class TestImports(unittest.TestCase):

    def test_imports(self):
        imports = ["from matplotlib import pyplot, ticker", 
            "import no_such_package",  
            "from cobra.test import test_all",
            ]
        failures = []
        tracebacks = []
        for cmd in imports:
            try:
                exec( cmd )
            except ImportError as e:
                (type, value, tb) = sys.exc_info()
                failures.append(cmd)
                tracebacks.append(tb)
                sys.stderr.write( "ImportError while executing '{}': {}\n\n".format( cmd, e ) )
                
        if failures:
            failures_and_tracebacks = []
            for (f,tb) in zip( failures, tracebacks ):
                failures_and_tracebacks.append( 
                    "Traceback from '{}':\n{}".format(f, ''.join( traceback.format_tb( tb ) ) ) )
            self.fail( msg="Failing imports: {}".format( '\n'.join( failures_and_tracebacks ) )  )
