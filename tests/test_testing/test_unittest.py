import unittest

from codeToTest import BaseClass
from codeToTest import Engine

# import someTestingFramework

"""
class MockClassToTest(BaseClass,unittest.TestCase):

    '''
    def __init__( self, ):
        self.testCase = unittest.TestCase( methodName='semiAbstractMethod' )
    '''

    def semiAbstractMethod( self, var ):
        super( ClassToTest, self ).semiAbstractMethod( var )
        # Do a test here: e.g., test that y = 3
        print 'test'
        y = 3
        self.assertEqual( y, 3 )

if __name__ == '__main__':
    try:
        # unittest.main()
        t = MockClassToTest()
        t.semiAbstractMethod() 
        
    except KeyboardInterrupt:
        pass
"""

class MockClassToTest(BaseClass):

    def semiAbstractMethod( self, var ):
        super( MockClassToTest, self ).semiAbstractMethod( var )
        # Do a test here: e.g., test that y = 3
        print 'test'
        y = 3
        # someTestingCommand( y == 3 )
        assert y == 3

class TestUnittest(unittest.TestCase):
    
    def test(self):
        s = MockClassToTest()
        s.semiAbstractMethod( 1 )
        Engine.schedule(1)
