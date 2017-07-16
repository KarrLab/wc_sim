'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-06
:Copyright: 2017, Karr Lab
:License: MIT
'''

import unittest, os, sys, sysconfig
import time, importlib
from distutils.command.clean import clean
import shutil

import wc_sim
from wc_sim.on_ROSS.try_c_and_python import setup_python_call_c, setup_python_with_c_callback

def distutils_dir_name(dname):
    # Return the name of a distutils build directory; http://stackoverflow.com/a/14369968/509882
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)

# TODO(Arthur): make dependencies work on Circle/Docker
@unittest.skip("skip until dependencies work on Circle/Docker")
class TestTryCandPython(unittest.TestCase):

    def setUp(self):
        self.build_dir = os.path.join(os.path.dirname(wc_sim.__file__), '..', 'build')
        self.lib_dir = os.path.join(self.build_dir, distutils_dir_name('lib'))

    # test the Python extensions
    def test_python_call_c(self):
        # build the example Python extension
        self.distro = setup_python_call_c.call_setup()
        # sleep because it appears that setup returns before the commands it launches finishes
        time.sleep(2)
        sys.path.append(self.lib_dir)
        spam = importlib.import_module('spam')

        self.assertEqual(spam.system('date'), 0)
        for arg in [7, None]:
            with self.assertRaises(TypeError) as context:
                spam.system(arg)
            self.assertIn('argument 1 must be str', str(context.exception))
        self.assertEqual(spam.nothing(), None)

    def test_c_call_python(self):
        # build the example Python extension
        setup_python_with_c_callback.call_setup()
        time.sleep(2)
        sys.path.append(self.lib_dir)
        callbacks = importlib.import_module('callbacks')

        def f1():
            pass

        callbacks.set_callback(f1)
        self.assertEqual(callbacks.call_callback_simple(), None)

        # test callbacks with return values
        a = 3
        # test Python return of Integer, parsed by PyArg_Parse
        def f2(n):
            # test that Python receives the right argument from C
            self.assertEqual(n, a)
            return 1+n
        callbacks.set_callback(f2)
        self.assertEqual(callbacks.call_callback(a), None)

        # test Python return of a tuple, parsed by PyArg_ParseTuple
        def f3(n):
            # test that Python receives the right argument from C
            self.assertEqual(n, a)
            return (1+n,)
        callbacks.set_callback(f3)
        self.assertEqual(callbacks.call_callback(a), None)

        # test exception raised in C by PyArg_Parse for incorrect type
        def f4(n):
            return 'string'
        callbacks.set_callback(f4)
        with self.assertRaises(TypeError) as context:
            callbacks.call_callback(a)
        self.assertIn('an integer is required', str(context.exception))

    def tearDown(self):
        # todo: if the self.build_dir could be controlled, then remove it
        pass
