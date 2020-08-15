""" 
:Author: Yin Hoon Chew <yinhoon.chew@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-08-13
:Copyright: 2016-2020, Karr Lab
:License: MIT
"""

from capturer import CaptureOutput
from collections import namedtuple
from inspect import currentframe, getframeinfo
from pprint import pprint
from scipy.constants import Avogadro
import cProfile
import datetime
import math
import numpy as np
import os
import pandas
import pstats
import re
import shutil
import tempfile
import unittest

from wc_onto import onto as wc_onto
from wc_sim.run_results import RunResults
from wc_sim.testing.verify import (VerificationError, VerificationTestCaseType, 
                                   VerificationResultType, VerificationRunResult)
from wc_sim.testing.verify_fba import (FbaVerificationTestReader, FbaResultsComparator, 
									   FbaCaseVerifier, FbaVerificationSuite)
import obj_tables
import wc_lang
import wc_sim.submodels.dfba as dfba


TEST_CASES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fixtures',
                                          'verification', 'testing'))

def make_test_case_dir(test_case_num, test_case_type='FLUX_BALANCE_STEADY_STATE'):
    return os.path.join(TEST_CASES, VerificationTestCaseType[test_case_type].value, test_case_num)

def make_verification_test_reader(test_case_num, test_case_type):
    return FbaVerificationTestReader(test_case_type, make_test_case_dir(test_case_num, test_case_type),
                                  test_case_num)


class TestFbaVerificationTestReader(unittest.TestCase):

    def test_read_settings(self):
        settings = make_verification_test_reader('01186', 'FLUX_BALANCE_STEADY_STATE').read_settings()
        some_expected_settings = dict(
            variables=['R01', 'R26', 'OBJF'],
            absolute=0.001,
            relative=0.001
        )
        for expected_key, expected_value in some_expected_settings.items():
            self.assertEqual(settings[expected_key], expected_value)
        settings = make_verification_test_reader('00003', 'FLUX_BALANCE_STEADY_STATE').read_settings()
        self.assertEqual(settings['key1'], 'value1: has colon')
        self.assertEqual(
            make_verification_test_reader('00004', 'FLUX_BALANCE_STEADY_STATE').read_settings()['variables'], 
            ['R01', 'R07', 'R10', 'OBJF', 'R26', 'R25'])

        with self.assertRaisesRegexp(VerificationError,
            re.escape("VerificationTestCaseType is 'DISCRETE_STOCHASTIC' and not 'FLUX_BALANCE_STEADY_STATE'")):
            make_verification_test_reader('00004', 'DISCRETE_STOCHASTIC')

    def test_read_expected_predictions(self):
        verification_test_reader = make_verification_test_reader('01186', 'FLUX_BALANCE_STEADY_STATE')
        verification_test_reader.settings = verification_test_reader.read_settings()
        expected_predictions_df = verification_test_reader.read_expected_predictions()
        self.assertTrue(isinstance(expected_predictions_df, pandas.core.frame.DataFrame))

        # wrong columns
        missing_variable = 'MissingVariable'        
        verification_test_reader = make_verification_test_reader('01186', 'FLUX_BALANCE_STEADY_STATE')
        verification_test_reader.settings = verification_test_reader.read_settings()
        verification_test_reader.settings['variables'].append(missing_variable)
        with self.assertRaisesRegexp(VerificationError, 
        	"some variables missing from expected predictions '.*01186-results.csv': {{'{}'}}".format(missing_variable)):
            verification_test_reader.read_expected_predictions()

    def test_read_model(self):
        verification_test_reader = make_verification_test_reader('01186', 'FLUX_BALANCE_STEADY_STATE')
        model = verification_test_reader.read_model()
        self.assertTrue(isinstance(model, wc_lang.Model))
        self.assertEqual(model.id, 'test_case_' + verification_test_reader.test_case_num)
        self.assertEqual(len(model.species), 23)
        self.assertEqual(len(model.compartments), 1)
        self.assertEqual(len(model.submodels), 1)
        self.assertEqual(model.submodels[0].id, verification_test_reader.test_case_num + '-fba')
        self.assertEqual(model.submodels[0].framework, wc_onto['WC:dynamic_flux_balance_analysis'])
        self.assertEqual(len(model.submodels[0].reactions), 26)
        self.assertEqual(model.reactions.get_one(id='R01').flux_bounds.min, 0.)
        self.assertEqual(model.reactions.get_one(id='R01').flux_bounds.max, 1.)
        self.assertEqual(model.reactions.get_one(id='R02').flux_bounds.min, -1000.)
        self.assertEqual(model.reactions.get_one(id='R02').flux_bounds.max, 1000.)


        verification_test_reader = make_verification_test_reader('00003', 'FLUX_BALANCE_STEADY_STATE')
        with self.assertRaisesRegexp(VerificationError, 
            "Test case model file '.*00003-sbml-l3v2.xml' does not exists"):
            verification_test_reader.read_model()
