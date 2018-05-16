""" Test wc_sim main program

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-05-11
:Copyright: 2018, Karr Lab
:License: MIT
"""

import sys
import os
import unittest
import shutil
import tempfile
from copy import copy
from capturer import CaptureOutput
from argparse import Namespace

from wc_lang.core import SpeciesType
from wc_sim.multialgorithm.wc_simulate import RunSimulation
from tests.utilities_for_testing import make_args


class TestRunSimulation(unittest.TestCase):
    
    required = ['model_file', 'end_time']
    options = ['checkpoint_period', 'checkpoints_dir', 'dataframe_file', 'FBA_time_step']

    def setUp(self):
        SpeciesType.objects.reset()
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
            '2_species_1_reaction.xlsx')
        self.checkpoints_dir = tempfile.mkdtemp()

    def tearDown(self):
        # shutil.rmtree(self.checkpoints_dir)
        pass

    def test_parse_args(self):
        arguments = dict(
            model_file='wc_lang_model.xlsx',
            end_time=100,
            checkpoint_period=3,
            checkpoints_dir=self.checkpoints_dir,
            dataframe_file=os.path.join(self.checkpoints_dir, 'dataframe_file.h5'),
            FBA_time_step=5.5
        )
        args = make_args(arguments, self.required, self.options)
        parsed_args = RunSimulation.parse_args(args)
        for arg,value in arguments.items():
            self.assertEqual(getattr(parsed_args, arg), value)

        # test error detection
        errors = dict(
            end_time = [-3, 0],
            checkpoint_period = [-2, 0, arguments['end_time'] + 1],
            FBA_time_step = [-2, 0, arguments['end_time'] + 1],
        )
        with CaptureOutput(relay=False):
            print('\n--- testing RunSimulation.parse_args() error handling ---', file=sys.stderr)
            for arg,error_vals in errors.items():
                for error_val in error_vals:
                    arguments2 = copy(arguments)
                    arguments2[arg] = error_val
                    args = make_args(arguments2, self.required, self.options)
                    with self.assertRaises(SystemExit):
                        RunSimulation.parse_args(args)

            # test dataframe_file requires checkpoints_dir
            arguments = dict(
                model_file='wc_lang_model.xlsx',
                end_time=100,
                dataframe_file='dataframe_file.h5'
            )
            args = make_args(arguments, self.required, self.options)
            with self.assertRaises(SystemExit):
                RunSimulation.parse_args(args)
            print('--- done testing RunSimulation.parse_args() error handling ---', file=sys.stderr)

    def test_run(self):
        args = Namespace(
            model_file=self.MODEL_FILENAME,
            end_time=10,
            checkpoint_period=3,
            checkpoints_dir=self.checkpoints_dir,
            dataframe_file=None,
            # dataframe_file=os.path.join(self.checkpoints_dir, 'dataframe_file.h5'),
            FBA_time_step=5.5
        )
        print('args', args)
        res_dirname, num_events = RunSimulation.run(args)
        print('res_dirname, num_events', res_dirname, num_events)
        self.assertTrue(0 < num_events)
        self.assertTrue(res_dirname.startswith(self.checkpoints_dir))
