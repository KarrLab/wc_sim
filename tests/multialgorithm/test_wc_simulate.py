""" Test wc_sim main program

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 20180511
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import sys
from copy import copy
from capturer import CaptureOutput

from wc_sim.multialgorithm.wc_simulate import RunSimulation


class TestRunSimulation(unittest.TestCase):
    
    def make_args(self, args_dict, include=None):
        options = ['checkpoint_period', 'checkpoints_dir', 'FBA_time_step', 'num_simulations']
        args = []
        for opt in options:
            if opt in args_dict:
                args.append('--' + opt)
                args.append(str(args_dict[opt]))
        args.extend([args_dict['model_file'], str(args_dict['end_time'])])
        return args

    def test_parse_args(self):
        arguments = dict(
            model_file='wc_lang_model.xlsx',
            end_time=100,
            checkpoint_period=3,
            checkpoints_dir='/tmp/foo',
            FBA_time_step=5.5,
            num_simulations=3,
        )
        args = self.make_args(arguments)
        parsed_args = RunSimulation.parse_args(args)
        for arg,value in arguments.items():
            self.assertEqual(getattr(parsed_args, arg), value)

        # test error detection
        errors = dict(
            end_time = [-3, 0],
            checkpoint_period = [-2, 0, arguments['end_time'] + 1],
            FBA_time_step = [-2, 0, arguments['end_time'] + 1],
            num_simulations = [-1],
        )
        with CaptureOutput(relay=False):
            print('\n--- testing RunSimulation.parse_args() error handling ---', file=sys.stderr)
            for arg,error_vals in errors.items():
                for error_val in error_vals:
                    arguments2 = copy(arguments)
                    arguments2[arg] = error_val
                    args = self.make_args(arguments2)
                    with self.assertRaises(SystemExit):
                        RunSimulation.parse_args(args)
            print('--- done testing RunSimulation.parse_args() error handling ---', file=sys.stderr)

    # TODO(Arthur)
    def test_run(self):
        pass
