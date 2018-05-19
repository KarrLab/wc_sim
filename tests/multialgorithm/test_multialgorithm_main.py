""" Test multialgorithm main program

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-05-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import re
import shutil
import tempfile
import unittest
from argparse import Namespace
from capturer import CaptureOutput
from copy import copy

from wc_lang.core import SpeciesType
from wc_sim import __main__
from wc_sim.multialgorithm.__main__ import SimController
from wc_sim.log.checkpoint import Checkpoint


class SimControllerTestCase(unittest.TestCase):
    def setUp(self):
        SpeciesType.objects.reset()
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           '2_species_1_reaction.xlsx')
        self.checkpoints_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.checkpoints_dir)

    def test_parse_args(self):
        args = Namespace(
            model_file='wc_lang_model.xlsx',
            end_time=100,
            checkpoint_period=3,
            checkpoints_dir=self.checkpoints_dir,
            dataframe_file=os.path.join(self.checkpoints_dir, 'dataframe_file.h5'),
            fba_time_step=5.5
        )
        original_args = copy(args)
        SimController.process_and_validate_args(args)
        for arg, value in args.__dict__.items():
            self.assertEqual(getattr(original_args, arg), value)

        args.dataframe_file = os.path.join(self.checkpoints_dir, 'dataframe_file_no_suffix')
        original_args = copy(args)
        SimController.process_and_validate_args(args)
        self.assertEqual(args.dataframe_file, original_args.dataframe_file + '.h5')

        # test error detection
        errors = dict(
            end_time=[-3, 0],
            checkpoint_period=[-2, 0, args.end_time + 1],
            fba_time_step=[-2, 0, args.end_time + 1],
        )
        for arg, error_vals in errors.items():
            for error_val in error_vals:
                args2 = copy(args)
                setattr(args2, arg, error_val)
                with self.assertRaises(ValueError):
                    SimController.process_and_validate_args(args2)

        # test dataframe_file requires checkpoints_dir
        args = Namespace(
            model_file='wc_lang_model.xlsx',
            end_time=100,
            dataframe_file='dataframe_file.h5',
            checkpoint_period=1,
            checkpoints_dir=None,
            fba_time_step=10,
        )
        with self.assertRaisesRegexp(ValueError,
            'dataframe_file cannot be specified unless checkpoints_dir is provided'):
            SimController.process_and_validate_args(args)

        # TODO(Arthur): test files specified relative to home directory

    # @unittest.skip("Only works when simulation does not write to stdout")
    def test_app_run(self):
        argv = [
            'sim',
            self.MODEL_FILENAME,
            '10',
            '--checkpoint-period', '3',
            '--checkpoints-dir', self.checkpoints_dir,
            '--dataframe-file', os.path.join(self.checkpoints_dir, 'dataframe_file.h5'),
            '--fba-time-step', '5',
        ]
        with __main__.App(argv=argv) as app:
            with CaptureOutput(relay=False) as capturer:
                app.run()
                events = re.search('^Simulated (\d+) events', capturer.get_text())
                checkpoints = re.search("Saved chcekpoints in '(.*?)'$", capturer.get_text())
        num_events = int(events.group(1))
        results_dir = checkpoints.group(1)
        self.assertTrue(0 < num_events)
        self.assertTrue(results_dir.startswith(self.checkpoints_dir))

    def run_simulate(self, args):
        SimController.process_and_validate_args(args)
        with CaptureOutput(relay=False):
            return(SimController.simulate(args))

    def test_simulate(self):
        args = Namespace(checkpoint_period=4.0,
            checkpoints_dir=self.checkpoints_dir,
            dataframe_file=os.path.join(self.checkpoints_dir, 'dataframe_file.h5'),
            debug=False,
            end_time=10.0,
            fba_time_step=5.0,
            model_file=self.MODEL_FILENAME,
            suppress_output=False)
        num_events, results_dir = self.run_simulate(args)

        # check time, and simulation config in checkpoints
        for time in Checkpoint.list_checkpoints(results_dir):
            ckpt = Checkpoint.get_checkpoint(results_dir, time=time)
            self.assertEqual(time, ckpt.time)
            self.assertEqual(ckpt.metadata.simulation.time_init, 0)
            self.assertEqual(ckpt.metadata.simulation.time_max, args.end_time)
            self.assertEqual(ckpt.metadata.simulation.time_step, args.fba_time_step)
            self.assertTrue(ckpt.random_state != None)

        # TODO(Arthur): check # of checkpoints, right aggregate and metadata in checkpoints & dataframe
