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
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError


class SimControllerTestCase(unittest.TestCase):

    def setUp(self):
        SpeciesType.objects.reset()
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           '2_species_1_reaction.xlsx')
        self.checkpoints_dir = tempfile.mkdtemp()
        tmp = os.path.expanduser('~/tmp')
        if not os.path.isdir(tmp):
            os.makedirs(tmp)
        self.user_tmp_dir = tempfile.mkdtemp(dir=tmp)
        self.args = Namespace(
            model_file=self.MODEL_FILENAME,
            end_time=100,
            checkpoint_period=3,
            checkpoints_dir=self.checkpoints_dir,
            dataframe_file=os.path.join(self.checkpoints_dir, 'dataframe_file.h5'),
            fba_time_step=5
        )

    def tearDown(self):
        shutil.rmtree(self.checkpoints_dir)
        shutil.rmtree(self.user_tmp_dir)

    def test_process_and_validate_args1(self):
        original_args = copy(self.args)
        SimController.process_and_validate_args(self.args)
        for arg in ['model_file', 'end_time', 'checkpoint_period', 'fba_time_step']:
            self.assertEqual(getattr(original_args, arg), self.args.__dict__[arg])
        for arg in ['checkpoints_dir', 'dataframe_file']:
            self.assertTrue(self.args.__dict__[arg].startswith(getattr(original_args, arg)))

        self.args.dataframe_file = os.path.join(self.checkpoints_dir, 'dataframe_file_no_suffix')
        original_args = copy(self.args)
        SimController.process_and_validate_args(self.args)
        self.assertEqual(self.args.dataframe_file, original_args.dataframe_file + '.h5')

    def test_process_and_validate_args2(self):
        # test files specified relative to home directory
        relative_tmp_dir = os.path.join('~/tmp/', os.path.basename(self.user_tmp_dir))
        self.args.checkpoints_dir=relative_tmp_dir
        self.args.dataframe_file=os.path.join(relative_tmp_dir, 'dataframe_file.h5')
        SimController.process_and_validate_args(self.args)
        for arg in ['checkpoints_dir', 'dataframe_file']:
            self.assertIn(getattr(self.args, arg).replace('~', ''), self.args.__dict__[arg])

    def test_process_and_validate_args3(self):
        # test no files
        self.args.checkpoints_dir=None
        self.args.dataframe_file=None
        SimController.process_and_validate_args(self.args)
        for arg, value in self.args.__dict__.items():
            self.assertEqual(getattr(self.args, arg), value)

    def test_process_and_validate_args4(self):
        # test error detection
        errors = dict(
            end_time=[-3, 0],
            checkpoint_period=[-2, 0, self.args.end_time + 1],
            fba_time_step=[-2, 0, self.args.end_time + 1],
        )
        for arg, error_vals in errors.items():
            for error_val in error_vals:
                bad_args = copy(self.args)
                setattr(bad_args, arg, error_val)
                with self.assertRaises(ValueError):
                    SimController.process_and_validate_args(bad_args)

    def test_process_and_validate_args5(self):
        # test dataframe_file requires checkpoints_dir
        self.args.checkpoints_dir=None
        with self.assertRaisesRegexp(ValueError,
            'dataframe_file cannot be specified unless checkpoints_dir is provided'):
            SimController.process_and_validate_args(self.args)

    # @unittest.skip("Fails when simulation writes to stdout, as when debugging")
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

    def test_simulate_wo_output_files(self):
        self.args.checkpoints_dir = None
        self.args.dataframe_file = None
        num_events, results_dir = self.run_simulate(self.args)
        self.assertTrue(0 < num_events)
        self.assertEqual(results_dir, None)

    def test_simulate(self):
        num_events, results_dir = self.run_simulate(self.args)

        # check time, and simulation config in checkpoints
        for time in Checkpoint.list_checkpoints(results_dir):
            ckpt = Checkpoint.get_checkpoint(results_dir, time=time)
            self.assertEqual(time, ckpt.time)
            self.assertEqual(ckpt.metadata.simulation.time_init, 0)
            self.assertEqual(ckpt.metadata.simulation.time_max, self.args.end_time)
            self.assertEqual(ckpt.metadata.simulation.time_step, self.args.fba_time_step)
            self.assertTrue(ckpt.random_state != None)

        # TODO(Arthur): check sequence of checkpoints, and checkpoint and dataframe contents
