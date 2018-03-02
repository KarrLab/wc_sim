"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-02-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import unittest
import warnings
import random

from argparse import Namespace
from examples.phold import RunPhold
from wc_sim.core.debug_logs import config
# TODO(Arthur): combine tests/examples/test_* into one file


class TestPhold(unittest.TestCase):

    def setUp(self):
        # turn off console logging
        self.console_level = config['debug_logs']['loggers']['wc.debug.console']['level']
        config['debug_logs']['loggers']['wc.debug.console']['level'] = 'ERROR'
        warnings.simplefilter("ignore")
        # TODO(Arthur): also turn off logging via from examples.debug_logs import logs

    def tearDown(self):
        # restore console logging
        config['debug_logs']['loggers']['wc.debug.console']['level'] = self.console_level

    def run_phold(self, seed, end_time):
        args = Namespace(end_time=end_time, frac_self_events=0.3, num_phold_procs=10, seed=seed)
        random.seed(seed)
        return(RunPhold.main(args))

    def test_phold_reproducibility(self):
        num_events1=self.run_phold(123, 10)
        num_events2=self.run_phold(123, 10)
        self.assertEqual(num_events1, num_events2)

        num_events2=self.run_phold(173, 10)
        self.assertNotEqual(num_events1, num_events2)

    def test_phold_parse_args(self):
        num_procs = 3
        frac_self = 0.2
        end_time = 25.0
        seed = 1234
        cl = "{} {} {}".format(num_procs, frac_self, end_time)
        args = RunPhold.parse_args(cli_args=cl.split())
        self.assertEqual(args.num_phold_procs, num_procs)
        self.assertEqual(args.frac_self_events, frac_self)
        self.assertEqual(args.end_time, end_time)
        cl = "{} {} {} --seed {}".format(num_procs, frac_self, end_time, seed)
        args = RunPhold.parse_args(cli_args=cl.split())
        self.assertEqual(args.seed, seed)
