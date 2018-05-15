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
from capturer import CaptureOutput

# turn off logging by changing config to raise the 'wc.debug.console', 'level' above DEBUG, to ERROR
# share this with phold by setting a temporary config environ variable & then importing phold which imports debug_logs
# to avoid any other side effects, run the import in a context that creates a temporary environment
from wc_utils.util.environ import EnvironUtils, MakeEnvironArgs
make_environ_args = MakeEnvironArgs()
make_environ_args.add_to_env(['debug_logs', 'loggers', 'wc.debug.console', 'level'], 'ERROR')
env = make_environ_args.get_env()
with EnvironUtils.make_temp_environ(**env):
    from examples.phold import RunPhold


class TestPhold(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def run_phold(self, seed, end_time):
        args = Namespace(end_time=end_time, frac_self_events=0.3, num_phold_procs=10, seed=seed)
        random.seed(seed)
        with CaptureOutput(relay=False):
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

    # TODO(Arthur): test parser error handling as done in test_wc_simulate.py
    def test_phold_parse_args_errors(self):
        pass
