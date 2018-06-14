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
import warnings

from wc_lang.core import SpeciesType
from wc_sim import __main__
from wc_sim.multialgorithm.__main__ import SimController
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.core.sim_metadata import SimulationMetadata


class SimControllerTestCase(unittest.TestCase):

    def setUp(self):
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           '2_species_1_reaction.xlsx')
        self.checkpoints_dir = tempfile.mkdtemp()
        self.args = Namespace(
            model_file=self.MODEL_FILENAME,
            end_time=100,
            checkpoint_period=4,
            checkpoints_dir=self.checkpoints_dir,
            fba_time_step=5
        )

    def tearDown(self):
        shutil.rmtree(self.checkpoints_dir)

    # @unittest.skip("Fails when simulation writes to stdout, as when debugging")
    def test_app_run(self):
        argv = [
            'sim',
            self.MODEL_FILENAME,
            '10',
            '--checkpoint-period', '2',
            '--results-dir', self.checkpoints_dir,
            '--fba-time-step', '5',
        ]
        with __main__.App(argv=argv) as app:
            with CaptureOutput(relay=False) as capturer:
                app.run()
                events = re.search('^Simulated (\d+) events', capturer.get_text())
                results = re.search("Saved checkpoints and run results in '(.*?)'$", capturer.get_text())
        num_events = int(events.group(1))
        results_dir = results.group(1)
        self.assertTrue(0 < num_events)
        self.assertTrue(results_dir.startswith(self.checkpoints_dir))
