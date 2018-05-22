""" Test RunResults

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import unittest
import shutil
import tempfile
from argparse import Namespace
import warnings

from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.__main__ import SimController
from wc_sim.multialgorithm.run_results import RunResults


class TestRunResults(unittest.TestCase):

    def setUp(self):
        self.CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'checkpoints_dir')
        self.checkpoints_dir = tempfile.mkdtemp()
        self.checkpoints_copy = os.path.join(self.checkpoints_dir, 'checkpoints_copy')
        shutil.copytree(self.CHECKPOINTS_DIR, self.checkpoints_copy)
        self.args = Namespace(
            model_file='filename',
            end_time=10,
            checkpoint_period=3,
            checkpoints_dir=self.checkpoints_dir,
            fba_time_step=5
        )
        self.metadata = SimController.create_metadata(self.args)

    def tearDown(self):
        shutil.rmtree(self.checkpoints_dir)

    def test_run_results(self):

        run_results_1 = RunResults(self.checkpoints_copy, self.metadata)
        for component in RunResults.COMPONENTS:
            rv = run_results_1.get(component)
            # TODO(Arthur): check contents of returned component
            self.assertTrue(rv is not None)
        run_results_2 = RunResults(self.checkpoints_copy)
        for component in RunResults.COMPONENTS:
            self.assertTrue(run_results_1.get(component).equals(run_results_2.get(component)))

    def test_run_results_errors(self):

        with self.assertRaises(MultialgorithmError):
            RunResults(self.checkpoints_copy)
        run_results = RunResults(self.checkpoints_copy, self.metadata)
        with self.assertRaisesRegexp(MultialgorithmError, "component '.*' is not an element of "):
            run_results.get('not_a_component')
