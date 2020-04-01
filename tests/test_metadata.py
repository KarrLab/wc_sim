""" Tests of metadata about a WC simulation run

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-03-29
:Copyright: 2020, Karr Lab
:License: MIT
"""

import os
import shutil
import tempfile
import unittest
import warnings

from de_sim.simulation_config import SimulationConfig
from wc_sim.metadata import WCSimulationMetadata
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.sim_config import WCSimulationConfig
from wc_utils.util.git import GitHubRepoForTests


class TestWCSimulationMetadata(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.time_max = 5
        self.de_simulation_config = SimulationConfig(time_max=self.time_max)
        self.wc_sim_config = WCSimulationConfig(self.de_simulation_config)
        self.wc_simulation_metadata = WCSimulationMetadata(self.wc_sim_config)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_init(self):
        self.assertEqual(self.wc_simulation_metadata.wc_sim_config.de_simulation_config.time_max, self.time_max)

        with self.assertRaisesRegex(MultialgorithmError, 'must be a RepositoryMetadata'):
            self.wc_simulation_metadata.wc_simulator_repo = 1

    def test_set_wc_model_repo(self):
        test_repo_name = 'test_wc_sim_metadata'
        github_repo = GitHubRepoForTests(test_repo_name)
        repo = github_repo.make_test_repo(self.tempdir)

        # test with model path in a git repo
        file_in_git_repo = os.path.join(self.tempdir, 'README.md')
        wc_simulation_metadata = WCSimulationMetadata(self.wc_sim_config)
        wc_simulation_metadata.set_wc_model_repo(file_in_git_repo)
        self.assertIn(test_repo_name, wc_simulation_metadata.wc_model_repo.url)

        # test with model path not in a git repo
        wc_simulation_metadata = WCSimulationMetadata(self.wc_sim_config)
        with warnings.catch_warnings(record=True) as w:
            wc_simulation_metadata.set_wc_model_repo('/')
            self.assertIn("Cannot obtain metadata for git repo containing model", str(w[-1].message))
        self.assertEqual(wc_simulation_metadata.wc_model_repo, None)

        # delete the repo
        github_repo.delete_test_repo()

    def test_get_pathname(self):
        self.assertEqual(os.path.basename(WCSimulationMetadata.get_pathname(self.tempdir)),
                         'wc_simulation_metadata.pickle')
