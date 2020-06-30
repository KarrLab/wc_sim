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

        # fixtures to test obtaining git metadata
        self.test_repo_name = 'test_wc_sim_metadata'
        self.github_repo = GitHubRepoForTests(self.test_repo_name)
        self.repo_dir = tempfile.mkdtemp(dir=self.tempdir)
        self.github_repo.make_test_repo(self.repo_dir)
        self.file_in_git_repo = os.path.join(self.repo_dir, 'README.md')
        self.wc_simulation_metadata.set_wc_model_repo(self.file_in_git_repo)

    def tearDown(self):
        shutil.rmtree(self.tempdir)
        # delete the repo
        self.github_repo.delete_test_repo()

    def test_init(self):
        self.assertEqual(self.wc_simulation_metadata.wc_sim_config.de_simulation_config.time_max, self.time_max)

        with self.assertRaisesRegex(MultialgorithmError, 'must be a RepositoryMetadata'):
            self.wc_simulation_metadata.wc_simulator_repo = 1

    def test__get_repo_metadata(self):
        # test with model path in a git repo
        metadata = self.wc_simulation_metadata._get_repo_metadata(self.file_in_git_repo)
        self.assertIn(self.test_repo_name, metadata.url)

        # test with model path not in a git repo
        with warnings.catch_warnings(record=True) as w:
            rv = self.wc_simulation_metadata._get_repo_metadata('/')
            self.assertIn("Cannot obtain metadata for git repo containing model", str(w[-1].message))
        self.assertEqual(rv, None)

    def test_set_wc_model_repo(self):
        # test with model path in a git repo
        self.assertIn(self.test_repo_name, self.wc_simulation_metadata.wc_model_repo.url)

        # test with model path not in a git repo
        with warnings.catch_warnings(record=True) as w:
            self.wc_simulation_metadata.set_wc_model_repo('/')
            self.assertIn("Cannot obtain metadata for git repo containing model", str(w[-1].message))
        self.assertEqual(self.wc_simulation_metadata.wc_model_repo, None)

    def test_get_pathname(self):
        self.assertEqual(os.path.basename(WCSimulationMetadata.get_pathname(self.tempdir)),
                         'wc_simulation_metadata.pickle')

    def test_semantically_equal(self):
        # WCSimulationMetadata instances with a repo
        wc_simulation_metadata_equal = WCSimulationMetadata(self.wc_sim_config)
        wc_simulation_metadata_equal.set_wc_model_repo(self.file_in_git_repo)
        self.assertTrue(self.wc_simulation_metadata.semantically_equal(wc_simulation_metadata_equal))

        wc_simulation_metadata_equal.wc_sim_config = WCSimulationConfig(SimulationConfig(time_max=1))
        self.assertFalse(self.wc_simulation_metadata.semantically_equal(wc_simulation_metadata_equal))
        wc_simulation_metadata_equal.wc_sim_config = self.wc_sim_config
        self.assertTrue(self.wc_simulation_metadata.semantically_equal(wc_simulation_metadata_equal))

        wc_simulation_metadata_equal.wc_model_repo = None
        self.assertFalse(self.wc_simulation_metadata.semantically_equal(wc_simulation_metadata_equal))
        wc_simulation_metadata_equal.set_wc_model_repo(self.file_in_git_repo)
        self.assertTrue(self.wc_simulation_metadata.semantically_equal(wc_simulation_metadata_equal))
