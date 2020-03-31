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

from de_sim.simulation_config import SimulationConfig
from wc_sim.metadata import WCSimulationMetadata
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.sim_config import WCSimulationConfig


class TestWCSimulationMetadata(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_init(self):
        time_max = 5
        de_simulation_config = SimulationConfig(time_max=time_max)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        wc_simulation_metadata = WCSimulationMetadata(wc_sim_config)
        self.assertEqual(wc_simulation_metadata.wc_sim_config.de_simulation_config.time_max, time_max)

        with self.assertRaisesRegex(MultialgorithmError, 'must be a RepositoryMetadata'):
            wc_simulation_metadata.wc_simulator_repo = 1

    def test_set_wc_model_repo(self):
        pass

    def test_get_pathname(self):
        self.assertEqual(os.path.basename(WCSimulationMetadata.get_pathname(self.tempdir)),
                         'wc_simulation_metadata.pickle')
