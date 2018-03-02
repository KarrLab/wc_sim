""" Log metadata tests

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-08-18
:Copyright: 2016, Karr Lab
:License: MIT
"""

import socket
import unittest
import wc_sim.sim_config
import wc_sim.sim_metadata


class TestMetadata(unittest.TestCase):

    def setUp(self):
        model = wc_sim.sim_metadata.ModelMetadata.create_from_repository()

        changes = [
            wc_sim.sim_config.Change(target='rxn-1.km', value=1),
            wc_sim.sim_config.Change(target='species-1', value=1),
        ]
        perturbations = [
            wc_sim.sim_config.Perturbation(wc_sim.sim_config.Change(target='rxn-1.km', value=1), start_time=5),
            wc_sim.sim_config.Perturbation(wc_sim.sim_config.Change(target='species-1', value=1), start_time=0, end_time=10),
        ]
        simulation = wc_sim.sim_config.SimulationConfig(time_max=100, time_step=1, changes=changes,
                                                    perturbations=perturbations, random_seed=1)

        ip_address = socket.gethostbyname(socket.gethostname())
        author = wc_sim.sim_metadata.AuthorMetadata(name='Test user', email='test@test.com',
                                                organization='Test organization', ip_address=ip_address)

        run = wc_sim.sim_metadata.RunMetadata()
        run.record_ip_address()

        self.metadata = wc_sim.sim_metadata.Metadata(model, simulation, run, author)

    def test_build_metadata(self):
        model = self.metadata.model
        self.assertIn(model.url, ['https://github.com/KarrLab/wc_sim.git',
                                  'git@github.com:KarrLab/wc_sim.git',
                                  'ssh://git@github.com/KarrLab/wc_sim.git'])
        self.assertEqual(model.branch, 'master')

        run = self.metadata.run
        run.record_start()
        run.record_end()
        self.assertGreaterEqual(run.run_time, 0)
