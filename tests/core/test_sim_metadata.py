""" Test simulation metadata object

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-08-18
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import socket
import unittest
import copy

from wc_sim.core import sim_config
from wc_sim.core.sim_metadata import SimulationMetadata, ModelMetadata, AuthorMetadata, RunMetadata
from wc_utils.util.misc import as_dict


class TestMetadata(unittest.TestCase):

    def setUp(self):
        self.model = model = ModelMetadata.create_from_repository()
        self.model_equal = ModelMetadata.create_from_repository()
        self.model_different = copy.copy(self.model_equal)
        self.model_different.branch = self.model_equal.branch + 'x'

        changes = [
            sim_config.Change(target='rxn-1.km', value=1),
            sim_config.Change(target='species-1', value=1),
        ]
        perturbations = [
            sim_config.Perturbation(sim_config.Change(target='rxn-1.km', value=1), start_time=5),
            sim_config.Perturbation(sim_config.Change(target='species-1', value=1), start_time=0, end_time=10),
        ]
        simulation = sim_config.SimulationConfig(time_max=100, time_step=1, changes=changes,
                                                    perturbations=perturbations, random_seed=1)

        self.author = author = AuthorMetadata(name='Test user', email='test@test.com',
            username='Test username', organization='Test organization')

        self.run = run = RunMetadata()
        run.record_start()
        run.record_ip_address()
        self.run_equal = copy.copy(run)
        self.run_different = copy.copy(run)
        self.run_different.record_end()

        self.metadata = SimulationMetadata(model, simulation, run, author)
        self.metadata_equal = SimulationMetadata(model, simulation, run, author)
        self.author_equal = copy.copy(author)
        self.author_different = author_different = copy.copy(author)
        author_different.name = 'Joe Smith'
        self.metadata_different = SimulationMetadata(model, simulation, run, author_different)

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

    def test_equality(self):
        obj = object()
        self.assertEqual(self.model, self.model_equal)
        self.assertNotEqual(self.model, obj)
        self.assertNotEqual(self.model, self.model_different)
        self.assertFalse(self.model != self.model_equal)

        self.assertEqual(self.run, self.run_equal)
        self.assertNotEqual(self.run, obj)
        self.assertNotEqual(self.run, self.run_different)
        self.assertFalse(self.run != self.run_equal)

        self.assertEqual(self.metadata, self.metadata_equal)
        self.assertNotEqual(self.metadata, obj)
        self.assertNotEqual(self.metadata, self.metadata_different)

        self.assertEqual(self.author, self.author_equal)
        self.assertNotEqual(self.author, obj)
        self.assertNotEqual(self.author, self.author_different)

    def test_as_dict(self):
        d = as_dict(self.metadata)
        self.assertEqual(d['author']['name'], self.metadata.author.name)
        self.assertEqual(d['model']['branch'], self.metadata.model.branch)
        self.assertEqual(d['run']['start_time'], self.metadata.run.start_time)
        self.assertEqual(d['simulation']['changes'], self.metadata.simulation.changes)

    def test_str(self):
        self.assertIn(self.metadata.author.name, str(self.metadata))
        self.assertIn(self.metadata.model.branch, str(self.metadata))
        self.assertIn(self.metadata.run.ip_address, str(self.metadata))
        self.assertIn(str(self.metadata.simulation.time_max), str(self.metadata))
