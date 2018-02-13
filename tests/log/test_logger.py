""" Logging tests

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-08-17
:Copyright: 2016, Karr Lab
:License: MIT
"""

import math
import mock
import numpy
import os
import shutil
import tempfile
import unittest
import wc_sim.log.core
import wc_sim.sim_config

# .. todo :: test with the actual simulator


class TestLogging(unittest.TestCase):
    """ Tests of the complete example model """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_ExampleSimulator(self):
        # build simulation configuration
        sim_config = wc_sim.sim_config.SimulationConfig(time_max=10., time_step=1.)

        # run simulation and log results
        log_path = os.path.join(self.dir, 'log.pickle')
        simulator = ExampleSimulator()
        simulator.run(sim_config, log_path)

        # assertions
        results = wc_sim.log.core.Reader().run(log_path)

        numpy.testing.assert_array_equal(results['time'], numpy.arange(0., 11., 1.).reshape((1, 1, 11)))

        numpy.testing.assert_array_equal(results['volume'][:, :, 0], numpy.full((1, 2), 1.))
        numpy.testing.assert_array_equal(results['volume'][:, :, -1], numpy.full((1, 2), 2.))

        numpy.testing.assert_array_equal(results['growth'][:, :, 0], numpy.full((1, 2), 1.))
        numpy.testing.assert_array_equal(results['growth'][:, :, -1], numpy.full((1, 2), 2.))

        self.assertEqual(results['species_counts'].shape, (5, 2, 11))


class ExampleSimulator(object):

    def run(self, sim_config, log_path):
        time_max = sim_config.time_max
        time_step = sim_config.time_step
        num_time_steps = sim_config.get_num_time_steps()

        state = mock.Mock(
            time=0.,
            volume=numpy.ones((1, 2)),
            growth=numpy.ones((1, 2)),
            species_counts=numpy.random.random_integers(0, 100, size=(5, 2)),
        )

        # initialize log
        writer = wc_sim.log.core.Writer(state, num_time_steps, log_path)
        writer.start()

        # simulate and log results
        time = 0.
        for i_time_step in range(num_time_steps):
            time += time_step

            state.time = time
            state.volume = numpy.full((1, 2), math.exp(math.log(2) * time / time_max))
            state.growth = numpy.full((1, 2), math.exp(math.log(2) * time / time_max))
            state.species_counts = numpy.random.random_integers(0, 100, size=(5, 2))

            writer.append(time)

        # finalize logs
        writer.close()
