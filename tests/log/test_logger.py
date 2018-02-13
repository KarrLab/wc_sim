""" Logging tests

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-08-17
:Copyright: 2016, Karr Lab
:License: MIT
"""

import unittest
# import wc.sim.core
import wc_lang.core
import wc_sim.log.core
import wc_sim.sim_config


class TestLogging(unittest.TestCase):
    """ Tests of the complete example model """

    def setUp(self):
        # build model
        self.model = wc_lang.core.Model()

        # build simulation configuration
        self.sim_config = wc_sim.sim_config.SimulationConfig(time_max=10, time_step=1)

    @unittest.skip('skipping temporarily')
    def test_logger(self):
        simulator = wc.sim.core.Simulator(self.model)
        simulator.run(self.sim_config)


class SimulatorExample(object):

    def __init__(self):
        self.processes = [
            ExampleProcess(id='proc-1'),
            ExampleProcess(id='proc-2'),
        ]

    def run(self, time_max, time_step, log_path):
        # initialize logs
        log_writers = []
        for process in self.processes:
            log_writer = wc_sim.log.core.Writer()
            log_writer.start()
            log_writers.append(log_writer)

        # simulate and log results
        num_time_steps = int(self.time_max / self.time_step)
        for i_time_step in range(num_time_steps):
            time += time_step
            for process, log_writer in zip(self.processes, log_writers):
                process.integrate(time_step, log_writer)

        # finalize logs
        for log_writer in log_writers:
            log_writer.close()


class ExampleProcess(object):

    def __init__(self, id):
        self.id = id

    def integrate(self, log_writer):
        pass
