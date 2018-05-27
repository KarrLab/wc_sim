""" Simulate a multialgorithm model

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-25
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import datetime

import wc_lang
from wc_lang.io import Reader
from wc_lang.core import Model
from wc_lang.prepare import PrepareModel, CheckModel
from wc_sim.core import sim_config
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.run_results import RunResults
from wc_sim.core.sim_metadata import SimulationMetadata, ModelMetadata, AuthorMetadata, RunMetadata

'''
usage:

from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.run_results import RunResults

    one run:
        model = ... model object or model file
        simulation = Simulation(model)
        events, results_dir = simulation.run(end_time, results_dir, checkpoint_period)
        run_results = RunResults(results_dir)

from wc_sim.core.sim_config import SimulationConfig

    batch run:
        model = ... model object or model file
        sim_config = SimulationConfig(...)
        simulation = Simulation(model, sim_config)
        events_array, results_dir = simulation.run_batch()
'''

class Simulation(object):
    """ Simulate a multialgorithm model

    Attributes:
        model_path (:obj:`str`): path to a file describing a `wc_lang` model
        model (:obj:`Model`): a `wc_lang` model description
        sim_config (:obj:`sim_config.SimulationConfig`): a simulation configuration
    """
    def __init__(self, model, sim_config=None):
        """
        Args:
            model (:obj:`str` or `Model`): either a path to file(s) describing a `wc_lang` model, or
                a `wc_lang.Model` instance
            sim_config (:obj:`sim_config.SimulationConfig`, optional): a simulation configuration

        Raises:
            :obj:`MultialgorithmError`: if `model` is invalid
        """
        if isinstance(model, Model):
            self.model = model
            self.model_path = None
        elif isinstance(model, str):
            # read model
            self.model_path = os.path.abspath(os.path.expanduser(model))
            self.model = Reader().run(self.model_path, strict=False)
            if self.model is None:
                raise MultialgorithmError("No model found in model file '{}'".format(self.model_path))
        else:
            raise MultialgorithmError("model must be a wc_lang Model or a pathname in a str, but its "
                "type is {}".format(type(model)))

        self.sim_config = sim_config

    def _prepare(self):
        """ Prepare simulation model and metadata
        """
        # prepare & check the model
        PrepareModel(self.model).run()
        CheckModel(self.model).run()

        # create metadata
        self.simulation_metadata = self._create_metadata()

    def _create_metadata(self):
        """ Obtain and assemble metadata for a simulation run

        Returns:
            :obj:`SimulationMetadata`: a metadata record for a simulation run;
                `SimulationMetadata.simulation` may need initialization;
                `SimulationMetadata.run_time` does need initialization
        """
        # model metadata
        model = ModelMetadata.create_from_repository()

        # author metadata
        try:
            username = getpass.getuser()
        except:     # pragma: no cover
            username = 'Unknown username'
        # TODO: collect more comprehensive and specific author information
        author = AuthorMetadata(name='Unknown name', email='Unknown email', username=username,
                                organization='Unknown organization')

        # run metadata
        run = RunMetadata()
        run.record_start()
        run.record_ip_address()

        if self.sim_config:
            return SimulationMetadata(model, self.sim_config, run, author)
        else:
            raise MultialgorithmError('Simulation configuration needed')

    def process_and_validate_args(self, args):
        """ Process and validate simulation arguments

        Supported arguments are results_dir, checkpoint_period, time_step, and end_time. end_time is
        required, while the others are optional.

        Args:
            args (:obj:`dict`): dictionary of arguments

        Raises:
            :obj:`MultialgorithmError`: if any of the arguments are invalid
        """
        # todo: remove checks which are redundant with SimulationConfig
        # process results directory
        if 'results_dir' in args:
            results_sup_dir = os.path.abspath(os.path.expanduser(args['results_dir']))

            # if results_sup_dir is a file, raise error
            if os.path.isfile(results_sup_dir):
                raise MultialgorithmError("results_dir ({}) is a file, not a dir".format(results_sup_dir))

            # if results_sup_dir does not exist, make it
            if not os.path.exists(results_sup_dir):
                os.makedirs(results_sup_dir)

            # make a time-stamped sub-dir for this run
            time_stamped_sub_dir = os.path.join(results_sup_dir, datetime.datetime.now().strftime(
                '%Y-%m-%d-%H-%M-%S'))
            if os.path.exists(time_stamped_sub_dir):
                raise MultialgorithmError("timestamped sub-directory of results_dir ({}) already exists".format(
                    time_stamped_sub_dir))
            else:
                os.makedirs(time_stamped_sub_dir)
            # update results_dir
            args['results_dir'] = time_stamped_sub_dir

        # validate args
        if 'end_time' not in args:
            raise MultialgorithmError("Simulation end time, end_time, must be provided")

        if args['end_time'] <= 0:
            raise MultialgorithmError("End time ({}) must be positive".format(args['end_time']))

        if 'checkpoint_period' in args:
            if args['checkpoint_period'] <= 0 or args['end_time'] < args['checkpoint_period']:
                raise MultialgorithmError("Checkpointing period ({}) must be positive and less than or equal to end time".format(
                    args['checkpoint_period']))

            if args['end_time'] / args['checkpoint_period'] % 1 != 0:
                raise MultialgorithmError('end_time ({}) must be a multiple of checkpoint_period ({})'.format(
                    args['end_time'], args['checkpoint_period']))

        if 'time_step' in args:
            if args['time_step'] <= 0.0 or args['end_time'] < args['time_step']:
                raise MultialgorithmError("Timestep for time-stepped submodels ({}) must be positive and less than or "
                    "equal to end time".format(args['time_step']))

    def run(self, end_time, results_dir, checkpoint_period, time_step=1):
        """ Run one simulation

        Args:
            end_time (:obj:`float`): the end time of the simulation (sec)
            results_dir (:obj:`str`): path to a directory in which results should be stored
            checkpoint_period (:obj:`float`): the period between simulation state checkpoints (sec)
            time_step (:obj:`float`, optional): time step length of time-stepped submodels (sec)

        Returns:
            :obj:`tuple` of (`int`, `str`): number of simulation events, pathname of directory
                containing the results
        """
        self.sim_config = sim_config.SimulationConfig(time_max=end_time, time_step=time_step)
        self._prepare()

        # create a multi-algorithmic simulator
        simulation_args = dict(
            results_dir=results_dir,
            checkpoint_period=checkpoint_period,
            metadata=self.simulation_metadata,
            end_time=end_time,
            time_step=time_step
        )
        self.process_and_validate_args(simulation_args)
        results_dir = simulation_args['results_dir']

        multialgorithm_simulation = MultialgorithmSimulation(self.model, simulation_args)
        simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
        simulation_engine.initialize()

        # run simulation
        # todo: take metadata out of Checkpoint, and record it in a file
        num_events = simulation_engine.simulate(end_time)
        self.simulation_metadata.run.record_end()
        # todo: handle exceptions
        # todo: update metadata in file

        print('Simulated {} events'.format(num_events))
        if results_dir:
            # use RunResults to summarize results in an HDF5 file in results_dir
            RunResults(results_dir, self.simulation_metadata)
            print("Saved checkpoints and run results in '{}'".format(results_dir))

        return (num_events, results_dir)

    def run_batch(self, results_dir, checkpoint_period):    # pragma: no cover  # not implemented 
        """ Run all simulations specified by the simulation configuration

        Args:
            results_dir (:obj:`str`): path to a directory in which results should be stored
            checkpoint_period (:obj:`float`): the period between simulation state checkpoints (sec)

        Returns:
            :obj:`tuple` of (`int`, `str`): number of simulation events, pathname of directory
                containing the results
        """
        # todo: iterate over sim configs
        for simulation in self.sim_config.iterator():
            # setup simulation changes
            pass
