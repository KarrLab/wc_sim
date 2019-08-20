""" Simulate a multialgorithm model

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-25
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import datetime
import numpy

from de_sim.sim_metadata import SimulationMetadata, AuthorMetadata, RunMetadata
from de_sim.simulation_engine import SimulationEngine
from wc_lang import Model, Validator
from wc_lang.io import Reader
from wc_lang.transform import PrepForWcSimTransform
from wc_sim.core import sim_config
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.run_results import RunResults
from wc_utils.util.git import get_repo_metadata, RepoMetadataCollectionType
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.string import indent_forest


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
        simulation_engine (:obj:`SimulationEngine`): the `SimulationEngine`
    """
    def __init__(self, model, sim_config=None):
        """
        Args:
            model (:obj:`str` or `Model`): either a path to file(s) describing a `wc_lang` model, or
                a `Model` instance
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
            self.model = Reader().run(self.model_path)[Model][0]
            if self.model is None:
                raise MultialgorithmError("No model found in model file '{}'".format(self.model_path))
        else:
            raise MultialgorithmError("model must be a wc_lang Model or a pathname for a model, but its "
                "type is {}".format(type(model)))

        self.sim_config = sim_config

    def _prepare(self):
        """ Prepare simulation model and metadata
        """
        # prepare & check the model
        PrepForWcSimTransform().run(self.model)
        errors = Validator().run(self.model)
        if errors:
            raise ValueError(indent_forest(['The model is invalid:', [errors]]))

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
        model, _ = get_repo_metadata(repo_type=RepoMetadataCollectionType.SCHEMA_REPO)

        # author metadata
        try:
            username = getpass.getuser()
        except:     # pragma: no cover
            username = 'Unknown username'
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

    def run(self, end_time, results_dir=None, checkpoint_period=None, time_step=1, seed=None):
        """ Run one simulation

        Args:
            end_time (:obj:`float`): the end time of the simulation (sec)
            results_dir (:obj:`str`, optional): path to a directory in which results are stored
            checkpoint_period (:obj:`float`, optional): the period between simulation state checkpoints (sec)
            time_step (:obj:`float`, optional): time step length of time-stepped submodels (sec)
            seed (:obj:`object`, optional): a seed for the simulation's `numpy.random.RandomState`;
                if provided, `seed` will reseed the simulator's PRNG

        Returns:
            :obj:`tuple` of (`int`, `str`): number of simulation events, pathname of directory
                containing the results, or :obj:`tuple` of (`int`, `None`): number of simulation events,
                `None` if `results_dir=None`
        """
        if seed is not None:
            RandomStateManager.initialize(seed=seed)

        self.sim_config = sim_config.SimulationConfig(time_max=end_time, time_step=time_step)
        self._prepare()

        # create a multi-algorithmic simulator
        simulation_args = dict(
            metadata=self.simulation_metadata,
            end_time=end_time,
            time_step=time_step
        )
        if results_dir:
            simulation_args['results_dir'] = results_dir
            simulation_args['checkpoint_period'] = checkpoint_period

        self.process_and_validate_args(simulation_args)
        timestamped_results_dir = None
        if results_dir:
            timestamped_results_dir = simulation_args['results_dir']

        multialgorithm_simulation = MultialgorithmSimulation(self.model, simulation_args)
        self.simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
        self.simulation_engine.initialize()

        if timestamped_results_dir:
            SimulationMetadata.write_metadata(self.simulation_metadata, timestamped_results_dir)

        # run simulation
        try:
            num_events = self.simulation_engine.simulate(end_time)
        except MultialgorithmError as e:
            print('Simulation terminated with multialgorithm error: {}'.format(e))
            return
        except BaseException as e:
            print('Simulation terminated with error: {}'.format(e))
            return

        self.simulation_metadata.run.record_end()
        # update metadata in file
        if timestamped_results_dir:
            SimulationMetadata.write_metadata(self.simulation_metadata, timestamped_results_dir)

        print('Simulated {} events'.format(num_events))
        if timestamped_results_dir:
            # summarize results in an HDF5 file in timestamped_results_dir
            RunResults(timestamped_results_dir)
            print("Saved checkpoints and run results in '{}'".format(timestamped_results_dir))
            return (num_events, timestamped_results_dir)
        else:
            return (num_events, None)

    def get_simulation_engine(self):
        """ Provide the simulation's simulation engine

        Returns:
            :obj:`SimulationEngine`: the simulation's simulation engine
        """
        if hasattr(self, 'simulation_engine'):
            return self.simulation_engine
        return None

    def provide_event_counts(self):
        """ Provide the last simulation's categorized event counts

        Returns:
            :obj:`str`: the last simulation's categorized event counts, in a tab-separated table
        """
        if self.get_simulation_engine():
            return self.get_simulation_engine().provide_event_counts()
        return 'execute run() to obtain event counts'

    def run_batch(self, results_dir, checkpoint_period):    # pragma: no cover  # not implemented
        """ Run all simulations specified by the simulation configuration

        Args:
            results_dir (:obj:`str`): path to a directory in which results should be stored
            checkpoint_period (:obj:`float`): the period between simulation state checkpoints (sec)

        Returns:
            :obj:`tuple` of (`int`, `str`): number of simulation events, pathname of directory
                containing the results
        """
        # todo: implement; iterate over sim configs
        for simulation in self.sim_config.iterator():
            # setup simulation changes
            pass
