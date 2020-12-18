""" Simulate a multialgorithm model

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-25
:Copyright: 2018, Karr Lab
:License: MIT
"""

from collections import namedtuple
import datetime
import getpass
import numpy
import os
import pstats

from de_sim.errors import SimulatorError
from de_sim.simulation_config import SimulationConfig
from de_sim.simulator import Simulator
from de_sim.simulation_metadata import SimulationMetadata, AuthorMetadata
from wc_lang import Model

from wc_lang.io import Reader
from wc_sim.metadata import WCSimulationMetadata
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.run_results import RunResults
from wc_sim.sim_config import WCSimulationConfig
from wc_utils.util.git import get_repo_metadata, RepoMetadataCollectionType
from wc_utils.util.rand import RandomStateManager
import wc_sim.config

config_multialgorithm = wc_sim.config.core.get_config()['wc_sim']['multialgorithm']
'''
TODO: put in docstring
usage:

from wc_sim.simulation import Simulation
from wc_sim.run_results import RunResults

    one run:
        model = ... model object or model file
        simulation = Simulation(model)
        events, results_dir = simulation.run(max_time, results_dir, checkpoint_period)
        run_results = RunResults(results_dir)

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
        dynamic_model (:obj:`DynamicModel`): the simulation's :obj:`DynamicModel`
        de_sim_config (:obj:`SimulationConfig`): a DE-Sim simulation configuration
        author_metadata (:obj:`AuthorMetadata`): metadata about the author of a whole-cell simulation run
        wc_sim_config (:obj:`WCSimulationConfig`): a WC-Sim simulation configuration
        simulator (:obj:`Simulator`): the simulation engine
    """
    def __init__(self, model):
        """
        Args:
            model (:obj:`str` or `Model`): either a path to file(s) describing a `wc_lang` model, or
                a `Model` instance

        Raises:
            :obj:`MultialgorithmError`: if `model` is invalid
        """
        if isinstance(model, Model):
            self.model_path = None
            self.model = model
        elif isinstance(model, str):
            # read model
            self.model_path = os.path.abspath(os.path.expanduser(model))
            self.model = Reader().run(self.model_path)[Model][0]
        else:
            raise MultialgorithmError("model must be a `wc_lang Model` or a pathname for a model, "
                                      "but its type is {}".format(type(model)))

    SimulationReturnValue = namedtuple('SimulationReturnValue', 'num_events results_dir profile_stats',
                                       defaults=(None, None))
    SimulationReturnValue.__doc__ += ': Value returned by a simulation run'
    SimulationReturnValue.num_events.__doc__ = 'Number of events executed by the run'
    SimulationReturnValue.results_dir.__doc__ = 'Directory containing results, if provided'
    SimulationReturnValue.profile_stats.__doc__ = 'A :obj:`pstats.Stats` instance of a profile, if requested'

    def run(self, max_time, results_dir=None, progress_bar=True, checkpoint_period=None,
            seed=None, ode_time_step=None, dfba_time_step=None, profile=False, submodels_to_skip=None,
            verbose=True, object_memory_change_interval=0, options=None):
        """ Run one simulation

        Args:
            max_time (:obj:`float`): the maximum time of a simulation; a stop condition may end it
                earlier (sec)
            results_dir (:obj:`str`, optional): path to a directory in which results are stored
            progress_bar (:obj:`bool`, optional): whether to show the progress of a simulation in
                in a real-time bar on a terminal
            checkpoint_period (:obj:`float`, optional): the period between simulation state checkpoints (sec)
            ode_time_step (:obj:`float`, optional): time step length of ODE submodel (sec)
            dfba_time_step (:obj:`float`, optional): time step length of dFBA submodel (sec)
            profile (:obj:`bool`, optional): whether to output a profile of the simulation's performance
                created by a Python profiler
            seed (:obj:`object`, optional): a seed for the simulation's `numpy.random.RandomState`;
                if provided, `seed` will reseed the simulator's PRNG
            submodels_to_skip (:obj:`list` of :obj:`str`, optional): submodels that should not be run,
                identified by their ids
            verbose (:obj:`bool`, optional): whether to print success output
            object_memory_change_interval (:obj:`int`, optional): event interval between memory profiles
                of the simulation; default of 0 indicates no profile
            options (:obj:`dict`, optional): options for submodels, passed to `MultialgorithmSimulation`

        Returns:
            :obj:`SimulationReturnValue`: containing 1) an :obj:`int` holding the number of simulation
                events, 2) if `results_dir` is provided, a :obj:`str` containing the pathname of the
                directory containing the results, and 3) if `profile is True`, profile stats for the
                simulation

        Raises:
            :obj:`MultialgorithmError`: if the simulation raises an exception
        """
        # create simulation configurations
        # create and validate DE sim configuration
        self.de_sim_config = SimulationConfig(max_time, output_dir=results_dir, progress=progress_bar,
                                              profile=profile,
                                              object_memory_change_interval=object_memory_change_interval)
        self.de_sim_config.validate()

        # create and validate WC configuration
        self.wc_sim_config = WCSimulationConfig(de_simulation_config=self.de_sim_config,
                                                random_seed=seed,
                                                ode_time_step=ode_time_step,
                                                dfba_time_step=dfba_time_step,
                                                checkpoint_period=checkpoint_period,
                                                submodels_to_skip=submodels_to_skip,
                                                verbose=verbose)
        self.wc_sim_config.validate()

        # create author metadata for DE sim
        try:
            username = getpass.getuser()
        except KeyError:     # pragma: no cover
            username = 'Unknown username'
        self.author_metadata = AuthorMetadata(name='Unknown name', email='Unknown email', username=username,
                                              organization='Unknown organization')

        # create WC sim metadata
        wc_simulation_metadata = WCSimulationMetadata(self.wc_sim_config)
        if self.model_path is not None:
            wc_simulation_metadata.set_wc_model_repo(self.model_path)
        # add WC sim metadata to the output
        if self.de_sim_config.output_dir is not None:
            WCSimulationMetadata.write_dataclass(wc_simulation_metadata, self.de_sim_config.output_dir)

        if seed is not None:
            RandomStateManager.initialize(seed=seed)

        # create a multi-algorithmic simulator
        multialgorithm_simulation = MultialgorithmSimulation(self.model, self.wc_sim_config, options)
        self.simulator, self.dynamic_model = multialgorithm_simulation.build_simulation()
        self.simulator.initialize()
        print('*** Model initialized ***')
        print(self.dynamic_model.species_population)

        # set stop_condition after the dynamic model is created
        self.de_sim_config.stop_condition = self.dynamic_model.get_stop_condition()

        # run simulation
        try:
            # provide DE config and author metadata to DE sim
            simulate_rv = self.simulator.simulate(sim_config=self.de_sim_config,
                                                          author_metadata=self.author_metadata)

        except SimulatorError as e:     # pragma: no cover
            raise MultialgorithmError(f'Simulation terminated with simulator error:\n{e}')
        except BaseException as e:      # pragma: no cover
            print('*** BaseException ***')
            print(self.dynamic_model.species_population)
            raise MultialgorithmError(f'Simulation terminated with error:\n{e}')

        if verbose:
            print(f'Simulated {simulate_rv.num_events} events')
            print('Caching statistics:')
            print(self.dynamic_model.cache_manager.cache_stats_table())
        if results_dir:
            # summarize results in an HDF5 file in results_dir
            RunResults(results_dir)
            if verbose:
                print(f"Saved checkpoints and run results in '{results_dir}'")
        return self.SimulationReturnValue(simulate_rv.num_events, results_dir, simulate_rv.profile_stats)

    def get_simulator(self):
        """ Provide the simulation's simulation engine

        Returns:
            :obj:`Simulator`: the simulation's simulation engine
        """
        if hasattr(self, 'simulator'):
            return self.simulator
        return None

    def provide_event_counts(self):
        """ Provide the last simulation's categorized event counts

        Returns:
            :obj:`str`: the last simulation's categorized event counts, in a tab-separated table
        """
        if self.get_simulator():
            return self.get_simulator().provide_event_counts()
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
