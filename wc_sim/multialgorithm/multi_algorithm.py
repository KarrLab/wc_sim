"""A draft modular, mult-algorithmic, discrete event WC simulator.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-07-14
:Copyright: 2016, Karr Lab
:License: MIT

Model descriptions are read from Excel spreadsheets and instantiated using the model module from
WcModelingTutorial.

SSA and FBA are simulation objects.

SSA and FBA could directly exchange species population data. But the cell's state
(LocalSpeciesPopulation) is included so other sub-models can be added and access the state information.
For parallelization, we'll partition the cell state as described in our PADS 2016 paper.

Both SSA and FBA are self-clocking.
"""

# TODO(Arthur): provide 'returns' documentation for all return operations

import sys
import argparse
import warnings
import errno

from wc_analysis import exercise
from wc_lang.io import Reader
from wc_sim.core.config import paths as config_paths_core
from wc_sim.core.simulation_object import EventQueue
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
from wc_sim.multialgorithm.submodels.ssa import SsaSubmodel
from wc_sim.multialgorithm.submodels.fba import FbaSubmodel
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation
from wc_sim.multialgorithm.executable_model import ExecutableModel
from wc_utils.config.core import ConfigManager
from wc_utils.util.rand import RandomStateManager

config_core = ConfigManager(config_paths_core.core).get_config()['wc_sim']['core']
config_multialgorithm = ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']

class LogAtTimeZero(object):
    debug_log = debug_logs.get_log( 'wc.debug.console' )

    @staticmethod
    def debug(msg):
        LogAtTimeZero.debug_log.debug( msg, sim_time=0, local_call_depth=1 )

    @staticmethod
    def info(msg):
        LogAtTimeZero.debug_log.info( msg, sim_time=0, local_call_depth=1 )


class MultiAlgorithm(object):
    """A modular, mult-algorithmic, discrete event WC simulator.
    """

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Run a modular, mult-algorithmic, discrete event WC simulation. ")
        parser.add_argument('model_filename', type=str, help="Excel file containing the model")

        parser.add_argument('--end_time', '-e', type=float, help="End time for the simulation (s)")

        parser.add_argument('--output_directory', '-o', type=str,
            help="Output directory; default '{}'".format(
                config_multialgorithm['default_output_directory']),
                default=config_multialgorithm['default_output_directory'])

        parser.add_argument('--FBA_time_step', '-F', type=float,
            help="FBA time step in sec; default: '{:3.1f}'".format(
                config_multialgorithm['default_fba_time_step']),
                default=config_multialgorithm['default_fba_time_step'])

        output_options = parser.add_mutually_exclusive_group()
        parser.add_argument('--seed', '-s', type=int, help='Random number seed; if not provided '
                            'then reproducibility determined by config_core[''REPRODUCIBLE_SEED'']')
        args = parser.parse_args()
        return args

    @staticmethod
    def initialize_simulation(args):
        '''Initialize a WC simulation.

        Steps:
        0. read model description
        1. create and configure simulation objects, including their initial events

        Args:
            args (`Namespace`): command line arguments obtained by argparse.

        Raises:

        '''

        SimulationEngine.reset()

        # setup PRNG
        if args.seed:
            seed = args.seed
        else:
            seed = config_core['reproducible_seed']
        RandomStateManager.instance().seed(seed)

        # TODO(Arthur): IMPORTANT: create backward reactions for reversible reactions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 0. read model description
            LogAtTimeZero.info("Reading model from '{}'".format(args.model_filename))
            model = Reader().run(args.model_filename)
            LogAtTimeZero.debug(model.summary())

            '''Prepare submodels for computation'''
            ExecutableModel.set_up_simulation(model)

        # 1. create and configure simulation submodels, including initial events
        algs_to_run = 'FBA SSA'.split()     # todo: take this from config
        for submodel_spec in model.submodels:
            if submodel_spec.algorithm == 'SSA':
                if submodel_spec.algorithm in algs_to_run:
                    LogAtTimeZero.info("create SSA submodel: {} {}".format(submodel_spec.name,
                        submodel_spec.algorithm))
                    submodel_spec.the_submodel = SsaSubmodel(model, submodel_spec.name,
                                                                     model.cell_state,
                                                                     submodel_spec.reactions,
                                                                     submodel_spec.species,
                                                                     submodel_spec.parameters)
            elif submodel_spec.algorithm == 'FBA':
                if submodel_spec.algorithm in algs_to_run:
                    LogAtTimeZero.info("create FBA submodel: {} {}".format(submodel_spec.name,
                        submodel_spec.algorithm))
                    submodel_spec.the_submodel = FbaSubmodel(model, submodel_spec.name,
                                                             model.cell_state,
                                                             submodel_spec.reactions,
                                                             submodel_spec.species,
                                                             submodel_spec.parameters,
                                                             args.FBA_time_step)
            else:
                raise Exception("Undefined algorithm '{}' for submodel '{}'".format(
                    submodel_spec.algorithm, submodel_spec.name))

    @staticmethod
    def run_simulation(args):
        '''Run a WC simulation.

        Args:
            args (`Namespace`): command line arguments obtained by argparse.

        Returns:
            `Model`: the simulation model.
        '''
        if args.end_time:
            LogAtTimeZero.info("Simulating to: {}".format(args.end_time))
            SimulationEngine.simulate(args.end_time)
        else:
            LogAtTimeZero.info("Simulating to cellCycleLength: {}".format(
                model.get_component_by_id('cellCycleLength').value))
            SimulationEngine.simulate(model.get_component_by_id('cellCycleLength').value)

        return model

if __name__ == '__main__':
    try:
        args = MultiAlgorithm.parse_args()
        MultiAlgorithm.initialize_simulation(args)
        model = MultiAlgorithm.run_simulation(args)
        time_hist, species_counts_hist = model.cell_state.report_history(numpy_format=True)
        volume = None
        growth = None
        exercise.analyzeResults(model, time_hist, volume, growth, species_counts_hist)
    except KeyboardInterrupt:
        pass
    # do not report IOError: [Errno 32] Broken pipe
    except IOError as e:
        if isinstance(e.args, tuple):
            if e[0] != errno.EPIPE:
                # determine and handle different error
                raise Exception(e)
