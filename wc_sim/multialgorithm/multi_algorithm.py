"""
A draft modular, mult-algorithmic, discrete event WC simulator.

Model descriptions are read from Excel spreadsheets and instantiated using the model module from
WcModelingTutorial.

CellState, SSA and FBA are simulation objects. 

SSA and FBA could directly exchange species population data. But the cell's state (CellState) is
included so other sub-models can be added and access the state information. For parallelization, we'll
partition the cell state as described in our PADS 2016 paper.

Both SSA and FBA are self-clocking.

Created 2016/07/14
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

# TODO(Arthur): provide 'returns' documentation for all return operations
# TODO(Arthur): rename Seq_DES_WC_simulator

import sys
import argparse
import warnings
import errno

from wc_lang.model_loader import getModelFromExcel
from wc_lang.model_representation import Model
from wc_sim.core.config import paths as config_paths_core
from wc_sim.core.simulation_object import EventQueue
from wc_sim.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
from wc_sim.multialgorithm.cell_state import CellState
from wc_sim.multialgorithm.submodels.simple_SSA_submodel import simple_SSA_submodel
from wc_sim.multialgorithm.submodels.FBA_submodel import FbaSubmodel
from wc_sim.multialgorithm.message_types import *
import wc_sim.multialgorithm.temp.exercise as Exercise
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_utils.config.core import ConfigManager
from wc_utils.util.rand import RandomStateManager

config_core = ConfigManager(config_paths_core.core).get_config()['wc_sim']['core']
config_multialgorithm = ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']

class log_at_time_zero(object):
    debug_log = debug_logs.get_log( 'wc.debug.console' )

    @staticmethod
    def debug(msg):
        log_at_time_zero.debug_log.debug( msg, sim_time=0, local_call_depth=1 )

    @staticmethod
    def info(msg):
        log_at_time_zero.debug_log.info( msg, sim_time=0, local_call_depth=1 )


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
                            help="Output directory; default '{}'".format(config_multialgorithm['default_output_directory']),
                            default=config_multialgorithm['default_output_directory'])

        parser.add_argument('--FBA_time_step', '-F', type=float,
                            help="FBA time step in sec; default: '{:3.1f}'".format(config_multialgorithm['default_fba_time_step']),
                            default=config_multialgorithm['default_fba_time_step'])

        output_options = parser.add_mutually_exclusive_group()
        parser.add_argument('--seed', '-s', type=int, help='Random number seed; if not provided '
                            'then reproducibility determined by config_core[''REPRODUCIBLE_SEED'']')
        args = parser.parse_args()
        return args

    @staticmethod
    def main(args):

        SimulationEngine.reset()

        # setup PRNG
        if args.seed:
            seed = args.seed
        else:
            seed = config_core['reproducible_seed']
        RandomStateManager.instance().seed(seed)

        '''
        Steps:
        0. read model description
        1. create and configure simulation objects, including their initial events
        2. run simulation
        3. plot results
        '''

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 0. read model description
            log_at_time_zero.info("Reading model from '{}'".format(args.model_filename))
            the_model = getModelFromExcel(args.model_filename)
            log_at_time_zero.debug(the_model.summary())

            '''Prepare submodels for computation'''
            the_model.setupSimulation()

        # 1. create and configure simulation submodels, including initial events
        algs_to_run = 'FBA SSA'.split()
        shared_cell_states = [the_model.the_SharedMemoryCellState]
        for submodel_spec in the_model.submodels:
            if submodel_spec.algorithm == 'SSA':
                if submodel_spec.algorithm in algs_to_run:
                    log_at_time_zero.info("create SSA submodel: {} {}".format(submodel_spec.name,
                                                                              submodel_spec.algorithm))
                    submodel_spec.the_submodel = simple_SSA_submodel(the_model, submodel_spec.name,
                                                                     submodel_spec.id, None, 
                                                                     shared_cell_states, 
                                                                     submodel_spec.reactions,
                                                                     submodel_spec.species)
            elif submodel_spec.algorithm == 'FBA':
                if submodel_spec.algorithm in algs_to_run:
                    log_at_time_zero.info("create FBA submodel: {} {}".format(submodel_spec.name,
                                                                              submodel_spec.algorithm))
                    submodel_spec.the_submodel = FbaSubmodel(the_model, submodel_spec.name,
                                                             submodel_spec.id, None, 
                                                             shared_cell_states, 
                                                             submodel_spec.reactions,
                                                             submodel_spec.species, 
                                                             args.FBA_time_step)
            else:
                raise Exception("Undefined algorithm '{}' for submodel '{}'".format(
                    submodel_spec.algorithm, submodel_spec.name))

        # 2. run simulation
        if args.end_time:
            log_at_time_zero.info("Simulating to: {}".format(args.end_time))
            SimulationEngine.simulate(args.end_time)
        else:
            log_at_time_zero.info("Simulating to cellCycleLength: {}".format(
                the_model.getComponentById('cellCycleLength').value))
            SimulationEngine.simulate(the_model.getComponentById('cellCycleLength').value)

        return the_model

if __name__ == '__main__':
    try:
        args = MultiAlgorithm.parse_args()
        the_model = MultiAlgorithm.main(args)
        time_hist, species_counts_hist = the_model.the_SharedMemoryCellState.report_history(
            numpy_format=True)
        volume = None
        growth = None
        Exercise.analyzeResults(the_model, time_hist, volume, growth, species_counts_hist)
    except KeyboardInterrupt:
        pass
    # do not report IOError: [Errno 32] Broken pipe
    except IOError as e:
        if isinstance(e.args, tuple):
            if e[0] != errno.EPIPE:
                # determine and handle different error
                raise Exception(e)
