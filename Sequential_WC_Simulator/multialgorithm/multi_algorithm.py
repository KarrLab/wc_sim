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
# TODO(Arthur): convert 'printing =' to logging statements


import sys
import logging
import argparse
import warnings
import errno

from Sequential_WC_Simulator.core.config.config import SimulatorConfig
from Sequential_WC_Simulator.core.logging_config import setup_logger_old
from Sequential_WC_Simulator.core.utilities import ReproducibleRandom
from Sequential_WC_Simulator.core.simulation_object import EventQueue, SimulationObject
from Sequential_WC_Simulator.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from Sequential_WC_Simulator.multialgorithm.config import WC_SimulatorConfig
from Sequential_WC_Simulator.multialgorithm.model_representation import Model
from Sequential_WC_Simulator.multialgorithm.model_loader import getModelFromExcel
from Sequential_WC_Simulator.multialgorithm.cell_state import CellState
from Sequential_WC_Simulator.multialgorithm.submodels.simple_SSA_submodel import simple_SSA_submodel
from Sequential_WC_Simulator.multialgorithm.submodels.FBA_submodel import FbaSubmodel
from Sequential_WC_Simulator.multialgorithm.message_types import *

import Sequential_WC_Simulator.multialgorithm.temp.exercise as Exercise

class MultiAlgorithm( object ):
    """A modular, mult-algorithmic, discrete event WC simulator.
    """
    
    @staticmethod
    def parseArgs():
        parser = argparse.ArgumentParser( description="Run a modular, mult-algorithmic, discrete event WC simulation. " )
        parser.add_argument( 'model_filename', type=str, help="Excel file containing the model" )

        parser.add_argument( '--end_time', '-e', type=float, help="End time for the simulation (s)" )

        parser.add_argument( '--output_directory', '-o', type=str, 
            help="Output directory; default '{}'".format(WC_SimulatorConfig.DEFAULT_OUTPUT_DIRECTORY), 
            default=WC_SimulatorConfig.DEFAULT_OUTPUT_DIRECTORY )

        parser.add_argument( '--FBA_time_step', '-F', type=float, 
            help="FBA time step in sec; default: '{:3.1f}'".format(WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP), 
            default=WC_SimulatorConfig.DEFAULT_FBA_TIME_STEP )

        output_options = parser.add_mutually_exclusive_group()
        output_options.add_argument( '--debug', '-d', action='store_true', help='Print debug output' )
        parser.add_argument( '--seed', '-s', type=int, help='Random number seed; if not provided '
            'then reproducibility determined by SimulatorConfig.REPRODUCIBLE_SEED' )
        args = parser.parse_args()
        return args
    
    @staticmethod
    def main( args ):

        SimulationEngine.reset()
        # setup PRNG
        if args.seed:
            ReproducibleRandom.init( seed=args.seed )
        else:
            ReproducibleRandom.init( seed=SimulatorConfig.REPRODUCIBLE_SEED )
        
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
            printing =  "Reading model from '{}'".format( args.model_filename, debug=args.debug ) 
            the_model = getModelFromExcel( args.model_filename, debug_option=args.debug )
            printing =  the_model.summary()
            print(printing)
            
            '''Prepare submodels for computation'''
            the_model.setupSimulation()
        
        # 1. create and configure simulation submodels, including initial events
        algs_to_run = 'FBA SSA'.split()
        shared_cell_states=[ the_model.the_SharedMemoryCellState ]
        for submodel_spec in the_model.submodels:
            if submodel_spec.algorithm == 'SSA':
                if submodel_spec.algorithm in algs_to_run:
                    printing = "create SSA submodel: {} {}".format( submodel_spec.name, 
                        submodel_spec.algorithm )
                    submodel_spec.the_submodel = simple_SSA_submodel( the_model, submodel_spec.name,
                        submodel_spec.id, None, shared_cell_states, submodel_spec.reactions, 
                        submodel_spec.species, debug=args.debug )
            elif submodel_spec.algorithm == 'FBA':
                if submodel_spec.algorithm in algs_to_run:
                    printing = "create FBA submodel: {} {}".format( submodel_spec.name, 
                        submodel_spec.algorithm )
                    submodel_spec.the_submodel = FbaSubmodel( the_model, submodel_spec.name, 
                        submodel_spec.id, None, shared_cell_states, submodel_spec.reactions,
                        submodel_spec.species, args.FBA_time_step, debug=args.debug )
            else:
                raise Exception("Undefined algorithm '{}' for submodel '{}'".format(
                    submodel_spec.algorithm, submodel_spec.name ) )

        # 2. run simulation
        if args.end_time:
            printing =  "Simulating to: {}".format( args.end_time ) 
            SimulationEngine.simulate( args.end_time )
        else:
            printing =  "Simulating to cellCycleLength: {}".format( 
                the_model.getComponentById('cellCycleLength').value )
            SimulationEngine.simulate( the_model.getComponentById('cellCycleLength').value )
        
        return the_model
    
if __name__ == '__main__':
    try:
        args = MultiAlgorithm.parseArgs()
        theModel = MultiAlgorithm.main( args )
        timeHist, speciesCountsHist = the_model.the_SharedMemoryCellState.report_history(
            numpy_format=True )
        volume = None
        growth = None
        Exercise.analyzeResults(the_model, timeHist, volume, growth, speciesCountsHist)
    except KeyboardInterrupt:
        pass
    # do not report IOError: [Errno 32] Broken pipe
    except IOError as e:
        if isinstance(e.args, tuple):
            if e[0] != errno.EPIPE:
               # determine and handle different error
               raise Exception( e )
