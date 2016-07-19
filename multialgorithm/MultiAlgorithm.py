#!/usr/bin/env python
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

import sys
import logging
import argparse
from numpy import random
import warnings

# Use refactored WcModelingTutorial modules
from Sequential_WC_Simulator.multialgorithm.model_representation import Model
from Sequential_WC_Simulator.multialgorithm.model_loader import getModelFromExcel

'''
import inspect
print inspect.getfile( load_workbook )
import sys
for p in sys.path:
    if 'Library' in p:
        print p
'''

from Sequential_WC_Simulator.core.SimulationObject import EventQueue, SimulationObject
from Sequential_WC_Simulator.core.SimulationEngine import SimulationEngine, MessageTypesRegistry
from Sequential_WC_Simulator.multialgorithm.CellState import CellState
from Sequential_WC_Simulator.multialgorithm.SimpleStochasticSimulationAlgorithm import SimpleStochasticSimulationAlgorithm
from Sequential_WC_Simulator.multialgorithm.MessageTypes import (MessageTypes, 
    ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    Continuous_change, ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, 
    GET_POPULATION_body, GIVE_POPULATION_body,
    EXECUTE_SSA_REACTION_body )
from Sequential_WC_Simulator.core.LoggingConfig import setup_logger

class MultiAlgorithm( object ):
    """A modular, mult-algorithmic, discrete event WC simulator.
    """
    
    @staticmethod
    def parseArgs():
        parser = argparse.ArgumentParser( description="Run a modular, mult-algorithmic, discrete event WC simulation. " )
        parser.add_argument( 'model_filename', type=str, help="Excel file containing the model" )
        # TODO(Arthur): attempt to simulate to 'cell division'
        parser.add_argument( 'end_time', type=float, help="End time for the simulation (s)" )

        DEFAULT_OUTPUT_DIRECTORY = '.'
        parser.add_argument( '--output_directory', '-o', type=str, 
            help="Output directory; default '{}'".format(DEFAULT_OUTPUT_DIRECTORY), 
            default=DEFAULT_OUTPUT_DIRECTORY )

        output_options = parser.add_mutually_exclusive_group()
        output_options.add_argument( '--debug', '-d', action='store_true', help='Print debug output' )
        output_options.add_argument( '--plot', '-p', action='store_true', 
            help='Write plot input for plotSpaceTimeDiagram.py to stdout.' )
        parser.add_argument( '--seed', '-s', type=int, help='Random number seed' )
        args = parser.parse_args()
        if args.seed:
            # TODO(Arthur): pass seed to submodels
            random.seed( args.seed )
        return args
    
    @staticmethod
    def main():
        args = MultiAlgorithm.parseArgs()
        '''
        Steps:
        0. read model description
        1. create and configure simulation objects
        2. create initial events
        3. run simulation
        4. plot results
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 0. read model description
            print "Reading model from '{}'".format( args.model_filename )
            the_model = getModelFromExcel( args.model_filename )
            print the_model.summary()
        
        # 1. create and configure simulation submodels
        # SSA
        # FBA
        
        # 2. create initial events

        # 3. run simulation
        SimulationEngine.simulate( args.end_time )
    
if __name__ == '__main__':
    try:
        MultiAlgorithm.main()
    except KeyboardInterrupt:
        pass
