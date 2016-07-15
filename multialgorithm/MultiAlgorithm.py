#!/usr/bin/env python
"""
A draft modular, mult-algorithmic, discrete event WC simulator.

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

from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import (SimulationEngine, MessageTypesRegistry)
from Sequential_WC_Simulator.multialgorithm.CellState import CellState
from Sequential_WC_Simulator.multialgorithm.SimpleStochasticSimulationAlgorithm import SimpleStochasticSimulationAlgorithm
# TODO(Arthur): remove from produciton code:
from Sequential_WC_Simulator.unittests.UniversalSenderReceiverSimulationObject import UniversalSenderReceiverSimulationObject

from Sequential_WC_Simulator.multialgorithm.MessageTypes import (MessageTypes, 
    ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    Continuous_change, ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, 
    GET_POPULATION_body, GIVE_POPULATION_body,
    EXECUTE_SSA_REACTION_body )

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger

'''
Steps:
0. read initial configuration
1. create and configure simulation objects
2. create initial events
3. run simulation
'''
# TODO(Arthur): MOVE to unittests & EXPAND into tests
ssa1 = SimpleStochasticSimulationAlgorithm( 'name1' )
ssa2 = SimpleStochasticSimulationAlgorithm( 'name2', random_seed=123, debug=True, write_plot_output=True )

usr1 = UniversalSenderReceiverSimulationObject( 'usr1' )
usr1.send_event( 1, ssa2, MessageTypes.GIVE_POPULATION, 
    event_body=GIVE_POPULATION_body( { 'x':1, 'y':2,  } ) )
usr1.send_event( 2, ssa2, MessageTypes.EXECUTE_SSA_REACTION, 
    event_body=EXECUTE_SSA_REACTION_body( 'a reaction, internals TBD'  ) )
SimulationEngine.simulate( 5.0 )