#!/usr/bin/env python
"""
A draft modular, mult-algorithmic, discrete event WC simulator.

CellState, SSA and FBA are simulation objects. 

SSA and FBA could directly exchange species population data. But the cell's state (CellState) is
included so other sub-models can be added and access the state information. For parallelization, we'll
partition the cell state as described in our PADS 2016 paper.

Both SSA and FBA are self-clocking.
"""

from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from CellState import CellState
import MessageTypes

