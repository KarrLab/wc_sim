"""
Configuration for the multi-algorithm WC simulator.
Based on guidance at https://docs.python.org/2.7/faq/programming.html#how-do-i-share-global-variables-across-modules

Created 2016/08/01
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

# TODO(Arthur): IMPORTANT: put these into use
import logging

class WC_SimulatorConfig(object):
    LOGGING_LEVEL=logging.DEBUG     # see levels at https://docs.python.org/2/library/logging.html#levels
    
    INTERPOLATE=True                # whether to interpolate specie counts when executing Specie.get_population()
                                    # interpolation incorporates a linear estimate of the change in copy number 
                                    # since the last update by a continuous model; see the documentation of 
                                    # CellState for more details
                                    # TODO(Arthur): unitests only work with INTERPOLATE=True; have them work either
    DEFAULT_OUTPUT_DIRECTORY='.'    # the default output directory
    DEFAULT_FBA_TIME_STEP = 3.0     # the default time step for FBA submodels


'''
the following variables are in use:
INTERPOLATE
DEFAULT_OUTPUT_DIRECTORY
DEFAULT_FBA_TIME_STEP
'''