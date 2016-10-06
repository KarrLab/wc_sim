"""
Configuration for the core simulation system.
Based on guidance at https://docs.python.org/2.7/faq/programming.html#how-do-i-share-global-variables-across-modules

Created 2016/08/01
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""
import os.path as path

# TODO(Arthur): IMPORTANT: put these into use
# TODO(Arthur): IMPORTANT: move these to a config data file

class SimulatorConfig(object):
    COPY_EVENT_BODIES=False         # whether to deepcopy each event_body in SimulationObject.send_event() before 
                                    # scheduling an event that stores the event_body in an Event().
                                    # deepcopy avoids possible data sharing conflicts, but costs time & memory
    DEFAULT_CENTER_OF_MASS=10       # the default center of mass for exponential moving average calculations
                                    # TODO(Arthur): DEFAULT_CENTER_OF_MASS should be moved to config for utilities
    REPRODUCIBLE_SEED=None          # a random seed that makes a simulation reproducible
                                    # set to None for non-reproducible simulations
    
    '''
    the following variables are in use:
    DEFAULT_CENTER_OF_MASS
    REPRODUCIBLE_SEED
    '''
