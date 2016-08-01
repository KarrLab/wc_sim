"""
Configuration for the core simulation system.
Based on guidance at https://docs.python.org/2.7/faq/programming.html#how-do-i-share-global-variables-across-modules

Created 2016/08/01
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""
import logging
import os.path as path

# TODO(Arthur): IMPORTANT: put these into use

class SimulatorConfig(object):
    LOGGING_LEVEL=logging.DEBUG     # see levels at https://docs.python.org/2/library/logging.html#levels
    # the default root directory for logging 
    DEFAULT_LOGGING_ROOT_DIR=path.expanduser( "~/tmp/Sequential_WC_Simulator_logging")
    PLOT_OUTPUT=False               # whether to log a line for each executed event, suitable for plotting
    COPY_EVENT_BODIES=False         # whether to deepcopy each event_body in SimulationObject.send_event() before 
                                    # scheduling an event that stores the event_body in an Event(). deepcopy avoids possible
                                    # data sharing conflicts, but costs time & memory
    DEFAULT_CENTER_OF_MASS=10       # the default center of mass for exponential moving average calculations
    DEFAULT_SEED=19                 # the default random seed for ReproducibleRandom
    REPRODUCIBLE=False              # whether a simulation should be reproducible
    
    # the following variables are in use:
