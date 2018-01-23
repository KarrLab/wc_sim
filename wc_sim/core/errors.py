'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-01-22
:Copyright: 2018, Karr Lab
:License: MIT
'''

class Error(Exception):
    """ Base class for exceptions in wc_sim.core
    """
    pass


class SimulatorError(Error):
    """ Exception raised for errors in wc_sim.core

    Attributes:
        message (:obj:`srt`): the exception's message
    """
    def __init__(self, message):
        self.message = message
