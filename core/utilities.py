from __future__ import print_function

from random import Random
import math
import logging
logger = logging.getLogger(__name__)

# control logging level with: logger.setLevel()
# this enables debug output: logging.basicConfig( level=logging.DEBUG )

"""
Simulation utilities. 

Created 2016/07/11
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

class StochasticRound( object ):
    """Stochastically round floating point values.
    
    A float is rounded to one of the two nearest integers. The mean of the rounded values for a set of floats
    converges to the mean of the floats. This is achieved by making P[rounding x down] = 1 - (x - floor(x) ), and
    P[rounding x up] = 1 - P[rounding x down].
    This avoids the bias that would arise from always using floor() or ceiling(), especially with 
    small populations.
    
    Attributes:
        RNG: A Random instance, initialized on creation of a StochasticRound.
    """

    def __init__( self, seed=None ):
        """Initialize a StochasticRound.
        
        Args:
            seed: a hashable object; optional; to deterministically initialize the basic random number generator 
            provide seed. Otherwise some system-dependent randomness source will be used to initialize 
            the generator. See Python documentation for random.seed().
        """
        if seed:
            self.RNG = Random( seed )
        else:
            self.RNG = Random( )
        
    def Round( self, x ):
        """Stochastically round a floating point value.
        
        Args:
            x: a float to be stochastically rounded.
            
        Returns:
            A stochastic round of x.
        """
        floor_x = math.floor( x )
        fraction = x - floor_x
        if 0==fraction:
            return x
        else:
            if self.RNG.random( ) < fraction:
                return floor_x + 1
            else:
                return floor_x

''' 
Utility functions

@author Jonathan Karr, karr@mssm.edu
@date 3/22/2016
'''

import numpy as np

N_AVOGADRO = 6.022e23 #Avogadro constant

def nanminimum(x, y):
    return np.where(np.logical_or(np.isnan(y), np.logical_and(x <= y, np.logical_not(np.isnan(x)))), x, y)
    
def nanmaximum(x, y):
    return np.where(np.logical_or(np.isnan(y), np.logical_and(x >= y, np.logical_not(np.isnan(x)))), x, y)

