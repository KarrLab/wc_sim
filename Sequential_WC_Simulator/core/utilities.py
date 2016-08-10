from __future__ import print_function

"""
Simulation utilities. 

Created 2016/07/11
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

from random import Random
from numpy import random as numpy_random
import random as python_random
import sys
import math
import logging
import numpy as np

from Sequential_WC_Simulator.core.config import SimulatorConfig

class ExponentialMovingAverage(object):
    """An exponential moving average.
    
    Each moving average S is computed recursively from the sample values Y:
        S_1 = Y_1
        S_t = alpha * Y_t + (1-alpha)*S_(t-1)
    
    Attributes:
        alpha: float; the decay factor
    """
    def __init__( self, initial_value, alpha=None, center_of_mass=None ):
        """Initialize an ExponentialMovingAverage.
        
        Args:
            alpha: float; the decay factor
            center_of_mass: number; a center of mass for initializing alpha, the decay factor
                in an exponential moving average. alpha = 1/center_of_mass
        
        Raises:
            ValueError if alpha <= 0 or 1 <= alpha
        """
        if alpha != None:
            self.alpha = alpha
        elif center_of_mass != None:
            self.alpha = 1./(1.+center_of_mass)
        else:
            raise ValueError( "alpha or center_of_mass must be provided" )
        if self.alpha <= 0 or 1 <= self.alpha:
            raise ValueError( "alpha should satisfy 0<alpha<1: but alpha={}".format( self.alpha ) )
        self.exponential_moving_average = initial_value
        
    def add_value( self, value ):
        """Add a sample to this ExponentialMovingAverage, and get the next average.
        
        Args:
            value: number; the next value to contribute to the exponential moving average.
        
        Returns:
            The next exponential moving average.
        """
        self.exponential_moving_average = (self.alpha * value) \
            + (1-self.alpha)*self.exponential_moving_average
        return self.exponential_moving_average
    
    def get_value( self ):
        """Get the curent next average.
        
        Returns:
            The curent exponential moving average.
        """
        return self.exponential_moving_average
    
class ReproducibleRandom(object):
    """A source of reproducible random numbers.
    
    A static, singleton class that can provide reproducible random numbers, and 
    independent, reproducible random number streams. These can be seeded by a 
    built-in seed or a single random seed provided at the command line.
    If ReproducibleRandom is initialized without either of these seeds, then it
    will random numbers and streams seeded by  
    
    Attributes:
        built_in_seed: a built-in RNG seed, to provide reproducibility when no 
            command line seed is provided
        _private_PRNG: a private PRNG, for generating random values
        RNG_generator: numpy RandomState(); a PRNG for generating additional, independent
            random number streams
    """
    
    _built_in_seed=17
    _private_PRNG=None
    _RNG_generator=None
    
    @staticmethod
    def init( reproducible=False, seed=None ):
        """Initialize ReproducibleRandom.
        
        Args:
            reproducible: boolean; if set, use the hard-coded, built-in PRNG seed
            seed: a hashable object; if reproducible is not set, a seed that seeds
                all random numbers and random streams provided by ReproducibleRandom. 
        """
        if reproducible:
            ReproducibleRandom._private_PRNG = numpy_random.RandomState( 
                ReproducibleRandom._built_in_seed )
        elif seed != None:
            ReproducibleRandom._private_PRNG = numpy_random.RandomState( seed )
        else:
            ReproducibleRandom._private_PRNG = numpy_random.RandomState( )
        ReproducibleRandom._RNG_generator = numpy_random.RandomState( 
            ReproducibleRandom._private_PRNG.randint( np.iinfo(np.uint32).max ) )
        

    @staticmethod
    def _check_that_init_was_called():
        """Checks whether ReproducibleRandom.init() was called.
        
        Raises:
            ValueError: if init() was not called
        """
        if ReproducibleRandom._private_PRNG==None:
            raise ValueError( "Error: ReproducibleRandom: ReproducibleRandom.init() must"
            "be called before calling other ReproducibleRandom methods." )
    
    @staticmethod
    def get_numpy_random_state( ):
        """Provide a new numpy RandomState() instance.
        
        The RandomState() instance can be used by threads or to initialize
        concurrent processes which cannot share a random number stream because
        they execute asynchronously.
        
        Returns:
            A new numpy RandomState() instance. If ReproducibleRandom.init() was called to 
            make reproducible random numbers, then the RandomState() instance will do so.
            
        Raises:
            ValueError: if init() was not called
        """
        ReproducibleRandom._check_that_init_was_called()
        return numpy_random.RandomState( ReproducibleRandom._RNG_generator.randint( np.iinfo(np.uint32).max ) )
        
    @staticmethod
    def get_numpy_random( ):
        """Provide a numpy RandomState() instance.
        
        The output of this RandomState() instance is not reproducible across threads or
        other non-deterministically asynchronous components, but it can shared by components 
        of a deterministic sequentia program.
        
        Returns:
            A numpy RandomState() instance. If ReproducibleRandom.init() was called to 
            make reproducible random numbers, then the RandomState() instance will do so.

        Raises:
            ValueError: if init() was not called
        """
        ReproducibleRandom._check_that_init_was_called()
        return ReproducibleRandom._private_PRNG
        
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

N_AVOGADRO = 6.022e23 #Avogadro constant

def nanminimum(x, y):
    return np.where(np.logical_or(np.isnan(y), np.logical_and(x <= y, np.logical_not(np.isnan(x)))), x, y)
    
def nanmaximum(x, y):
    return np.where(np.logical_or(np.isnan(y), np.logical_and(x >= y, np.logical_not(np.isnan(x)))), x, y)

def compare_name_with_class( a_name, a_class ):
    """Compares class name with the type of a_class.
    
    Used by SimulationObject instances in handle_event() to compare the event message type 
    field against event message types.
    
    Returns:
        True if the the name of class a_class is a_name.
    """
    return a_name == a_class.__name__
