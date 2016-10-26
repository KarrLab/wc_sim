''' 
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
@date 7/19/2016
'''

from Sequential_WC_Simulator.multialgorithm.config import paths as config_paths
from wc_utils.config.core import ConfigManager
from wc_utils.util.RandomUtilities import StochasticRound, ReproducibleRandom

config = ConfigManager(config_paths.core).get_config()['Sequential_WC_Simulator']['multialgorithm']


class Specie(object):
    """ Specie tracks the population of a single specie.
    
    We have these cases. Suppose that the specie's population is adjusted by:

    * DISCRETE model only: estimating the population obtains the last value written
    * CONTINUOUS model only: estimating the population obtains the last value written plus a linear interpolated change 
      since the last continuous adjustment
    * Both model types: reading the populations obtains the last value written plus a linear interpolated change 
      based on the last continuous adjustment

    Without loss of generality, we assume that all species can be adjusted by both model types and that at most
    one continuous model adjusts a specie's population. (Multiple continuous models of a specie - or any 
    model attribute - would be non-sensical, as it implies multiple conflicting, simultaneous rates of change.) 
    We also assume that if a population is adjusted by a continuous model then the adjustments occur sufficiently 
    frequently that the last adjustment alway provides a useful estimate of flux. Adjustments take the following form:
    
    * DISCRETE adjustment: (time, population_change)
    * CONTINUOUS adjustment: (time, population_change, flux_estimate_at_time)
    
    A Specie stores the (time, population) of the most recent adjustment and (time, flux) of the most
    recent continuous adjustment. Naming these R_time, R_pop, C_time, and C_flux, the population p at time t is 
    estimated by:::

        interpolation = 0
        if C_time:
            interpolation = (t - C_time)*C_flux
        p = R_pop + interpolation

    This approach is completely general, and can be applied to any simulation value.
    
    Real versus integer copy numbers: Clearly, specie copy number values are non-negative integers. However,
    continuous models may estimate copy numbers as real number values. E.g., ODEs calculate real valued concentrations.
    But SSA models do not naturally handle non-integral copy numbers, and models that represent the state of 
    individual species -- such as a bound molecule -- are not compatible with non-integral copy numbers. Therefore,
    Specie stores a real valued population, but reports only a non-negative integral population. 
    In particular, the population reported by get_population() is rounded. Rounding is done  stochastically to avoid the 
    bias that would arise from always using floor() or ceiling(), especially with small populations.
    
    Attributes:
        last_population: population after the most recent adjustment
        continuous_flux: flux provided by the most recent adjustment by a continuous model, 
            if there has been an adjustment by a continuous model; otherwise, uninitialized
        continuous_time: time of the most recent adjustment by a continuous model; None if the specie has not
            received a continuous adjustment
            
        Specie objects do not include the specie's name, as we assume they'll be stored in structures
        which contain the names.
    
    # TODO(Arthur): optimization: put Specie functionality into CellState and SharedMemoryCellState, avoiding 
        overhead of a Class instance for each Specie.
    """
    # use __slots__ to save space
    __slots__ = "specie_name last_population continuous_time continuous_flux stochasticRounder".split()

    def __init__( self, specie_name, initial_population, initial_flux=None ):
        """Initialize a Specie object.
        
        Args:
            specie_name: string; the specie's name; not logically needed, but may be helpful for logging, debugging, etc.
            initial_population: non-negative number; initial population of the specie
            initial_flux: number; required for Specie whose population is partially estimated by a 
                continuous mode; initial flux for the specie
        """
        
        # TODO(Arthur): perhaps: add optional arg to not round copy number values reported
        assert 0 <= initial_population, '__init__(): population should be >= 0'
        self.specie_name = specie_name
        self.last_population = initial_population
        if initial_flux == None:
            self.continuous_time = None
        else:
            self.continuous_time = 0
            self.continuous_flux = initial_flux

        self.stochasticRounder = StochasticRound( ReproducibleRandom.get_numpy_random() ).Round

    def discrete_adjustment( self, population_change ):
        """A discrete model adjusts the specie's population.

        Args:
            population_change: number; modeled increase or decrease in the specie's population
            
        Raises:
            ValueError: if population goes negative
        """
        if self.last_population + population_change < 0:
            raise ValueError( "discrete_adjustment(): negative population: "
                "last_population + population_change = {:.2f} + {:.2f}".format( 
                self.last_population, population_change ) )
        self.last_population += population_change
    
    def continuous_adjustment( self, population_change, time, flux ):
        """A continuous model adjusts the specie's population.

        Args:
            population_change: number; modeled increase or decrease in the specie's population
            
        Raises:
            ValueError: if initial flux was not provided for a continuously updated variable
            ValueError: if time is <= the time of the most recent continuous adjustment
            ValueError: if population goes negative
            AssertionError: if the time is negative
        """
        if self.continuous_time == None:
            raise ValueError( "continuous_adjustment(): initial flux was not provided" )
        assert 0 <= time, 'negative time: {:.2f}'.format( time )
        # multiple continuous adjustments at a time that does not advance the specie's state do not make sense
        if time <= self.continuous_time:
            raise ValueError( "continuous_adjustment(): time <= self.continuous_time: {:.2f} < {:.2f}".format( 
                time, self.continuous_time ) )
        if self.last_population + population_change < 0:
            raise ValueError( "continuous_adjustment(): negative population: "
                "last_population + population_change = {:.2f} + {:.2f}".format( 
                self.last_population, population_change ) )
        self.continuous_time = time
        self.continuous_flux = flux
        self.last_population += population_change
    
    def get_population( self, time=None ):
        """Get the specie's population at time.
        
        Interpolate continuous values as described in the documentation of class Specie.
        
        Args:
            time: number; optional; simulation time of the request; 
                time is required if the specie has had a continuous adjustment
            
        Raises:
            ValueError: if time is required and not provided
            ValueError: time is earlier than a previous continuous adjustment
        """
        if self.continuous_time == None:
            return self.stochasticRounder( self.last_population )
        else:
            if time == None:
                raise ValueError( "get_population(): time needed because "
                    "continuous adjustment received at time {:.2f}".format( self.continuous_time ) )
            if time < self.continuous_time:
                raise ValueError( "get_population(): time < self.continuous_time: {:.2f} < {:.2f}\n".format( 
                    time, self.continuous_time ) )
            interpolation=0
            if config['interpolate']:
                interpolation = (time - self.continuous_time) * self.continuous_flux
            float_copy_number = self.last_population + interpolation
            return self.stochasticRounder( float_copy_number )
            
    def __str__( self ):
        return "specie_name:{}; last_population:{}; continuous_time:{}; continuous_flux:{}".format( 
            self.specie_name, self.last_population, self.continuous_time, self.continuous_flux )

        
