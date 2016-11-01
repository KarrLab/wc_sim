'''Track the population of a single specie.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-07-19
:Copyright: 2016, Karr Lab
:License: MIT
'''

from Sequential_WC_Simulator.multialgorithm.config import paths as config_paths
from wc_utils.config.core import ConfigManager
from wc_utils.util.RandomUtilities import StochasticRound, ReproducibleRandom

config = ConfigManager(config_paths.core).get_config()['Sequential_WC_Simulator']['multialgorithm']


class Specie(object):
    '''Specie tracks the population of a single specie.

    We have these cases. Suppose that the specie's population is adjusted by:

    * DISCRETE model only: estimating the population obtains the last value written
    * CONTINUOUS model only: estimating the population obtains the last value written plus a linear interpolated change
      since the last continuous adjustment
    * Both model types: reading the populations obtains the last value written plus a linear interpolated change
      based on the last continuous adjustment

    Without loss of generality, we assume that all species can be adjusted by both model types and that at most
    one continuous model adjusts a specie's population. (Multiple continuous models of a specie - or any
    model attribute - would be nonsensical, as it implies multiple conflicting, simultaneous rates of change.)
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
        last_population (float): population after the most recent adjustment
        continuous_flux (float): flux provided by the most recent adjustment by a continuous model,
            if there has been an adjustment by a continuous model; otherwise, uninitialized
        continuous_time: time of the most recent adjustment by a continuous model; None if the specie has not
            received a continuous adjustment

        Specie objects do not include the specie's name, as we assume they'll be stored in structures
        which contain the names.

    # TODO(Arthur): optimization: put Specie functionality into CellState and SharedMemoryCellState, avoiding
        overhead of a Class instance for each Specie.
    '''
    # use __slots__ to save space
    __slots__ = "specie_name last_population continuous_time continuous_flux stochasticRounder".split()

    def __init__( self, specie_name, initial_population, initial_flux=None ):
        '''Initialize a Specie object.

        Args:
            specie_name (str): the specie's name; not logically needed, but helpful for error
                reporting, logging, debugging, etc.
            initial_population (int): non-negative number; initial population of the specie
            initial_flux (number, optional): initial flux for the specie; required for Species whose
                population is estimated, at least in part, by a continuous model
        '''
        assert 0 <= initial_population, '__init__(): population should be >= 0'
        self.specie_name = specie_name
        self.last_population = initial_population
        if initial_flux == None:
            self.continuous_time = None
        else:
            self.continuous_time = 0
            self.continuous_flux = initial_flux

        self.stochasticRounder = StochasticRound( ReproducibleRandom.get_numpy_random() ).Round

    def discrete_adjustment( self, population_change, time ):
        '''Make a discrete adjustment of the specie's population.

        A discrete-time sub-model, such as the stochastic simulation algorithm (SSA), must use this
        method to adjust the specie's population.

        Args:
            population_change (number): the modeled increase or decrease in the specie's population
            time (number): the simulation time at which the predicted change occurs; this time is
                used by `get_population` to interpolate continuous-time predictions between integrations.

        Returns:
            number: the specie's adjusted population

        Raises:
            NegativePopulationError: if the predicted population at `time` is negative or
            if decreasing the population by `population_change` would make the population negative
        '''
        current_population = self.get_population( time )
        if current_population + population_change < 0:
            raise NegativePopulationError('discrete_adjustment', self.specie_name,
                self.last_population, population_change)
        self.last_population += population_change
        return self.get_population( time )

    def continuous_adjustment( self, population_change, time, flux ):
        '''A continuous-time sub-model adjusts the specie's state.

        A continuous-time sub-model, such as an ordinary differential equation (ODE) or a dynamic flux
        balance analysis (FBA) model, must use this method to adjust the specie's state. We assume that
        each integration of a continuous-time model predicts a specie's population and the
        population's short-term future rate of change, i.e., its `flux`. Further, since an
        integration of a continuous-time model at the current time must depend on this specie's
        population just before the integration, we assume that population changes predicted by the
        flux provided by the previous `continuous_adjustment` call are incorporated in this call's
        `population_change`.

        Args:
            population_change (number): modeled increase or decrease in the specie's population
            time (number): the simulation time at which the predicted change occurs; this time is
                used by `get_population` to interpolate continuous-time predictions between integrations.
            flux (number): the predicted flux of the specie at the provided time

        Returns:
            number: the specie's adjusted population

        Raises:
            ValueError: if flux was not provided when this `Specie` was instantiated
            ValueError: if `time` is not greater than the time of the most recent
                `continuous_adjustment` call on this `Specie`
            NegativePopulationError: if applying `population_change` makes the population go negative
        '''
        if self.continuous_time == None:
            raise ValueError( "continuous_adjustment(): initial flux was not provided" )
        # the simulation time must advance between adjacent continuous adjustments
        if time <= self.continuous_time:
            raise ValueError( "continuous_adjustment(): time <= self.continuous_time: {:.2f} < {:.2f}".format(
                time, self.continuous_time ) )
        if self.last_population + population_change < 0:
            raise NegativePopulationError('continuous_adjustment', self.specie_name,
                self.last_population, population_change, time-self.continuous_time)
        self.continuous_time = time
        self.continuous_flux = flux
        self.last_population += population_change
        return self.last_population

    def get_population( self, time=None ):
        '''Obtain the specie's population.

        Obtain the specie's current population. If one of the sub-model(s) predicting the specie's
        population is a continuous-time model, then use the specie's last flux to interpolate
        the current population as described in the documentation of this class.

        Args:
            time (number, optional): the current simulation time; `time` is required if one of the
                sub-models modeling the specie is a continuous-time sub-model.

        Raises:
            ValueError: if `time` is required but not provided
            ValueError: if `time` is earlier than the time of a previous continuous adjustment
            NegativePopulationError: if interpolation predicts a negative population
        '''
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
            if self.last_population + interpolation < 0:
                raise NegativePopulationError('get_population', self.specie_name,
                    self.last_population, interpolation, time - self.continuous_time)
            float_copy_number = self.last_population + interpolation
            return self.stochasticRounder( float_copy_number )

    def __str__( self ):
        return "specie_name:{}; last_population:{}; continuous_time:{}; continuous_flux:{}".format(
            self.specie_name, self.last_population, self.continuous_time, self.continuous_flux )

class Error(Exception):
    '''Base class for exceptions in Specie.'''
    pass

class NegativePopulationError(Error):
    '''Exception raised when a negative specie population is predicted.

    `last_population` plus `population_decrease` equals the predicted negative population.

    Attributes:
        method (:obj:`str`): name of the method in which the exception occured
        specie (:obj:`str`): name of the specie whose population has become negative
        last_population (:obj:`float`): last population of the specie
        population_decrease (:obj:`float`): change to the population which makes it negative
        delta_time (:obj:`float`, optional): if the specie has been updated by a continuous submodel,
            time since the last continuous update
    '''
    def __init__(self, method, specie, last_population, population_decrease, delta_time=None):
        self.method=method
        self.specie=specie
        self.last_population=last_population
        self.population_decrease=population_decrease
        self.delta_time=delta_time

    def __eq__(self, other):
        '''Override the default Equals behavior'''
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        '''Define a non-equality test'''
        return not self.__eq__(other)

    def __hash__(self):
        return hash( (self.method, self.specie, self.last_population, self.population_decrease,
            self.delta_time))

    def __str__(self):
        '''string representation'''
        rv = "{}(): negative population for '{}', with decline from {:g} to {:g}".format( self.method,
                self.specie, self.last_population, self.last_population+self.population_decrease )
        if self.delta_time is None:
            return rv
        else:
            return rv + " over {:g} time units".format(self.delta_time)
