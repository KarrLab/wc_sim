""" Define multi-algoritmic simulation errors.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-12-12
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""


class Error(Exception):
    """ Base class for exceptions involving multi-algoritmic simulation

    Attributes:
        message (:obj:`str`): the exception's message
    """

    def __init__(self, message=None):
        super().__init__(message)


class MultialgorithmError(Error):
    """ Exception raised for errors in package `wc_sim`

    Attributes:
        message (:obj:`str`): the exception's message
    """

    def __init__(self, message=None):
        super().__init__(message)


class DynamicMultialgorithmError(Error):
    """ Exception raised for errors in package `wc_sim` that occur during a simulation

    Attributes:
        time (:obj:`float`): the simulation time at which the error occurs
        message (:obj:`str`): the exception's message
    """

    def __init__(self, time, message=None):
        self.time = time
        super().__init__(f"{time}: {message}")


class SpeciesPopulationError(Error):
    """ Exception raised when species population management encounters a problem

    Attributes:
        message (:obj:`str`): the exception's message
    """

    def __init__(self, message=None):
        super().__init__(message)


class DynamicSpeciesPopulationError(DynamicMultialgorithmError):
    """ Exception raised when species population management encounters a problem during a simulation

    Attributes:
        time (:obj:`float`): the simulation time at which the error occurs
        message (:obj:`str`): the exception's message
    """

    def __init__(self, time, message=None):
        super().__init__(time, message)


class DynamicFrozenSimulationError(DynamicMultialgorithmError):
    """ Exception raised by an SSA submodel when it is the only submodel and total propensities == 0

    A simulation in this state cannot progress.

    Attributes:
        time (:obj:`float`): the simulation time at which the error occurs
        message (:obj:`str`): the exception's message
    """

    def __init__(self, time, message=None):
        super().__init__(time, message)


class DynamicNegativePopulationError(DynamicMultialgorithmError):
    """ Exception raised when a negative species population is predicted

    The sum of `last_population` and `population_decrease` equals the predicted negative population.

    Attributes:
        time (:obj:`float`): the simulation time at which the error occurs
        method (:obj:`str`): name of the method in which the exception occured
        species (:obj:`str`): name of the species whose population is predicted to be negative
        last_population (:obj:`float`): previous recorded population for the species
        population_decrease (:obj:`float`): change to the population which would make it negative
        delta_time (:obj:`float`, optional): if the species has been updated by a continuous submodel,
            time since the last continuous update
    """

    def __init__(self, time, method, species, last_population, population_decrease, delta_time=None):
        self.method = method
        self.species = species
        self.last_population = last_population
        self.population_decrease = population_decrease
        self.delta_time = delta_time
        super().__init__(time, 'negative population predicted')

    def __eq__(self, other):
        """ Determine whether two instances have the same content """
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.method, self.species, self.last_population, self.population_decrease,
                     self.delta_time))

    def __str__(self):
        """ Provide a readable `DynamicNegativePopulationError` which contains all its attributes

        Returns:
            :obj:`str`: a readable representation of this `DynamicNegativePopulationError`
        """
        rv = "at {:g}: {}(): negative population predicted for '{}', with decline from {:g} to {:g}".format(
            self.time, self.method, self.species, self.last_population,
            self.last_population + self.population_decrease)
        if self.delta_time is None:
            return rv
        else:
            if self.delta_time == 1:
                return rv + " over 1 time unit"
            else:
                return rv + " over {:g} time units".format(self.delta_time)


class MultialgorithmWarning(UserWarning):
    """ `wc_sim` multialgorithm warning """
    pass
