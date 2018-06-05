""" Dynamic observables, functionality that depends on them

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

import re
import warnings

from wc_lang import StopCondition, Function, Observable
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError

'''
Also:
    All instances in dynamicModel
    Check for errors including cycles in Check
    Include in checkpoints, and checkpoint processing
    Make available to rate law calculations
    Unittests of Observable and Function needed
    prohibit observable names that conflict with functions (but could be relaxed with smarter parsing)
    push wc_lang
Later:
    cache computed observalbes (use memoize?)
'''

class DynamicObservable(object):
    """ The dynamic representation of an `Observable`

    Attributes:
        dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
        local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
        id (:obj:`str`): unique id
        weighted_species (:obj:`list` of `tuple`): Pairs of :obj:`float`, :obj:`str` representing the
            coefficients and IDs of species whose populations are summed to compute a `DynamicObservable`
        weighted_observables (:obj:`list` of `tuple`): Pairs of :obj:`float`, :obj:`DynamicObservable`
            representing the coefficients and observables whose products are summed in a
            `DynamicObservable`'s value
    """
    def __init__(self, dynamic_model, local_species_population, observable):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            observable (:obj:`Observable`): a `wc_lang` `Observable`
        """
        self.dynamic_model = dynamic_model
        self.local_species_population = local_species_population
        if observable.id == '':
            raise MultialgorithmError("observable cannot have an empty id")
        self.id = observable.id
        self.weighted_species = []
        for species_coeff in observable.species:
            self.weighted_species.append((species_coeff.coefficient, species_coeff.species.id()))
        self.weighted_observables = []
        for observable_coeff in observable.observables:
            id = observable_coeff.observable.id
            if id not in self.dynamic_model.dynamic_observables:
                raise MultialgorithmError("Cannot find DynamicObservable '{}'".format(id))
            dynamic_observable = self.dynamic_model.dynamic_observables[id]
            self.weighted_observables.append((observable_coeff.coefficient, dynamic_observable))
        if self.id in self.dynamic_model.dynamic_observables:
            warnings.warn("Replacing observable '{}' with a new instance".format(self.id))
        self.dynamic_model.dynamic_observables[self.id] = self

    def eval(self, time):
        """ Evaluate the value of this dynamic observable at time `time`

        Args:
            time (:obj:`time`): simulation time

        Returns:
            :obj:`float`: the value of this dynamic observable at time `time`
        """
        value = 0
        for coeff, observable in self.weighted_observables:
            value += coeff * observable.eval(time)
        for coeff, species_id in self.weighted_species:
            value += coeff * self.local_species_population.read_one(time, species_id)
        return value

    def __str__(self):
        """ Provide a readable representation of this `DynamicObservable`

        Returns:
            :obj:`str`: a readable representation of this `DynamicObservable`
        """
        pass

'''
assumes that wc_lang parses a Function into str tokens and Observable observables
'''
class DynamicFunction(object):
    """ The dynamic representation of a `Function`

    Attributes:
        id (:obj:`str`): unique id for this `Function`
        tokens (:obj:`list` of `str`): sequence of Python tokens in this `Function`
        observables (:obj:`dict` of `str` to `DynamicObservable`): map from IDs of observables to
            `DynamicObservable`s used by this `Function`
        local_ns (:obj:`dict` of `str` to `callable`): the functions allowed in `Function`
    """
    def __init__(self, function):
        """
        Args:
            function (:obj:`Function`): a `wc_lang` `Function`
        """
        self.id = function.id
        self.tokens = function.tokens
        self.observables = {observable: observable.id for observable in function.observables}
        self.local_ns = {func.__name__: func for func in Function.Meta.valid_functions}

    def eval(self, time):
        """ Evaluate the value of this dynamic function at time `time`

        Args:
            time (:obj:`time`): simulation time

        Returns:
            :obj:`float`: the value of this dynamic function at time `time`
        """
        tmp_tokens = []
        for token in self.tokens:
            if token in self.observables:
                tmp_tokens.append(self.observables[token].eval(time))
            else:
                tmp_tokens.append(token)
        expression = ' '.join(tmp_tokens)
        return eval(expression, {}, local_ns)


class DynamicStopCondition(DynamicFunction):
    """ The dynamic representation of a `StopCondition`

    """
    pass
