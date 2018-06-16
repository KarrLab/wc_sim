""" Dynamic observables, and functionality that depends on them

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

import re
import os
import warnings
import tempfile

import wc_utils.cache
from wc_lang import StopCondition, Function, Observable
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.expression_utils import TokCodes

'''
Also:
    failing test for wc_lang that shows the str.replace() problem, and try to fix
    expression parser for DynamicFunction, generalized for ObjectiveFunction, RateLaws, StopCondition, etc.
        prohibit observable names that conflict with functions (but could be relaxed with smarter parsing)
    Make observables available to rate law calculations
Later:
    memoize performance comparison; decide whether to trash or finish implementing direct dependency tracking eval
    clean up memoize cache file?
'''

class ParseWcLangExpr(object):
    '''
        Parse expr into sequence of (value, `TokCodes`) tokens
        Report errors on failed parses
    '''
    def __init__(self, expr):
        """
        Args:
            expr (:obj:`str`): a Python expression used by `wc_lang`
        """
        self.expr = expr

    def tokenize(self, expr_type):
        """
        Args:
            expr_type (:obj:`DynamicModel`): the simulation's dynamic model

        Raises:
            :obj:`MultialgorithmError`: if `observable` has an empty id, or doesn't have a corresponding
                dynamic observable registered with `dynamic_model`
        """
        pass


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
        antecedent_observables (:obj:`set`): dynamic observables on which this dynamic observable depends
    """
    cache_dir = tempfile.mkdtemp()
    cache = wc_utils.cache.Cache(directory=os.path.join(cache_dir, 'cache'))

    def __init__(self, dynamic_model, local_species_population, observable):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            observable (:obj:`Observable`): a `wc_lang` `Observable`

        Raises:
            :obj:`MultialgorithmError`: if `observable` has an empty id, or doesn't have a corresponding
                dynamic observable registered with `dynamic_model`
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
        self.antecedent_observables = set()
        for observable_coeff in observable.observables:
            id = observable_coeff.observable.id
            if id not in self.dynamic_model.dynamic_observables:
                raise MultialgorithmError("DynamicObservable '{}' not registered".format(id))
            dynamic_observable = self.dynamic_model.dynamic_observables[id]
            self.weighted_observables.append((observable_coeff.coefficient, dynamic_observable))
            self.antecedent_observables.add(dynamic_observable)
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
        rv = []
        rv.append("id: {}".format(self.id))
        rv.append("LocalSpeciesPopulation: {}".format(self.local_species_population.name))
        rv.append("weighted_species: {}".format(self.weighted_species))
        rv.append("weighted_observables: {}".format([(coeff,dyn_obs.id)
            for coeff,dyn_obs in self.weighted_observables]))
        return '\n'.join(rv)


# assumes that wc_lang parses a Function into str tokens and Observable observables
class DynamicFunction(object):
    """ The dynamic representation of a `Function`

    Attributes:
        dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
        id (:obj:`str`): unique id for this `DynamicFunction`
        tokens (:obj:`list` of `tuple`): sequence of (value, `TokCodes`) tokens for this `Function`
        dynamic_observables (:obj:`dict` of `str` to `DynamicObservable`): map from IDs of dynamic
            observables to `DynamicObservable`s used by this `DynamicFunction`
        local_ns (:obj:`dict` of `str` to `callable`): the functions allowed in `Function`
    """
    def __init__(self, dynamic_model, function):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            function (:obj:`Function`): a `wc_lang` `Function`

        Raises:
            :obj:`MultialgorithmError`: if an observable in `function` doesn't have a corresponding
                dynamic observable registered with `dynamic_model`
        """
        self.dynamic_model = dynamic_model
        self.id = function.id
        self.tokens = function.tokens
        self.dynamic_observables = {}
        for observable in function.observables:
            if observable.id not in self.dynamic_model.dynamic_observables:
                raise MultialgorithmError("DynamicObservable '{}' not registered for '{}'".format(
                    observable.id, self.id))
            dynamic_observable = self.dynamic_model.dynamic_observables[observable.id]
            self.dynamic_observables[observable.id] = dynamic_observable
        self.local_ns = {func.__name__: func for func in Function.Meta.valid_functions}

    @DynamicObservable.cache.memoize()
    def eval(self, time):
        """ Evaluate the value of this dynamic function at time `time`

        Args:
            time (:obj:`time`): simulation time

        Returns:
            :obj:`obj`: the value of this dynamic function at time `time`

        Raises:
            :obj:`MultialgorithmError`: if a dynamic observable in this dynamic function isn't registered
        """
        # replace observable IDs with their current values
        tmp_tokens = []
        for value,code in self.tokens:
            if code == TokCodes.wc_lang_obj_id:
                if value not in self.dynamic_observables:   # pragma    no cover
                    raise MultialgorithmError("DynamicObservable '{}' not registered".format(id))
                tmp_tokens.append(str(self.dynamic_observables[value].eval(time)))
            else:
                tmp_tokens.append(value)
        expression = ' '.join(tmp_tokens)
        # todo: catch exceptions
        return eval(expression, {}, self.local_ns)

    '''
    # TODO: complete or toss
    def eval_dynamic_observables(self, time, observables_to_eval=None):
        """ Evaluate some dynamic observables at time `time`

        Because observables depend on each other, all observables that depend on `observables` must be
        evaluated. Observables that do not depend on other observables must be evaluated first.

        Args:
            time (:obj:`float`): the simulation time
            observables_to_eval (:obj:`list` of `str`, optional): if provided, ids of the observables to evaluate;
                otherwise, evaluate all observables

        Returns:
            :obj:`dict`: map from a super set of the IDs of dynamic observables in `observables_to_eval`
                to their values at simulation time `time`
        """
        if observables_to_eval is None:
            observables_to_eval = set(self.dynamic_observables.keys())
        else:
            # close observables_to_eval by searching the observable dependencies relation
            observables_to_eval = set([self.dynamic_observables[id] for id in observables_to_eval])
            observables_to_explore = set(observables_to_eval)
            while observables_to_explore:
                dyn_observable = observables_to_explore.pop()
                for _,dependent_observable in dyn_observable.weighted_observables:
                    if dependent_observable not in observables_to_eval:
                        observables_to_eval.add(dependent_observable)
                        observables_to_explore.add(dependent_observable)

        # map from observable to its current value
        evaluated_observables = {}
        # map from observable to set of non-evaluated observables on which it depends
        antecedent_observables = {}
        observables_ready_to_eval = set()
        for dyn_observable in observables_to_eval:
            if dyn_observable.antecedent_observables:
                antecedent_observables[dyn_observable] = set(dyn_observable.antecedent_observables)
            else:
                observables_ready_to_eval.add(dyn_observable)
        # for dyn_observable,antecedents in antecedent_observables.items():
        if not observables_ready_to_eval:
            raise MultialgorithmError("observables_ready_to_eval is empty")
    '''


class DynamicStopCondition(DynamicFunction):
    """ The dynamic representation of a `StopCondition`

    """
    def eval(self, time):
        """ Evaluate the value of this dynamic stop condition at time `time`

        Args:
            time (:obj:`time`): simulation time

        Returns:
            :obj:`bool`: the boolean value of this dynamic stop condition at time `time`

        Raises:
            :obj:`MultialgorithmError`: if eval doesn't return a boolean value
        """
        value = super().eval(time)
        if not isinstance(value, bool):
            raise MultialgorithmError("DynamicStopCondition evaluated to a {}, instead of a bool".format(
                type(value)))
        return value
