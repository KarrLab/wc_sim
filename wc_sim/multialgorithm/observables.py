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
from collections import namedtuple

from wc_utils.util.enumerate import CaseInsensitiveEnum
import wc_utils.cache
from wc_lang import StopCondition, Function, Observable
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.expression_utils import TokCodes

'''
# TODO:
build
    subclass all DynamicX from DynamicComponent
    memoize all evals
    rename to dynamic_expressions
    jupyter examples
cleanup
    memoize performance comparison; decide whether to trash or finish implementing direct dependency tracking eval
    clean up memoize cache file?
'''


class DynamicComponent(object):
    """ Component of a simulation

    Attributes:
        dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
        local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
        id (:obj:`str`): unique id
    """
    def __init__(self, dynamic_model, local_species_population, wc_lang_model):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            wc_lang_model (:obj:`obj_model.Model`): the corresponding `wc_lang` `Model`

        Raises:
            :obj:`MultialgorithmError`: if `wc_lang_model` does not have an id
        """
        self.dynamic_model = dynamic_model
        self.local_species_population = local_species_population
        if not hasattr(wc_lang_model, 'id') or wc_lang_model.id == '':
            raise MultialgorithmError("wc_lang_model must have an id")
        self.id = wc_lang_model.id


class SimTokCodes(int, CaseInsensitiveEnum):
    """ Token codes used in WcSimTokens """
    dynamic_component = 1
    math_fun = 2
    other = 3


# a token in DynamicExpression.wc_tokens
WcSimToken = namedtuple('WcSimToken', 'tok_code, token_string, dynamic_component')
# make dynamic_component optional: see https://stackoverflow.com/a/18348004
WcSimToken.__new__.__defaults__ = (None)
WcSimToken.__doc__ += ': Token in a validated expression'
WcSimToken.tok_code.__doc__ = 'SimTokCodes encoding'
WcSimToken.token_string.__doc__ = "The token's string"
WcSimToken.dynamic_component.__doc__ = "When tok_code is dynamic_component, the dynamic_component instance"


class DynamicExpression(object):
    """ Simulation representation of a mathematical expression, based on WcLangExpression

    Attributes:
        source (:obj:`str`): the `wc_lang` Model source for this expression
        wc_tokens (:obj:`list` of `WcSimToken`): a tokenized representation of the expression
    """
    def __init__(self, wc_lang_expression):
        """
        Args:
            wc_lang_expression (:obj:`WcLangExpression`): an analyzed and validated expression

        Raises:
            :obj:`MultialgorithmError`: if `wc_lang_expression` does not contain an analyzed,
                validated expression;
        """
        if not wc_lang_expression.wc_tokens:
            raise MultialgorithmError("wc_tokens cannot be empty")
        self.wc_tokens = []
        for wc_token in wc_lang_expression.wc_tokens:
            if wc_token.tok_code == TokCodes.wc_lang_obj_id:
                dynamic_component = get_dynamic_component(wc_token.model_type, wc_token.model_id)
                self.wc_tokens.append(WcSimToken(SimTokCodes.dynamic_component, wc_token.token_string,
                    dynamic_component))
            elif wc_token.tok_code == TokCodes.math_fun_id:
                self.wc_tokens.append(WcSimToken(SimTokCodes.math_fun, wc_token.token_string))
            elif wc_token.tok_code == TokCodes.other:
                self.wc_tokens.append(WcSimToken(SimTokCodes.other, wc_token.token_string))


class DynamicExpressionComponent(DynamicComponent):
    """ Component of a simulation that contains a mathematical expression

    Attributes:
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
