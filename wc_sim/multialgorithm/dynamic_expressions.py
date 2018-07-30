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
import math
from collections import namedtuple

from wc_utils.util.enumerate import CaseInsensitiveEnum
import wc_utils.cache
import wc_lang
from wc_lang import Parameter, StopCondition, Function, Observable, ObjectiveFunction, RateLawEquation
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.expression_utils import TokCodes

'''
# TODO:
build:
    unit test remaining code
    integrate into dynamic simulation
    add Observable
cleanup
    move dynamic_components to a more convenient place
    jupyter examples
    memoize performance comparison; decide whether to trash or finish implementing direct dependency tracking eval
    clean up memoize cache file?

Expression eval design:
    Algorithms:
        evaling expression model types:
            evaluate this generically: Observable
            special cases:
                ObjectiveFunction: used by FBA, so express as needed by the FBA solver
                RateLawEquation: needs special considaration of reactant order, intensive vs. extensive, volume, etc.
        evaling other model types used by expressions:
            Species: abundance or concentration in Observable?
            Reaction and BiomassReaction: flux units in ObjectiveFunction?
    Optimizations:
        evaluate parameters statically at initialization
        use memoization to avoid re-evaluation, if the benefit outweighs the overhead; like this:
            cache_dir = tempfile.mkdtemp()
            cache = wc_utils.cache.Cache(directory=os.path.join(cache_dir, 'cache'))
            @cache.memoize()
            def eval(time):
        fast access to specie counts and concentrations:
            eliminate lookups, extra objects and memory allocation/deallocation
        for maximum speed, don't use eval() -- convert expressions into trees, & use an evaluator that
            can process operators, literals, and Python functions
'''


class SimTokCodes(int, CaseInsensitiveEnum):
    """ Token codes used in WcSimTokens """
    dynamic_expression = 1
    other = 2


# a token in DynamicExpression.wc_tokens
WcSimToken = namedtuple('WcSimToken', 'tok_code, token_string, dynamic_expression')
# make dynamic_expression optional: see https://stackoverflow.com/a/18348004
WcSimToken.__new__.__defaults__ = (None, )
WcSimToken.__doc__ += ': Token in a validated expression'
WcSimToken.tok_code.__doc__ = 'SimTokCodes encoding'
WcSimToken.token_string.__doc__ = "The token's string"
WcSimToken.dynamic_expression.__doc__ = "When tok_code is dynamic_expression, the dynamic_expression instance"


WC_LANG_MODEL_TO_DYNAMIC_MODEL = {
    Function: 'DynamicFunction',
    Parameter: 'DynamicParameter',
    Observable: 'DynamicObservable',
    StopCondition: 'DynamicStopCondition',
    ObjectiveFunction: 'DynamicObjectiveFunction',
    RateLawEquation: 'DynamicRateLawEquation'
}


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
            wc_lang_model (:obj:`obj_model.Model`): a corresponding `wc_lang` `Model`, from which this
                `DynamicComponent` is derived
        """
        self.dynamic_model = dynamic_model
        self.local_species_population = local_species_population
        self.id = wc_lang_model.get_id()
        model_type = DynamicExpression.get_dynamic_model_type(wc_lang_model, from_wc_lang_mdl_type=True)
        if model_type not in DynamicExpression.dynamic_components:
            DynamicExpression.dynamic_components[model_type] = {}
        DynamicExpression.dynamic_components[model_type][self.id] = self


class DynamicExpression(DynamicComponent):
    """ Simulation representation of a mathematical expression, based on WcLangExpression

    Attributes:
        expression (:obj:`str`): the expression defined in the `wc_lang` Model
        wc_sim_tokens (:obj:`list` of `WcSimToken`): a tokenized, compressed representation of `expression`
        expr_substrings (:obj:`list` of `str`): strings which are joined to form the string which is 'eval'ed
        local_ns (:obj:`dict`): pre-computed local namespace of functions used in `expression`
    """

    def __init__(self, dynamic_model, local_species_population, wc_lang_model, wc_lang_expression):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            wc_lang_model (:obj:`obj_model.Model`): the corresponding `wc_lang` `Model`
            wc_lang_expression (:obj:`WcLangExpression`): an analyzed and validated expression

        Raises:
            :obj:`MultialgorithmError`: if `wc_lang_expression` does not contain an analyzed,
                validated expression
        """

        super().__init__(dynamic_model, local_species_population, wc_lang_model)

        # wc_lang_expression must have been successfully `tokenize`d.
        if not wc_lang_expression.wc_tokens:
            raise MultialgorithmError("wc_tokens cannot be empty")
        # optimization: self.wc_lang_expression will be deleted by prepare()
        self.wc_lang_expression = wc_lang_expression
        self.expression = wc_lang_expression.expression

    def prepare(self):
        """ Prepare this dynamic expression for simulation

        Because they refer to each other, all `DynamicExpression`s must be prepared after all
        `DynamicExpression`s are created.

        Raises:
            :obj:`MultialgorithmError`: if a Python function used in `wc_lang_expression` does not exist
                validated expression
        """

        # create self.wc_sim_tokens, which contains WcSimTokens that refer to other DynamicExpressions
        self.wc_sim_tokens = []
        # optimization: combine together adjacent wc_token.tok_code math_fun_id and other
        next_static_tokens = ''
        function_names = set()
        for wc_token in self.wc_lang_expression.wc_tokens:
            if wc_token.tok_code == TokCodes.math_fun_id:
                function_names.add(wc_token.token_string)
            if wc_token.tok_code == TokCodes.math_fun_id or wc_token.tok_code == TokCodes.other:
                next_static_tokens = next_static_tokens + wc_token.token_string
            elif wc_token.tok_code == TokCodes.wc_lang_obj_id:
                if next_static_tokens != '':
                    self.wc_sim_tokens.append(WcSimToken(SimTokCodes.other, next_static_tokens))
                    next_static_tokens = ''
                dynamic_expression = self.get_dynamic_component(wc_token.model, wc_token.model_id,
                    from_wc_lang_mdl_type=True)
                self.wc_sim_tokens.append(WcSimToken(SimTokCodes.dynamic_expression, wc_token.token_string,
                    dynamic_expression))
            else:   # pragma    no cover
                assert False, "unknown tok_code {} in {}".format(wc_token.tok_code, wc_token)
        if next_static_tokens != '':
            self.wc_sim_tokens.append(WcSimToken(SimTokCodes.other, next_static_tokens))
        # optimization: to conserve memory, delete self.wc_lang_expression
        del self.wc_lang_expression

        # optimization: pre-allocate and pre-populate substrings for the expression to eval
        self.expr_substrings = []
        for sim_token in self.wc_sim_tokens:
            if sim_token.tok_code == SimTokCodes.other:
                self.expr_substrings.append(sim_token.token_string)
            else:
                self.expr_substrings.append('')

        # optimization: pre-allocate Python functions in namespace
        self.local_ns = {}
        for func_name in function_names:
            if func_name in globals()['__builtins__']:
                self.local_ns[func_name] = globals()['__builtins__'][func_name]
            elif hasattr(globals()['math'], func_name):
                self.local_ns[func_name] = getattr(globals()['math'], func_name)
            else:   # pragma no cover, because only known functions are allowed in model expressions
                raise MultialgorithmError("loading expression '{}' cannot find function '{}'".format(
                    self.expression, func_name))

    def eval(self, time):
        """ Evaluate this mathematical expression

        Approach:
            * Replace references to related Models in `self.wc_sim_tokens` with their values
            * Join the elements of `self.wc_sim_tokens` into a Python expression
            * `eval` the Python expression

        Args:
            time (:obj:`float`): the current simulation time

        Raises:
            :obj:`MultialgorithmError`: if Python `eval` raises an exception
        """
        for idx, sim_token in enumerate(self.wc_sim_tokens):
            if sim_token.tok_code == SimTokCodes.dynamic_expression:
                self.expr_substrings[idx] = str(sim_token.dynamic_expression.eval(time))
        try:
            return eval(''.join(self.expr_substrings), {}, self.local_ns)
        except BaseException as e:
            raise MultialgorithmError("eval of '{}' raises {}: {}'".format(
                self.expression, type(e).__name__, str(e)))

    dynamic_components = {}

    @staticmethod
    def get_dynamic_model_type(model_type, from_wc_lang_mdl_type=False):
        """ Get a simulation's dynamic component type

        Convert to a dynamic component type from a corresponding `wc_lang` Model type and/or a
        string representation

        Args:
            model_type (:obj:`obj`): a string name for a subclass of `DynamicComponent`, or a
                corresponding `obj_model.Model`
            from_wc_lang_mdl_type (:obj:`bool`, optional): if set, `model_type` comes from `wc_lang`,
                that is, it's a subclass of `obj_model.Model`

        Returns:
            :obj:`type`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the dynamic component type cannot be found
        """
        if from_wc_lang_mdl_type:
            model_type = WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type.__class__]
        if isinstance(model_type, str):
            if model_type in globals():
                model_type = globals()[model_type]
            else:
                raise MultialgorithmError("model type '{}' not defined".format(model_type))
        return model_type

    @staticmethod
    def get_dynamic_component(model_type, id, from_wc_lang_mdl_type=False):
        """ Get a simulation's dynamic component

        Args:
            model_type (:obj:`type`): the subclass of `DynamicComponent` (or `obj_model.Model`) being retrieved
            id (:obj:`str`): the dynamic component's id
            from_wc_lang_mdl_type (:obj:`bool`, optional): if set, `model_type` comes from `wc_lang`,
                that is, it's a subclass of `obj_model.Model`

        Returns:
            :obj:`DynamicComponent`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the dynamic component cannot be found
        """
        model_type = DynamicExpression.get_dynamic_model_type(model_type, from_wc_lang_mdl_type)
        if model_type not in DynamicExpression.dynamic_components:
            raise MultialgorithmError("model type '{}' not in DynamicExpression.dynamic_components".format(
                model_type.__name__))
        if id not in DynamicExpression.dynamic_components[model_type]:
            raise MultialgorithmError("model type '{}' with id='{}' not in DynamicExpression.dynamic_components".format(
                model_type.__name__, id))
        return DynamicExpression.dynamic_components[model_type][id]

    def __str__(self):
        """ Provide a readable representation of this `DynamicExpression`

        Returns:
            :obj:`str`: a readable representation of this `DynamicExpression`
        """
        rv = ['DynamicExpression:']
        rv.append("type: {}".format(self.__class__.__name__))
        rv.append("id: {}".format(self.id))
        rv.append("LocalSpeciesPopulation: {}".format(self.local_species_population.name))
        rv.append("expression: {}".format(self.expression))
        return '\n'.join(rv)


class DynamicFunction(DynamicExpression):
    """ The dynamic representation of a `Function`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicStopCondition(DynamicExpression):
    """ The dynamic representation of a `StopCondition`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicObservable(DynamicExpression):
    """ The dynamic representation of an `Observable`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicParameter(DynamicComponent):
    """ The dynamic representation of a `Parameter`
    """

    def __init__(self, dynamic_model, local_species_population, wc_lang_model, value):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            wc_lang_model (:obj:`obj_model.Model`): the corresponding `wc_lang` `Model`
            value (:obj:`float`): the parameter's value
        """
        super().__init__(dynamic_model, local_species_population, wc_lang_model)
        self.value = value

    def eval(self, time):
        """ Provide the value of this parameter

        Args:
            time (:obj:`float`): the current simulation time; not needed, but included so that all
                dynamic expression models have the same signature for 'eval`
        """
        return self.value

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
'''
