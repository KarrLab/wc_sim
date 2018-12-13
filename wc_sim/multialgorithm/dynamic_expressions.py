""" Dynamic expressions

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-06-03
:Copyright: 2018, Karr Lab
:License: MIT
"""

from collections import namedtuple
from wc_lang.expression import WcTokenCodes, ParsedExpression
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_utils.util.enumerate import CaseInsensitiveEnum
import math
import obj_model
import os
import re
import tempfile
import wc_lang
import wc_utils.cache


'''
# TODO:
build:
    integrate into dynamic simulation
cleanup
    move dynamic_components to a more convenient place
    jupyter examples
    memoize performance comparison; decide whether to trash or finish implementing direct dependency tracking eval
    clean up memoize cache file?

Expression eval design:
    Algorithms:
        evaling expression model types:
            special cases:
                :obj:`wc_lang.DfbaObjective`: used by FBA, so express as needed by the FBA solver
                :obj:`wc_lang.RateLaw`: needs special considaration of reactant order, intensive vs. extensive, volume, etc.
        evaling other model types used by expressions:
            Reaction and BiomassReaction: flux units in :obj:`wc_lang.DfbaObjective`?
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


# a token in DynamicExpression._wc_tokens
WcSimToken = namedtuple('WcSimToken', 'code, token_string, dynamic_expression')
# make dynamic_expression optional: see https://stackoverflow.com/a/18348004
WcSimToken.__new__.__defaults__ = (None, )
WcSimToken.__doc__ += ': Token in a validated expression'
WcSimToken.code.__doc__ = 'SimTokCodes encoding'
WcSimToken.token_string.__doc__ = "The token's string"
WcSimToken.dynamic_expression.__doc__ = "When code is dynamic_expression, the dynamic_expression instance"


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
        self.id = wc_lang_model.id
        model_type = DynamicExpression.get_dynamic_model_type(wc_lang_model)
        if model_type not in DynamicExpression.dynamic_components:
            DynamicExpression.dynamic_components[model_type] = {}
        DynamicExpression.dynamic_components[model_type][self.id] = self

    def __str__(self):
        """ Provide a readable representation of this `DynamicComponent`

        Returns:
            :obj:`str`: a readable representation of this `DynamicComponent`
        """
        rv = ['DynamicComponent:']
        rv.append("type: {}".format(self.__class__.__name__))
        rv.append("id: {}".format(self.id))
        return '\n'.join(rv)


class DynamicExpression(DynamicComponent):
    """ Simulation representation of a mathematical expression, based on :obj:`ParsedExpression`

    Attributes:
        expression (:obj:`str`): the expression defined in the `wc_lang` Model
        wc_sim_tokens (:obj:`list` of :obj:`WcSimToken`): a tokenized, compressed representation of `expression`
        expr_substrings (:obj:`list` of :obj:`str`): strings which are joined to form the string which is 'eval'ed
        local_ns (:obj:`dict`): pre-computed local namespace of functions used in `expression`
    """

    def __init__(self, dynamic_model, local_species_population, wc_lang_model, wc_lang_expression):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            wc_lang_model (:obj:`obj_model.Model`): the corresponding `wc_lang` `Model`
            wc_lang_expression (:obj:`ParsedExpression`): an analyzed and validated expression

        Raises:
            :obj:`MultialgorithmError`: if `wc_lang_expression` does not contain an analyzed,
                validated expression
        """

        super().__init__(dynamic_model, local_species_population, wc_lang_model)

        # wc_lang_expression must have been successfully `tokenize`d.
        if not wc_lang_expression._wc_tokens:
            raise MultialgorithmError("_wc_tokens cannot be empty - ensure that '{}' is valid".format(
                wc_lang_model))
        # optimization: self.wc_lang_expression will be deleted by prepare()
        self.wc_lang_expression = wc_lang_expression
        self.expression = wc_lang_expression.expression

    def prepare(self):
        """ Prepare this dynamic expression for simulation

        Because they refer to each other, all `DynamicExpression`s must be created before any of them
        are prepared.

        Raises:
            :obj:`MultialgorithmError`: if a Python function used in `wc_lang_expression` does not exist
        """

        # create self.wc_sim_tokens, which contains WcSimTokens that refer to other DynamicExpressions
        self.wc_sim_tokens = []
        # optimization: combine together adjacent wc_token.tok_codes other than wc_obj_id
        next_static_tokens = ''
        function_names = set()
        non_lang_obj_id_tokens = set([WcTokenCodes.math_func_id, WcTokenCodes.number, WcTokenCodes.op, WcTokenCodes.other])

        i = 0
        while i < len(self.wc_lang_expression._wc_tokens):
            wc_token = self.wc_lang_expression._wc_tokens[i]
            if wc_token.code == WcTokenCodes.math_func_id:
                function_names.add(wc_token.token_string)
            if wc_token.code in non_lang_obj_id_tokens:
                next_static_tokens = next_static_tokens + wc_token.token_string
            elif wc_token.code == WcTokenCodes.wc_obj_id:
                if next_static_tokens != '':
                    self.wc_sim_tokens.append(WcSimToken(SimTokCodes.other, next_static_tokens))
                    next_static_tokens = ''
                try:
                    dynamic_expression = self.get_dynamic_component(wc_token.model, wc_token.model_id)
                except:
                    raise MultialgorithmError("'{}.{} must be prepared to create '{}''".format(
                        wc_token.model.__class__.__name__, wc_token.model_id, self.id))
                self.wc_sim_tokens.append(WcSimToken(SimTokCodes.dynamic_expression, wc_token.token_string,
                                                     dynamic_expression))
            else:   # pragma    no cover
                assert False, "unknown code {} in {}".format(wc_token.code, wc_token)
            if wc_token.code == WcTokenCodes.wc_obj_id and wc_token.model_type == wc_lang.Function:
                # skip past the () syntactic sugar on functions
                i += 2
            # advance to the next token
            i += 1
        if next_static_tokens != '':
            self.wc_sim_tokens.append(WcSimToken(SimTokCodes.other, next_static_tokens))
        # optimization: to conserve memory, delete self.wc_lang_expression
        del self.wc_lang_expression

        # optimization: pre-allocate and pre-populate substrings for the expression to eval
        self.expr_substrings = []
        for sim_token in self.wc_sim_tokens:
            if sim_token.code == SimTokCodes.other:
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
        assert hasattr(self, 'wc_sim_tokens'), "'{}' must use prepare() before eval()".format(
            self.id)
        for idx, sim_token in enumerate(self.wc_sim_tokens):
            if sim_token.code == SimTokCodes.dynamic_expression:
                self.expr_substrings[idx] = str(sim_token.dynamic_expression.eval(time))
        try:
            return eval(''.join(self.expr_substrings), {}, self.local_ns)
        except BaseException as e:
            raise MultialgorithmError("eval of '{}' raises {}: {}'".format(
                self.expression, type(e).__name__, str(e)))

    dynamic_components = {}

    @staticmethod
    def get_dynamic_model_type(model_type):
        """ Get a simulation's dynamic component type

        Convert to a dynamic component type from a corresponding `wc_lang` Model type, instance or
        string name

        Args:
            model_type (:obj:`Object`): a `wc_lang` Model type represented by a subclass of `obj_model.Model`,
                an instance of `obj_model.Model`, or a string name for a `obj_model.Model`

        Returns:
            :obj:`type`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the corresponding dynamic component type cannot be determined
        """
        if isinstance(model_type, type) and issubclass(model_type, obj_model.Model):
            if model_type in WC_LANG_MODEL_TO_DYNAMIC_MODEL:
                return WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type]
            raise MultialgorithmError("model class of type '{}' not found".format(model_type.__name__))

        if isinstance(model_type, obj_model.Model):
            if model_type.__class__ in WC_LANG_MODEL_TO_DYNAMIC_MODEL:
                return WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type.__class__]
            raise MultialgorithmError("model of type '{}' not found".format(model_type.__class__.__name__))

        if isinstance(model_type, str):
            model_type_type = getattr(wc_lang, model_type, None)
            if model_type_type is not None:
                if model_type_type in WC_LANG_MODEL_TO_DYNAMIC_MODEL:
                    return WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type_type]
                raise MultialgorithmError("model of type '{}' not found".format(model_type_type))
            raise MultialgorithmError("model type '{}' not defined".format(model_type))

        raise MultialgorithmError("model type '{}' has wrong type".format(model_type))

    @staticmethod
    def get_dynamic_component(model_type, id):
        """ Get a simulation's dynamic component

        Args:
            model_type (:obj:`type`): the subclass of `DynamicComponent` (or `obj_model.Model`) being retrieved
            id (:obj:`str`): the dynamic component's id

        Returns:
            :obj:`DynamicComponent`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the dynamic component cannot be found
        """
        model_type = DynamicExpression.get_dynamic_model_type(model_type)
        if model_type not in DynamicExpression.dynamic_components:
            raise MultialgorithmError("model type '{}' not in DynamicExpression.dynamic_components".format(
                model_type.__name__))
        if id not in DynamicExpression.dynamic_components[model_type]:
            raise MultialgorithmError("model type '{}' with id='{}' not in DynamicExpression.dynamic_components".format(
                model_type.__name__, id))
        # print('model_type, id', model_type, id, '->', DynamicExpression.dynamic_components[model_type][id])
        return DynamicExpression.dynamic_components[model_type][id]

    def __str__(self):
        """ Provide a readable representation of this `DynamicExpression`

        Returns:
            :obj:`str`: a readable representation of this `DynamicExpression`
        """
        rv = ['DynamicExpression:']
        rv.append("type: {}".format(self.__class__.__name__))
        rv.append("id: {}".format(self.id))
        rv.append("expression: {}".format(self.expression))
        return '\n'.join(rv)


class DynamicFunction(DynamicExpression):
    """ The dynamic representation of a :obj:`wc_lang.Function`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicStopCondition(DynamicExpression):
    """ The dynamic representation of a :obj:`wc_lang.StopCondition`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicObservable(DynamicExpression):
    """ The dynamic representation of an :obj:`wc_lang.Observable`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicParameter(DynamicComponent):
    """ The dynamic representation of a :obj:`wc_lang.Parameter`
    """

    def __init__(self, dynamic_model, local_species_population, wc_lang_model, value):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            wc_lang_model (:obj:`obj_model.Model`): the corresponding :obj:`wc_lang.Parameter`
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


class DynamicSpecies(DynamicComponent):
    """ The dynamic representation of a :obj:`wc_lang.Species`
    """

    def __init__(self, dynamic_model, local_species_population, wc_lang_model):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
            wc_lang_model (:obj:`obj_model.Model`): the corresponding :obj:`wc_lang.Species`
        """
        super().__init__(dynamic_model, local_species_population, wc_lang_model)
        # Grab a reference to the right wc_lang.Species object used by local_species_population
        self.species_obj = local_species_population._population[wc_lang_model.id]

    def eval(self, time):
        """ Provide the population of this species

        Args:
            time (:obj:`float`): the current simulation time
        """
        return self.species_obj.get_population(time)


class DynamicDfbaObjective(DynamicExpression):
    """ The dynamic representation of an :obj:`wc_lang.DfbaObjective`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicRateLaw(DynamicExpression):
    """ The dynamic representation of a :obj:`wc_lang.RateLaw`
    """

    def __init__(self, *args):
        super().__init__(*args)


WC_LANG_MODEL_TO_DYNAMIC_MODEL = {
    wc_lang.Function: DynamicFunction,
    wc_lang.Parameter: DynamicParameter,
    wc_lang.Species: DynamicSpecies,
    wc_lang.Observable: DynamicObservable,
    wc_lang.StopCondition: DynamicStopCondition,
    wc_lang.DfbaObjective: DynamicDfbaObjective,
    wc_lang.RateLaw: DynamicRateLaw,
}
