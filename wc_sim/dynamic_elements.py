""" Dynamic elements of a multialgorithm simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-02-07
:Copyright: 2017-2019, Karr Lab
:License: MIT
"""

import math
import numpy
import warnings
from collections import namedtuple

from obj_model import utils
from obj_model.expression import ObjModelTokenCodes
from wc_lang import Species, Compartment
from wc_onto import onto
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.species_populations import LocalSpeciesPopulation
from wc_utils.util.enumerate import CaseInsensitiveEnum
from wc_utils.util.ontology import are_terms_equivalent
import obj_model
import wc_lang


'''
# old TODOs:
cleanup
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
            estimate how much this would improve performance
'''


# mapping from wc_lang Models to DynamicComponents
WC_LANG_MODEL_TO_DYNAMIC_MODEL = {}


class SimTokCodes(int, CaseInsensitiveEnum):
    """ Token codes used in WcSimTokens """
    dynamic_expression = 1
    other = 2


# a token in DynamicExpression._obj_model_tokens
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
        dynamic_model (:obj:`DynamicModel`): the simulation's root dynamic model
        local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
        id (:obj:`str`): unique id
    """

    # map of all dynamic components, indexed by type and then identifier
    dynamic_components_objs = {}

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
        model_type = DynamicComponent.get_dynamic_model_type(wc_lang_model)
        if model_type not in DynamicComponent.dynamic_components_objs:
            DynamicComponent.dynamic_components_objs[model_type] = {}
        DynamicComponent.dynamic_components_objs[model_type][self.id] = self

    @staticmethod
    def get_dynamic_model_type(model_type):
        """ Get a simulation's dynamic component type

        Obtain a dynamic component type from a corresponding `wc_lang` Model type, instance or
        string name.

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
                raise MultialgorithmError("model of type '{}' not found".format(model_type_type.__name__))
            raise MultialgorithmError("model type '{}' not defined".format(model_type))

        raise MultialgorithmError("model type '{}' has wrong type".format(model_type))

    # todo: either use or discard this method
    @staticmethod
    def get_dynamic_component(model_type, id): # pragma: no cover # not used
        """ Get a simulation's dynamic component

        Args:
            model_type (:obj:`type`): the subclass of `DynamicComponent` (or `obj_model.Model`) being retrieved
            id (:obj:`str`): the dynamic component's id

        Returns:
            :obj:`DynamicComponent`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the dynamic component cannot be found
        """
        model_type = DynamicComponent.get_dynamic_model_type(model_type)
        if model_type not in DynamicComponent.dynamic_components_objs:
            raise MultialgorithmError("model type '{}' not in DynamicComponent.dynamic_components_objs".format(
                model_type.__name__))
        if id not in DynamicComponent.dynamic_components_objs[model_type]:
            raise MultialgorithmError("model type '{}' with id='{}' not in DynamicComponent.dynamic_components_objs".format(
                model_type.__name__, id))
        return DynamicComponent.dynamic_components_objs[model_type][id]

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

    NON_LANG_OBJ_ID_TOKENS = set([ObjModelTokenCodes.math_func_id,
                                  ObjModelTokenCodes.number,
                                  ObjModelTokenCodes.op,
                                  ObjModelTokenCodes.other])

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
        if not wc_lang_expression._obj_model_tokens:
            raise MultialgorithmError("_obj_model_tokens cannot be empty - ensure that '{}' is valid".format(
                wc_lang_model))
        # optimization: self.wc_lang_expression will be deleted by prepare()
        self.wc_lang_expression = wc_lang_expression
        self.expression = wc_lang_expression.expression

    def prepare(self):
        """ Prepare this dynamic expression for simulation

        Because they refer to each other, all :obj:`DynamicExpression`\ s must be created before any
        of them are prepared.

        Raises:
            :obj:`MultialgorithmError`: if a Python function used in `wc_lang_expression` does not exist
        """

        # create self.wc_sim_tokens, which contains WcSimTokens that refer to other DynamicExpressions
        self.wc_sim_tokens = []
        # optimization: combine together adjacent obj_model_token.tok_codes other than obj_id
        next_static_tokens = ''
        function_names = set()

        i = 0
        while i < len(self.wc_lang_expression._obj_model_tokens):
            obj_model_token = self.wc_lang_expression._obj_model_tokens[i]
            if obj_model_token.code == ObjModelTokenCodes.math_func_id:
                function_names.add(obj_model_token.token_string)
            if obj_model_token.code in self.NON_LANG_OBJ_ID_TOKENS:
                next_static_tokens = next_static_tokens + obj_model_token.token_string
            elif obj_model_token.code == ObjModelTokenCodes.obj_id:
                if next_static_tokens != '':
                    self.wc_sim_tokens.append(WcSimToken(SimTokCodes.other, next_static_tokens))
                    next_static_tokens = ''
                try:
                    dynamic_expression = DynamicComponent.get_dynamic_component(obj_model_token.model,
                                                                                obj_model_token.model_id)
                except:
                    raise MultialgorithmError("'{}.{} must be prepared to create '{}''".format(
                        obj_model_token.model.__class__.__name__, obj_model_token.model_id, self.id))
                self.wc_sim_tokens.append(WcSimToken(SimTokCodes.dynamic_expression,
                                                     obj_model_token.token_string,
                                                     dynamic_expression))
            else:   # pragma: no cover
                assert False, "unknown code {} in {}".format(obj_model_token.code, obj_model_token)
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
        assert hasattr(self, 'wc_sim_tokens'), "'{}' must use prepare() before eval()".format(self.id)
        for idx, sim_token in enumerate(self.wc_sim_tokens):
            if sim_token.code == SimTokCodes.dynamic_expression:
                self.expr_substrings[idx] = str(sim_token.dynamic_expression.eval(time))
        try:
            return eval(''.join(self.expr_substrings), {}, self.local_ns)
        except BaseException as e:
            raise MultialgorithmError("eval of '{}' raises {}: {}'".format(
                self.expression, type(e).__name__, str(e)))

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


class DynamicCompartment(DynamicComponent):
    """ A dynamic compartment

    A :obj:`DynamicCompartment` tracks the dynamic aggregate state of a compartment, primarily its
    mass. A :obj:`DynamicCompartment` is created for each `wc_lang` `Compartment` in a whole-cell
    model.

    Attributes:
        id (:obj:`str`): id of this :obj:`DynamicCompartment`, copied from `compartment`
        name (:obj:`str`): name of this :obj:`DynamicCompartment`, copied from `compartment`
        init_volume (:obj:`float`): initial volume, sampled from the distribution specified in the
            `wc_lang` model
        init_mass (:obj:`float`): initial mass
        species_population (:obj:`LocalSpeciesPopulation`): an object that represents
            the populations of species in this :obj:`DynamicCompartment`
        species_ids (:obj:`list` of :obj:`str`): the IDs of the species stored
            in this dynamic compartment; if `None`, use the IDs of all species in `species_population`
    """

    def __init__(self, dynamic_model, species_population, wc_lang_model, species_ids=None):
        """ Initialize this :obj:`DynamicCompartment`

        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            species_population (:obj:`LocalSpeciesPopulation`): an object that represents
                the populations of species in this :obj:`DynamicCompartment`
            wc_lang_model (:obj:`Compartment`): the corresponding static `wc_lang` `Compartment`
            species_ids (:obj:`list` of :obj:`str`, optional): the IDs of the species stored
                in this compartment; defaults to the IDs of all species in `species_population`

        Raises:
            :obj:`MultialgorithmError`: if `init_volume` is not a positive number
        """
        super(DynamicCompartment, self).__init__(dynamic_model, species_population, wc_lang_model)

        self.id = wc_lang_model.id
        self.name = wc_lang_model.name

        self.species_population = species_population
        self.species_ids = species_ids

        if wc_lang_model.init_volume and \
            are_terms_equivalent(wc_lang_model.init_volume.distribution, onto['WC:normal_distribution']):
            mean = wc_lang_model.init_volume.mean
            std = wc_lang_model.init_volume.std
            if numpy.isnan(std):
                std = mean / 10.
            self.init_volume = max(0., species_population.random_state.normal(mean, std))
        else:
            raise MultialgorithmError('Initial volume must be normally distributed')

        if math.isnan(self.init_volume):
            raise MultialgorithmError("DynamicCompartment {}: init_volume is NaN, but must be a positive "
                                      "number.".format(self.name))
        if self.init_volume <= 0:
            raise MultialgorithmError("DynamicCompartment {}: init_volume ({}) must be a positive number.".format(
                self.name, self.init_volume))

        self.init_mass = self.mass()
        if 0 == self.init_mass:
            warnings.warn("DynamicCompartment '{}': initial mass is 0".format(self.name))

        wc_lang_model.init_density.value = self.init_mass / self.init_volume

    def mass(self, time=None):
        """ Provide the total current mass of all species in this :obj:`DynamicCompartment`

        Args:
            time (number, optional): the current simulation time;

        Returns:
            :obj:`float`: this compartment's total current mass (g)
        """
        return self.species_population.compartmental_mass(self.id, time=time)

    def eval(self, time=None):
        """ Provide the mass of this :obj:`DynamicCompartment`

        Args:
            time (number, optional): the current simulation time;
        """
        return self.mass(time=time)

    def __str__(self):
        """ Provide a string representation of this :obj:`DynamicCompartment`

        Returns:
            :obj:`str`: a string representation of this compartment
        """
        values = []
        values.append("ID: " + self.id)
        values.append("Name: " + self.name)
        values.append("Initial volume (l^-1): {}".format(self.init_volume))
        values.append("Initial density (g l^-1): {}".format(self.init_mass / self.init_volume))
        values.append("Initial mass (g): {}".format(self.init_mass))
        values.append("Current mass (g): {}".format(self.mass()))
        values.append("Fold change mass: {}".format(self.mass() / self.init_mass))
        return "DynamicCompartment:\n{}".format('\n'.join(values))


# TODO(Arthur): define these in config data, which may come from wc_lang
EXTRACELLULAR_COMPARTMENT_ID = 'e'


class DynamicModel(object):
    """ Represent and access the dynamics of a whole-cell model simulation

    A `DynamicModel` provides access to dynamical components of the simulation, and
    determines aggregate properties that are not provided
    by other, more specific, dynamical components like species populations, submodels, and
    dynamic compartments.

    Attributes:
        dynamic_compartments (:obj:`dict`): map from compartment ID to :obj:`DynamicCompartment`\ ; the simulation's
            :obj:`DynamicCompartment`\ s, one for each compartment in `model`
        cellular_dyn_compartments (:obj:`list`): list of the cellular compartments
        species_population (:obj:`LocalSpeciesPopulation`): an object that represents
            the populations of species in this :obj:`DynamicCompartment`
        dynamic_species (:obj:`dict` of `DynamicSpecies`): the simulation's dynamic species,
            indexed by their ids
        dynamic_observables (:obj:`dict` of `DynamicObservable`): the simulation's dynamic observables,
            indexed by their ids
        dynamic_functions (:obj:`dict` of `DynamicFunction`): the simulation's dynamic functions,
            indexed by their ids
        dynamic_stop_conditions (:obj:`dict` of `DynamicStopCondition`): the simulation's stop conditions,
            indexed by their ids
        dynamic_parameters (:obj:`dict` of `DynamicParameter`): the simulation's parameters,
            indexed by their ids
    """

    def __init__(self, model, species_population, dynamic_compartments):
        """ Prepare a `DynamicModel` for a discrete-event simulation

        Args:
            model (:obj:`Model`): the description of the whole-cell model in `wc_lang`
            species_population (:obj:`LocalSpeciesPopulation`): an object that represents
                the populations of species in this :obj:`DynamicCompartment`
            dynamic_compartments (:obj:`dict`): the simulation's :obj:`DynamicCompartment`\ s, one for each
                compartment in `model`
        """
        self.dynamic_compartments = dynamic_compartments
        self.species_population = species_population
        self.num_submodels = len(model.get_submodels())

        # Classify compartments into extracellular and cellular; those which are not extracellular are cellular
        # Assumes at most one extracellular compartment
        extracellular_compartment = utils.get_component_by_id(model.get_compartments(),
                                                              EXTRACELLULAR_COMPARTMENT_ID)

        self.cellular_dyn_compartments = []
        for dynamic_compartment in dynamic_compartments.values():
            if dynamic_compartment.id == EXTRACELLULAR_COMPARTMENT_ID:
                continue
            self.cellular_dyn_compartments.append(dynamic_compartment)

        # === create dynamic objects that are not expressions ===
        # create dynamic parameters
        self.dynamic_parameters = {}
        for parameter in model.parameters:
            self.dynamic_parameters[parameter.id] = DynamicParameter(
                self, self.species_population,
                parameter, parameter.value)

        # create dynamic species
        self.dynamic_species = {}
        for species in model.get_species():
            self.dynamic_species[species.id] = DynamicSpecies(
                self, self.species_population,
                species)

        # === create dynamic expressions ===
        # create dynamic observables
        self.dynamic_observables = {}
        for observable in model.observables:
            self.dynamic_observables[observable.id] = DynamicObservable(
                self, self.species_population, observable,
                observable.expression._parsed_expression)

        # create dynamic functions
        self.dynamic_functions = {}
        for function in model.functions:
            self.dynamic_functions[function.id] = DynamicFunction(
                self, self.species_population, function,
                function.expression._parsed_expression)

        # create dynamic stop conditions
        self.dynamic_stop_conditions = {}
        for stop_condition in model.stop_conditions:
            self.dynamic_stop_conditions[stop_condition.id] = DynamicStopCondition(
                self, self.species_population,
                stop_condition, stop_condition.expression._parsed_expression)

        # prepare dynamic expressions
        for dynamic_expression_group in [self.dynamic_observables,
                                         self.dynamic_functions,
                                         self.dynamic_stop_conditions]:
            for dynamic_expression in dynamic_expression_group.values():
                dynamic_expression.prepare()

    def cell_mass(self):
        """ Compute the cell's mass

        Sum the mass of all :obj:`DynamicCompartment`\ s that are not extracellular.
        Assumes compartment volumes are in L and concentrations in mol/L.

        Returns:
            :obj:`float`: the cell's mass (g)
        """
        # TODO(Arthur): how should water be treated in mass calculations?
        return sum([dynamic_compartment.mass() for dynamic_compartment in self.cellular_dyn_compartments])

    def get_aggregate_state(self):
        """ Report the cell's aggregate state

        Returns:
            :obj:`dict`: the cell's aggregate state
        """
        aggregate_state = {
            'cell mass': self.cell_mass(),
        }

        compartments = {}
        for dynamic_compartment in self.cellular_dyn_compartments:
            compartments[dynamic_compartment.id] = {
                'name': dynamic_compartment.name,
                'mass': dynamic_compartment.mass(),
            }
        aggregate_state['compartments'] = compartments
        return aggregate_state

    def eval_dynamic_observables(self, time, observables_to_eval=None):
        """ Evaluate some dynamic observables at time `time`

        Args:
            time (:obj:`float`): the simulation time
            observables_to_eval (:obj:`list` of :obj:`str`, optional): if provided, ids of the observables to
                evaluate; otherwise, evaluate all observables

        Returns:
            :obj:`dict`: map from the IDs of dynamic observables in `observables_to_eval` to their
                values at simulation time `time`
        """
        if observables_to_eval is None:
            observables_to_eval = list(self.dynamic_observables.keys())
        evaluated_observables = {}
        for dyn_observable_id in observables_to_eval:
            evaluated_observables[dyn_observable_id] = self.dynamic_observables[dyn_observable_id].eval(time)
        return evaluated_observables

    def eval_dynamic_functions(self, time, functions_to_eval=None):
        """ Evaluate some dynamic functions at time `time`

        Args:
            time (:obj:`float`): the simulation time
            functions_to_eval (:obj:`list` of :obj:`str`, optional): if provided, ids of the functions to
                evaluate; otherwise, evaluate all functions

        Returns:
            :obj:`dict`: map from the IDs of dynamic functions in `functions_to_eval` to their
                values at simulation time `time`
        """
        if functions_to_eval is None:
            functions_to_eval = list(self.dynamic_functions.keys())
        evaluated_functions = {}
        for dyn_function_id in functions_to_eval:
            evaluated_functions[dyn_function_id] = self.dynamic_functions[dyn_function_id].eval(time)
        return evaluated_functions

    def get_num_submodels(self):
        """ Provide the number of submodels

        Returns:
            :obj:`int`: the number of submodels
        """
        return self.num_submodels

    def set_stop_condition(self, simulation):
        """ Set the simulation's stop_condition

        A simulation's stop condition is constructed as a logical 'or' of all `StopConditions` in
        a model.

        Args:
            simulation (:obj:`SimulationEngine`): a simulation
        """
        if self.dynamic_stop_conditions:
            dynamic_stop_conditions = self.dynamic_stop_conditions.values()

            def stop_condition(time):
                for dynamic_stop_condition in dynamic_stop_conditions:
                    print('checking dynamic_stop_condition', dynamic_stop_condition)
                    if dynamic_stop_condition.eval(time):
                        return True
                return False
            simulation.set_stop_condition(stop_condition)

    def get_species_count_array(self, now):     # pragma no cover, not used
        """ Map current species counts into an numpy array

        Args:
            now (:obj:`float`): the current simulation time

        Returns:
            numpy array, #species x # compartments, containing count of specie in compartment
        """
        species_counts = numpy.zeros((len(model.species), len(model.compartments)))
        for species in model.species:
            for compartment in model.compartments:
                species_id = Species._gen_id(species.id, compartment.id)
                species_counts[species.index, compartment.index] = \
                    model.local_species_population.read_one(now, species_id)
        return species_counts


WC_LANG_MODEL_TO_DYNAMIC_MODEL = {
    wc_lang.Function: DynamicFunction,
    wc_lang.Parameter: DynamicParameter,
    wc_lang.Species: DynamicSpecies,
    wc_lang.Observable: DynamicObservable,
    wc_lang.StopCondition: DynamicStopCondition,
    wc_lang.DfbaObjective: DynamicDfbaObjective,
    wc_lang.RateLaw: DynamicRateLaw,
    wc_lang.Compartment: DynamicCompartment,
}
