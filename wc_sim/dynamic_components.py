""" Dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-02-07
:Copyright: 2017-2019, Karr Lab
:License: MIT
"""

from enum import Enum, auto
from pprint import pformat
import collections
import inspect
import itertools
import math
import networkx
import numpy
import warnings

from obj_tables.math.expression import Expression, ObjTablesTokenCodes
from wc_lang import Species, Compartment
from wc_onto import onto
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_errors import MultialgorithmError, MultialgorithmWarning
from wc_sim.species_populations import LocalSpeciesPopulation
from wc_utils.util.enumerate import CaseInsensitiveEnum
from wc_utils.util.ontology import are_terms_equivalent
import obj_tables
import wc_lang
import wc_sim.config
import wc_sim.submodels

# mapping from wc_lang Models to DynamicComponents
WC_LANG_MODEL_TO_DYNAMIC_MODEL = {}


class SimTokCodes(int, CaseInsensitiveEnum):
    """ Token codes used in WcSimTokens """
    dynamic_expression = 1
    other = 2


# a token in DynamicExpression._obj_tables_tokens
WcSimToken = collections.namedtuple('WcSimToken', 'code, token_string, dynamic_expression')
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
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species
                population store
            wc_lang_model (:obj:`obj_tables.Model`): a corresponding `wc_lang` `Model`, from which this
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
            model_type (:obj:`Object`): a `wc_lang` Model type represented by a subclass of
                `obj_tables.Model`, an instance of `obj_tables.Model`, or a string name for a `obj_tables.Model`

        Returns:
            :obj:`type`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the corresponding dynamic component type cannot be determined
        """
        if isinstance(model_type, type) and issubclass(model_type, obj_tables.Model):
            if model_type in WC_LANG_MODEL_TO_DYNAMIC_MODEL:
                return WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type]
            raise MultialgorithmError(f"model class of type '{model_type.__name__}' not found")

        if isinstance(model_type, obj_tables.Model):
            if model_type.__class__ in WC_LANG_MODEL_TO_DYNAMIC_MODEL:
                return WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type.__class__]
            raise MultialgorithmError(f"model of type '{model_type.__class__.__name__}' not found")

        if isinstance(model_type, str):
            model_type_type = getattr(wc_lang, model_type, None)
            if model_type_type is not None:
                if model_type_type in WC_LANG_MODEL_TO_DYNAMIC_MODEL:
                    return WC_LANG_MODEL_TO_DYNAMIC_MODEL[model_type_type]
                raise MultialgorithmError(f"model of type '{model_type_type.__name__}' not found")
            raise MultialgorithmError(f"model type '{model_type}' not defined")

        raise MultialgorithmError(f"model type '{model_type}' has wrong type")

    @staticmethod
    def get_dynamic_component(model_type, id):
        """ Get a simulation's dynamic component

        Args:
            model_type (:obj:`type`): the subclass of `DynamicComponent` (or `obj_tables.Model`)
                being retrieved
            id (:obj:`str`): the dynamic component's id

        Returns:
            :obj:`DynamicComponent`: the dynamic component

        Raises:
            :obj:`MultialgorithmError`: if the dynamic component cannot be found
        """
        if not inspect.isclass(model_type) or not issubclass(model_type, DynamicComponent):
            model_type = DynamicComponent.get_dynamic_model_type(model_type)
        if model_type not in DynamicComponent.dynamic_components_objs:
            raise MultialgorithmError(f"model type '{model_type.__name__}' not in "
                                      f"DynamicComponent.dynamic_components_objs")
        if id not in DynamicComponent.dynamic_components_objs[model_type]:
            raise MultialgorithmError(f"model type '{model_type.__name__}' with id='{id}' not in "
                                      f"DynamicComponent.dynamic_components_objs")
        return DynamicComponent.dynamic_components_objs[model_type][id]

    def __str__(self):
        """ Provide a readable representation of this `DynamicComponent`

        Returns:
            :obj:`str`: a readable representation of this `DynamicComponent`
        """
        rv = ['DynamicComponent:']
        rv.append(f"type: {self.__class__.__name__}")
        rv.append(f"id: {self.id}")
        return '\n'.join(rv)


class DynamicExpression(DynamicComponent):
    """ Simulation representation of a mathematical expression, based on :obj:`ParsedExpression`

    Attributes:
        expression (:obj:`str`): the expression defined in the `wc_lang` Model
        wc_sim_tokens (:obj:`list` of :obj:`WcSimToken`): a tokenized, compressed representation of
            `expression`
        expr_substrings (:obj:`list` of :obj:`str`): strings which are joined to form the string
            which is 'eval'ed
        local_ns (:obj:`dict`): pre-computed local namespace of functions used in `expression`
    """

    NON_LANG_OBJ_ID_TOKENS = set([ObjTablesTokenCodes.math_func_id,
                                  ObjTablesTokenCodes.number,
                                  ObjTablesTokenCodes.op,
                                  ObjTablesTokenCodes.other])

    def __init__(self, dynamic_model, local_species_population, wc_lang_model, wc_lang_expression):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species
                population store
            wc_lang_model (:obj:`obj_tables.Model`): the corresponding `wc_lang` `Model`
            wc_lang_expression (:obj:`ParsedExpression`): an analyzed and validated expression

        Raises:
            :obj:`MultialgorithmError`: if `wc_lang_expression` does not contain an analyzed,
                validated expression
        """

        super().__init__(dynamic_model, local_species_population, wc_lang_model)

        # wc_lang_expression must have been successfully `tokenize`d.
        if not wc_lang_expression._obj_tables_tokens:
            raise MultialgorithmError(f"_obj_tables_tokens cannot be empty - ensure that "
                                      f"'{wc_lang_model}' is valid")
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
        # optimization: combine together adjacent obj_tables_token.tok_codes other than obj_id
        next_static_tokens = ''
        function_names = set()

        i = 0
        while i < len(self.wc_lang_expression._obj_tables_tokens):
            obj_tables_token = self.wc_lang_expression._obj_tables_tokens[i]
            if obj_tables_token.code == ObjTablesTokenCodes.math_func_id:
                function_names.add(obj_tables_token.token_string)
            if obj_tables_token.code in self.NON_LANG_OBJ_ID_TOKENS:
                next_static_tokens = next_static_tokens + obj_tables_token.token_string
            elif obj_tables_token.code == ObjTablesTokenCodes.obj_id:
                if next_static_tokens != '':
                    self.wc_sim_tokens.append(WcSimToken(SimTokCodes.other, next_static_tokens))
                    next_static_tokens = ''
                try:
                    dynamic_expression = DynamicComponent.get_dynamic_component(obj_tables_token.model,
                                                                                obj_tables_token.model_id)
                except:
                    raise MultialgorithmError("'{}.{} must be prepared to create '{}''".format(
                        obj_tables_token.model.__class__.__name__, obj_tables_token.model_id, self.id))
                self.wc_sim_tokens.append(WcSimToken(SimTokCodes.dynamic_expression,
                                                     obj_tables_token.token_string,
                                                     dynamic_expression))
            else:   # pragma: no cover
                assert False, f"unknown code {obj_tables_token.code} in {obj_tables_token}"
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
                raise MultialgorithmError(f"loading expression '{self.expression}' "
                                          f"cannot find function '{func_name}'")

    def eval(self, time):
        """ Evaluate this mathematical expression

        Approach:
            * Replace references to related Models in `self.wc_sim_tokens` with their values
            * Join the elements of `self.wc_sim_tokens` into a Python expression
            * `eval` the Python expression

        Args:
            time (:obj:`float`): the simulation time at which the expression should be evaluated

        Returns:
            :obj:`float` or :obj:`bool`: the value of this :obj:`DynamicExpression` at time `time`

        Raises:
            :obj:`MultialgorithmError`: if Python `eval` raises an exception
        """
        assert hasattr(self, 'wc_sim_tokens'), f"'{self.id}' must use prepare() before eval()"

        # if caching is enabled & the expression's value is cached, return it
        if self.dynamic_model.cache_manager.caching():
            try:
                return self.dynamic_model.cache_manager.get(self)
            except MultialgorithmError:
                pass

        for idx, sim_token in enumerate(self.wc_sim_tokens):
            if sim_token.code == SimTokCodes.dynamic_expression:
                self.expr_substrings[idx] = str(sim_token.dynamic_expression.eval(time))
        try:
            value = eval(''.join(self.expr_substrings), {}, self.local_ns)
            # if caching is enabled cache the expression's value
            self.dynamic_model.cache_manager.set(self, value)
            return value
        except BaseException as e:
            raise MultialgorithmError(f"eval of '{self.expression}' "
                                      f"raises {type(e).__name__}: {str(e)}'")

    def __str__(self):
        """ Provide a readable representation of this `DynamicExpression`

        Returns:
            :obj:`str`: a readable representation of this `DynamicExpression`
        """
        rv = ['DynamicExpression:']
        rv.append(f"type: {self.__class__.__name__}")
        rv.append(f"id: {self.id}")
        rv.append(f"expression: {self.expression}")
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


class DynamicDfbaObjective(DynamicExpression):
    """ The dynamic representation of a :obj:`wc_lang.DfbaObjective`
    """

    def __init__(self, *args):
        super().__init__(*args)


class DynamicRateLaw(DynamicExpression):
    """ The dynamic representation of a :obj:`wc_lang.RateLaw`
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
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species
                population store
            wc_lang_model (:obj:`obj_tables.Model`): the corresponding :obj:`wc_lang.Parameter`
            value (:obj:`float`): the parameter's value
        """
        super().__init__(dynamic_model, local_species_population, wc_lang_model)
        self.value = value

    def eval(self, time):
        """ Provide the value of this parameter

        Args:
            time (:obj:`float`): the current simulation time; not needed, but included so that all
                dynamic expression models have the same signature for `eval`

        Returns:
            :obj:`float`: the dynamic parameter's value
        """
        return self.value


class DynamicSpecies(DynamicComponent):
    """ The dynamic representation of a :obj:`wc_lang.Species`
    """

    def __init__(self, dynamic_model, local_species_population, wc_lang_model):
        """
        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            local_species_population (:obj:`LocalSpeciesPopulation`): the simulation's species
                population store
            wc_lang_model (:obj:`obj_tables.Model`): the corresponding :obj:`wc_lang.Species`
        """
        super().__init__(dynamic_model, local_species_population, wc_lang_model)

    def eval(self, time):
        """ Provide the population of this species

        Args:
            time (:obj:`float`): the current simulation time

        Returns:
            :obj:`float`: the population of this species at time `time`
        """
        return self.local_species_population.read_one(time, self.id)


class DynamicCompartment(DynamicComponent):
    """ A dynamic compartment

    A :obj:`DynamicCompartment` tracks the dynamic aggregate state of a compartment. A
    :obj:`DynamicCompartment` is created for each `wc_lang` `Compartment` in a whole-cell model.

    Attributes:
        id (:obj:`str`): id of this :obj:`DynamicCompartment`, copied from the `wc_lang` `Compartment`
        biological_type (:obj:`pronto.term.Term`): biological type of this :obj:`DynamicCompartment`,
            copied from the `Compartment`
        physical_type (:obj:`pronto.term.Term`): physical type of this :obj:`DynamicCompartment`,
            copied from the `Compartment`
        random_state (:obj:`numpy.random.RandomState`): a random state
        init_volume (:obj:`float`): initial volume, sampled from the distribution specified in the
            `wc_lang` model
        init_accounted_mass (:obj:`float`): the initial mass accounted for by the initial species
        init_mass (:obj:`float`): initial mass, including the mass not accounted for by
            explicit species
        init_density (:obj:`float`): the initial density of this :obj:`DynamicCompartment`, as
            specified by the model; this is the *constant* density of the compartment
        init_accounted_density (:obj:`float`): the initial density accounted for by the
            initial species
        accounted_fraction (:obj:`float`): the fraction of the initial mass or density accounted
            for by initial species; assumed to be constant throughout a dynamical model
        species_population (:obj:`LocalSpeciesPopulation`): the simulation's species population store
        species_ids (:obj:`list` of :obj:`str`): the IDs of the species stored in this
            :obj:`DynamicCompartment`\ ; if `None`, use the IDs of all species in `species_population`
    """

    def __init__(self, dynamic_model, random_state, wc_lang_compartment, species_ids=None):
        """ Initialize the volume and density of this :obj:`DynamicCompartment`\ .

        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's dynamic model
            random_state (:obj:`numpy.random.RandomState`): a random state
            wc_lang_compartment (:obj:`Compartment`): the corresponding static `wc_lang` `Compartment`
            species_ids (:obj:`list` of :obj:`str`, optional): the IDs of the species stored
                in this compartment

        Raises:
            :obj:`MultialgorithmError`: if `self.init_volume` or `self.init_density` are not
                positive numbers
        """
        super(DynamicCompartment, self).__init__(dynamic_model, None, wc_lang_compartment)

        self.id = wc_lang_compartment.id
        self.biological_type = wc_lang_compartment.biological_type
        self.physical_type = wc_lang_compartment.physical_type
        self.species_ids = species_ids

        # obtain initial compartment volume by sampling its specified distribution
        if wc_lang_compartment.init_volume and \
            are_terms_equivalent(wc_lang_compartment.init_volume.distribution,
                                 onto['WC:normal_distribution']):
            mean = wc_lang_compartment.init_volume.mean
            std = wc_lang_compartment.init_volume.std
            if numpy.isnan(std):
                config_multialgorithm = wc_sim.config.core.get_config()['wc_sim']['multialgorithm']
                MEAN_TO_STD_DEV_RATIO = config_multialgorithm['mean_to_std_dev_ratio']
                std = mean / MEAN_TO_STD_DEV_RATIO
            self.init_volume = ModelUtilities.non_neg_normal_sample(random_state, mean, std)
        else:
            raise MultialgorithmError('Initial volume must be normally distributed')

        if math.isnan(self.init_volume):    # pragma no cover: cannot be True
            raise MultialgorithmError(f"DynamicCompartment {self.id}: init_volume is NaN, but must "
                                      f"be a positive number.")
        if self.init_volume <= 0:
            raise MultialgorithmError(f"DynamicCompartment {self.id}: init_volume "
                                      f"({self.init_volume}) must be a positive number.")

        if not self._is_abstract():
            init_density = wc_lang_compartment.init_density.value
            if math.isnan(init_density):
                raise MultialgorithmError(f"DynamicCompartment {self.id}: init_density is NaN, "
                                          f"but must be a positive number.")
            if init_density <= 0:
                raise MultialgorithmError(f"DynamicCompartment {self.id}: init_density "
                                          f"({init_density}) must be a positive number.")
            self.init_density = init_density

    def initialize_mass_and_density(self, species_population):
        """ Initialize the species populations and the mass accounted for by species.

        Also initialize the fraction of density accounted for by species, `self.accounted_fraction`.

        Args:
            species_population (:obj:`LocalSpeciesPopulation`): the simulation's
                species population store

        Raises:
            :obj:`MultialgorithmError`: if `accounted_fraction == 0` or
                if `MAX_ALLOWED_INIT_ACCOUNTED_FRACTION < accounted_fraction`
        """
        config_multialgorithm = wc_sim.config.core.get_config()['wc_sim']['multialgorithm']
        MAX_ALLOWED_INIT_ACCOUNTED_FRACTION = config_multialgorithm['max_allowed_init_accounted_fraction']

        self.species_population = species_population
        self.init_accounted_mass = self.accounted_mass(time=0)
        if self._is_abstract():
            self.init_mass = self.init_accounted_mass
        else:
            self.init_mass = self.init_density * self.init_volume
            self.init_accounted_density = self.init_accounted_mass / self.init_volume
            # calculate fraction of initial mass or density represented by species
            self.accounted_fraction = self.init_accounted_density / self.init_density
            # also, accounted_fraction = self.init_accounted_mass / self.init_mass

            # usually epsilon < accounted_fraction <= 1, where epsilon depends on how thoroughly
            # processes in the compartment are characterized
            if 0 == self.accounted_fraction:
                raise MultialgorithmError(f"DynamicCompartment '{self.id}': "
                                          f"initial accounted ratio is 0")
            elif 1.0 < self.accounted_fraction <= MAX_ALLOWED_INIT_ACCOUNTED_FRACTION:
                warnings.warn(f"DynamicCompartment '{self.id}': "
                              f"initial accounted ratio ({self.accounted_fraction:.3E}) "
                              f"greater than 1.0", MultialgorithmWarning)
            if MAX_ALLOWED_INIT_ACCOUNTED_FRACTION < self.accounted_fraction:
                raise MultialgorithmError(f"DynamicCompartment {self.id}: "
                                          f"initial accounted ratio ({self.accounted_fraction:.3E}) "
                                          f"greater than MAX_ALLOWED_INIT_ACCOUNTED_FRACTION "
                                          f"({MAX_ALLOWED_INIT_ACCOUNTED_FRACTION}).")

    def _is_abstract(self):
        """ Indicate whether this is an abstract compartment

        An abstract compartment has a `physical_type` of `abstract_compartment` as defined in the WC
        ontology.
        Its contents do not represent physical matter, so no relationship exists among its mass,
        volume and
        density. Its volume is constant and its density is ignored and need not be defined. Abstract
        compartments are useful for modeling dynamics that are not based on physical chemistry, and
        for testing models and software.
        These :obj:`DynamicCompartment` attributes are not initialized in abstract compartments:
        `init_density`, `init_accounted_density` and `accounted_fraction`.

        Returns:
            :obj:`bool`: whether this is an abstract compartment
        """
        return are_terms_equivalent(self.physical_type, onto['WC:abstract_compartment'])

    def accounted_mass(self, time=None):
        """ Provide the total current mass of all species in this :obj:`DynamicCompartment`

        Args:
            time (:obj:`Rational`, optional): the current simulation time

        Returns:
            :obj:`float`: the total current mass of all species (g)
        """
        # if caching is enabled & the expression's value is cached, return it

        if self.dynamic_model.cache_manager.caching():
            try:
                return self.dynamic_model.cache_manager.get(self)
            except MultialgorithmError:
                pass

        value = self.species_population.compartmental_mass(self.id, time=time)
        # if caching is enabled cache the accounted_mass
        self.dynamic_model.cache_manager.set(self, value)
        return value

    def accounted_volume(self, time=None):
        """ Provide the current volume occupied by all species in this :obj:`DynamicCompartment`

        Args:
            time (:obj:`Rational`, optional): the current simulation time

        Returns:
            :obj:`float`: the current volume of all species (l)
        """
        if self._is_abstract():
            return self.volume()
        else:
            return self.accounted_mass(time=time) / self.init_density

    def mass(self, time=None):
        """ Provide the total current mass of this :obj:`DynamicCompartment`

        This mass includes the mass not accounted for by explicit species, as determined by
        the initial specified density, specified volume, and mass accounted for by species.

        Args:
            time (:obj:`Rational`, optional): the current simulation time

        Returns:
            :obj:`float`: this compartment's total current mass (g)
        """
        if self._is_abstract():
            return self.accounted_mass(time=time)
        else:
            return self.accounted_mass(time=time) / self.accounted_fraction

    def volume(self, time=None):
        """ Provide the current volume of this :obj:`DynamicCompartment`

        This volume includes the volume not accounted for by explicit species, as determined by
        the ratio of the specified initial density to the initial density accounted for by species.

        Args:
            time (:obj:`Rational`, optional): the current simulation time

        Returns:
            :obj:`float`: this compartment's current volume (l)
        """
        if self._is_abstract():
            return self.init_volume
        else:
            return self.accounted_volume(time=time) / self.accounted_fraction

    # todo: make time required, to avoid the possibility of eval'ing an expression @ multiple times
    def eval(self, time=None):
        """ Provide the mass of this :obj:`DynamicCompartment`

        Args:
            time (:obj:`Rational`, optional): the current simulation time

        Returns:
            :obj:`float`: this compartment's current mass (g)
        """
        return self.mass(time=time)

    def fold_change_total_mass(self, time=None):
        """ Provide the fold change of the total mass of this :obj:`DynamicCompartment`

        Args:
            time (:obj:`Rational`, optional): the current simulation time

        Returns:
            :obj:`float`: the fold change of the total mass of this compartment
        """
        return self.mass(time=time) / self.init_mass

    def fold_change_total_volume(self, time=None):
        """ Provide the fold change of the total volume of this :obj:`DynamicCompartment`

        Args:
            time (:obj:`Rational`, optional): the current simulation time

        Returns:
            :obj:`float`: the fold change of the total volume of this compartment
        """
        return self.volume(time=time) / self.init_volume

    def _initialized(self):
        """ Indicate whether this :obj:`DynamicCompartment` has been initialized

        Returns:
            :obj:`bool`: whether this compartment has been initialized by `initialize_mass_and_density()`
        """
        return hasattr(self, 'init_accounted_mass')

    def __str__(self):
        """ Provide a string representation of this :obj:`DynamicCompartment`

        Returns:
            :obj:`str`: a string representation of this compartment at the current simulation time
        """
        values = []
        values.append("ID: " + self.id)
        if self._initialized():
            values.append(f"Initialization state: '{self.id}' has been initialized.")
        else:
            values.append(f"Initialization state: '{self.id}' has not been initialized.")

        # todo: be careful with units; if initial values are specified in other units, are they converted?
        values.append(f"Initial volume (l): {self.init_volume:.3E}")
        values.append(f"Physical type: {self.physical_type.name}")
        values.append(f"Biological type: {self.biological_type.name}")
        if not self._is_abstract():
            values.append(f"Specified density (g l^-1): {self.init_density}")
        if self._initialized():
            values.append(f"Initial mass in species (g): {self.init_accounted_mass:.3E}")
            values.append(f"Initial total mass (g): {self.init_mass:.3E}")
            if not self._is_abstract():
                values.append(f"Fraction of mass accounted for by species (dimensionless): "
                              f"{self.accounted_fraction:.3E}")

            values.append(f"Current mass in species (g): {self.accounted_mass():.3E}")
            values.append(f"Current total mass (g): {self.mass():.3E}")
            values.append(f"Fold change total mass: {self.fold_change_total_mass():.3E}")

            values.append(f"Current volume in species (l): {self.accounted_volume():.3E}")
            values.append(f"Current total volume (l): {self.volume():.3E}")
            values.append(f"Fold change total volume: {self.fold_change_total_volume():.3E}")

        return "DynamicCompartment:\n{}".format('\n'.join(values))


class DynamicModel(object):
    """ Represent and access the dynamics of a whole-cell model simulation

    A `DynamicModel` provides access to dynamical components of the simulation, and
    determines aggregate properties that are not provided
    by other, more specific, dynamical components like species populations, submodels, and
    dynamic compartments.

    Attributes:
        id (:obj:`str`): id of the `wc_lang` model
        dynamic_compartments (:obj:`dict`): map from compartment ID to :obj:`DynamicCompartment`\ ;
            the simulation's :obj:`DynamicCompartment`\ s, one for each compartment in `model`
        cellular_dyn_compartments (:obj:`list`): list of the cellular compartments
        species_population (:obj:`LocalSpeciesPopulation`): populations of all the species in
            the model
        dynamic_submodels (:obj:`dict` of `DynamicSubmodel`): the simulation's dynamic submodels,
            indexed by their ids
        dynamic_species (:obj:`dict` of `DynamicSpecies`): the simulation's dynamic species,
            indexed by their ids
        dynamic_parameters (:obj:`dict` of `DynamicParameter`): the simulation's parameters,
            indexed by their ids
        dynamic_observables (:obj:`dict` of `DynamicObservable`): the simulation's dynamic
            observables, indexed by their ids
        dynamic_functions (:obj:`dict` of `DynamicFunction`): the simulation's dynamic functions,
            indexed by their ids
        dynamic_stop_conditions (:obj:`dict` of `DynamicStopCondition`): the simulation's stop
            conditions, indexed by their ids
        dynamic_rate_laws (:obj:`dict` of `DynamicRateLaw`): the simulation's rate laws,
            indexed by their ids
        dynamic_dfba_objectives (:obj:`dict` of `DynamicDfbaObjective`): the simulation's dFBA
            Objective, indexed by their ids
        cache_manager (:obj:`CacheManager`): a cache for potentially expensive expression evaluations
            that get repeated
        rxn_expression_dependencies (:obj:`dict`): map from reactions to lists of expressions
            whose values depend on species with non-zero stoichiometry in the reaction
        # TODO (APG): OPTIMIZE DFBA CACHING: describe dFBA caching optimization
        continuous_rxn_dependencies (:obj:`dict`): map from ids of continuous submodels to sets
            identifying expressions whose values depend on species with non-zero stoichiometry in
            reaction(s) modeled by the submodel
        all_continuous_rxn_dependencies (:obj:`tuple`): all expressions in `continuous_rxn_dependencies`
    """
    AGGREGATE_VALUES = ['mass', 'volume', 'accounted mass', 'accounted volume']
    def __init__(self, model, species_population, dynamic_compartments):
        """ Prepare a `DynamicModel` for a discrete-event simulation

        Args:
            model (:obj:`Model`): the description of the whole-cell model in `wc_lang`
            species_population (:obj:`LocalSpeciesPopulation`): the simulation's
                species population store
            dynamic_compartments (:obj:`dict`): the simulation's :obj:`DynamicCompartment`\ s, one
                for each compartment in `model`

        Raises:
            :obj:`MultialgorithmError`: if the model has no cellular compartments
        """
        self.id = model.id
        self.dynamic_compartments = dynamic_compartments
        self.species_population = species_population
        self.num_submodels = len(model.get_submodels())

        # determine cellular compartments
        self.cellular_dyn_compartments = []
        for dynamic_compartment in dynamic_compartments.values():
            if dynamic_compartment.biological_type == onto['WC:cellular_compartment']:
                self.cellular_dyn_compartments.append(dynamic_compartment)
        if dynamic_compartments and not self.cellular_dyn_compartments:
            raise MultialgorithmError(f"model '{model.id}' must have at least 1 cellular compartment")

        # === create dynamic objects that are not expressions ===
        # create dynamic parameters
        self.dynamic_parameters = {}
        for parameter in model.parameters:
            self.dynamic_parameters[parameter.id] = \
                DynamicParameter(self, self.species_population, parameter, parameter.value)

        # create dynamic species
        self.dynamic_species = {}
        for species in model.get_species():
            self.dynamic_species[species.id] = \
                DynamicSpecies(self, self.species_population, species)

        # === create dynamic expressions ===
        # create dynamic observables
        self.dynamic_observables = {}
        for observable in model.observables:
            self.dynamic_observables[observable.id] = \
                DynamicObservable(self, self.species_population, observable,
                                  observable.expression._parsed_expression)

        # create dynamic functions
        self.dynamic_functions = {}
        for function in model.functions:
            self.dynamic_functions[function.id] = \
                DynamicFunction(self, self.species_population, function,
                                function.expression._parsed_expression)

        # create dynamic stop conditions
        self.dynamic_stop_conditions = {}
        for stop_condition in model.stop_conditions:
            self.dynamic_stop_conditions[stop_condition.id] = \
                DynamicStopCondition(self, self.species_population, stop_condition,
                                     stop_condition.expression._parsed_expression)

        # create dynamic rate laws
        self.dynamic_rate_laws = {}
        for rate_law in model.rate_laws:
            self.dynamic_rate_laws[rate_law.id] = \
                DynamicRateLaw(self, self.species_population, rate_law,
                                     rate_law.expression._parsed_expression)

        # create dynamic dFBA Objectives
        self.dynamic_dfba_objectives = {}
        for dfba_objective in model.dfba_objs:
            error = dfba_objective.expression.validate()
            assert error is None, str(error)
            self.dynamic_dfba_objectives[dfba_objective.id] = \
                DynamicDfbaObjective(self, self.species_population, dfba_objective,
                                     dfba_objective.expression._parsed_expression)

        # prepare dynamic expressions
        for dynamic_expression_group in [self.dynamic_observables,
                                         self.dynamic_functions,
                                         self.dynamic_stop_conditions,
                                         self.dynamic_rate_laws]:
            for dynamic_expression in dynamic_expression_group.values():
                dynamic_expression.prepare()

        # initialize cache manager
        self.cache_manager = CacheManager()

    def cell_mass(self):
        """ Provide the cell's current mass

        Sum the mass of all cellular :obj:`DynamicCompartment`\ s.

        Returns:
            :obj:`float`: the cell's current mass (g)
        """
        return sum([dynamic_compartment.mass()
                    for dynamic_compartment in self.cellular_dyn_compartments])

    def cell_volume(self):
        """ Provide the cell's current volume

        Sum the volume of all cellular :obj:`DynamicCompartment`\ s.

        Returns:
            :obj:`float`: the cell's current volume (l)
        """
        return sum([dynamic_compartment.volume()
                    for dynamic_compartment in self.cellular_dyn_compartments])

    def cell_growth(self):
        """ Report the cell's growth in cell/s, relative to the cell's initial volume

        Returns:
            :obj:`float`: growth in cell/s, relative to the cell's initial volume
        """
        # TODO(Arthur): implement growth measurement
        pass

    def cell_accounted_mass(self):
        """ Provide the total current mass of all species in the cell

        Sum the current mass of all species in cellular :obj:`DynamicCompartment`\ s.

        Returns:
            :obj:`float`: the current mass of all species in the cell (g)
        """
        return sum([dynamic_compartment.accounted_mass()
                   for dynamic_compartment in self.cellular_dyn_compartments])

    def cell_accounted_volume(self):
        """ Provide the current volume occupied by all species in the cell

        Sum the current volume occupied by all species in cellular :obj:`DynamicCompartment`\ s.

        Returns:
            :obj:`float`: the current volume occupied by all species in the cell (l)
        """
        return sum([dynamic_compartment.accounted_volume()
                   for dynamic_compartment in self.cellular_dyn_compartments])

    def get_aggregate_state(self):
        """ Report the cell's aggregate state

        Returns:
            :obj:`dict`: the cell's aggregate state
        """
        # get the state values configured in DynamicModel.AGGREGATE_VALUES
        aggregate_state = {}

        cell_aggregate_values = [f'cell {value}' for value in self.AGGREGATE_VALUES]
        for cell_aggregate_value in cell_aggregate_values:
            aggregate_func = getattr(self, cell_aggregate_value.replace(' ', '_'))
            aggregate_state[cell_aggregate_value] = aggregate_func()

        compartment_values = {}
        for dynamic_compartment in self.cellular_dyn_compartments:
            compartment_values[dynamic_compartment.id] = {}
            for aggregate_value in self.AGGREGATE_VALUES:
                aggregate_func = getattr(dynamic_compartment, aggregate_value.replace(' ', '_'))
                compartment_values[dynamic_compartment.id][aggregate_value] = aggregate_func()
        aggregate_state['compartments'] = compartment_values
        return aggregate_state

    def eval_dynamic_observables(self, time, observables_to_eval=None):
        """ Evaluate some dynamic observables at time `time`

        Args:
            time (:obj:`float`): the simulation time
            observables_to_eval (:obj:`list` of :obj:`str`, optional): if provided, ids of the
                observables to evaluate; otherwise, evaluate all observables

        Returns:
            :obj:`dict`: map from the IDs of dynamic observables in `observables_to_eval` to their
                values at simulation time `time`
        """
        if observables_to_eval is None:
            observables_to_eval = list(self.dynamic_observables.keys())
        evaluated_observables = {}
        for dyn_obsable_id in observables_to_eval:
            evaluated_observables[dyn_obsable_id] = self.dynamic_observables[dyn_obsable_id].eval(time)
        return evaluated_observables

    def eval_dynamic_functions(self, time, functions_to_eval=None):
        """ Evaluate some dynamic functions at time `time`

        Args:
            time (:obj:`float`): the simulation time
            functions_to_eval (:obj:`list` of :obj:`str`, optional): if provided, ids of the
                functions to evaluate; otherwise, evaluate all functions

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

    def get_stop_condition(self):
        """ Provide a simulation's stop condition

        A simulation's stop condition is constructed as a logical 'or' of all :obj:`StopConditions`
        in a model.

        Returns:
            :obj:`function`: a function which computes the logical 'or' of all :obj:`StopConditions`,
                or `None` if no stop condition are defined
        """
        if self.dynamic_stop_conditions:
            dynamic_stop_conditions = self.dynamic_stop_conditions.values()

            def all_stop_conditions(time):
                for dynamic_stop_condition in dynamic_stop_conditions:
                    if dynamic_stop_condition.eval(time):
                        return True
                return False
            return all_stop_conditions
        else:
            return None

    def get_species_count_array(self, now):     # pragma no cover, not used
        """ Map current species counts into an numpy array

        Args:
            now (:obj:`float`): the current simulation time

        Returns:
            numpy array, #species x # compartments, containing count of species in compartment
        """
        species_counts = numpy.zeros((len(model.species), len(model.compartments)))
        for species in model.species:
            for compartment in model.compartments:
                species_id = Species.gen_id(species.id, compartment.id)
                species_counts[species.index, compartment.index] = \
                    model.local_species_population.read_one(now, species_id)
        return species_counts

    def obtain_dependencies(self, model):
        """ Obtain the dependencies of expressions on reactions in a WC Lang model

        An expression depends on a reaction if the expression uses any species whose population
        changes when the reaction executes, or the expression uses an expression that depends on
        the reaction.

        When caching is active, these dependencies identify which cached expressions to invalidate.
        They're also used by the Next Reaction Method to determine which rate laws must be evaluated
        after a reaction executes.

        `obtain_dependencies` is memory and compute intensive because it builds and walks an in-memory
        DAG that represents the dependency relationships among the WC-Lang models used by a whole-cell
        model.
        If a simulation fails with the error "killed" and no other information, then it is probably
        running on a system or in a container which does not have sufficient memory to complete
        this function. Try running on a system with more memory, simulating a smaller model, or
        disabling caching in `wc_sim.cfg`.

        Args:
            model (:obj:`Model`): the description of the whole-cell model in `wc_lang`

        Returns:
            :obj:`dict` of :obj:`list`: the dependencies of expressions on reactions, which maps each
                reaction to a list of expressions
        """
        used_model_types = set((wc_lang.Function,
                                wc_lang.Observable,
                                wc_lang.RateLaw,
                                wc_lang.Species,
                                wc_lang.StopCondition,
                                wc_lang.Compartment))

        model_entities = itertools.chain(model.functions,
                                         model.observables,
                                         model.rate_laws,
                                         model.stop_conditions)

        # 1) make digraph of dependencies among model instances
        dependency_graph = networkx.DiGraph()

        def map_model_to_dynamic_model(model):
            return DynamicComponent.get_dynamic_component(type(model), model.id)

        for dependent_model_entity in model_entities:
            dependent_model_entity_expr = dependent_model_entity.expression

            # get all instances of types in used_model_types used by dependent_model_entity
            used_models = []
            for attr_name, attr in dependent_model_entity_expr.Meta.attributes.items():
                if isinstance(attr, obj_tables.RelatedAttribute):
                    if attr.related_class in used_model_types:
                        used_models.extend(getattr(dependent_model_entity_expr, attr_name))

            # add edges from model_type entities to the dependent_model_entity that uses them
            for used_model in used_models:
                dependency_graph.add_edge(map_model_to_dynamic_model(used_model),
                                          map_model_to_dynamic_model(dependent_model_entity))

        # 2) a compartment in an expression is a special case that computes the compartment's mass
        # add dependencies between each compartment used in an expression and all the species
        # in the compartment
        dynamic_compartments_in_use = set()
        for node in dependency_graph.nodes():
            if isinstance(node, DynamicCompartment):
                dynamic_compartments_in_use.add(node)
        for dynamic_compartment in dynamic_compartments_in_use:
                compartment = model.compartments.get(id=dynamic_compartment.id)[0]
                for species in compartment.species:
                    dependency_graph.add_edge(map_model_to_dynamic_model(species),
                                              map_model_to_dynamic_model(compartment))

        # 3) add edges of species altered by reactions to determine dependencies of expressions on
        # reactions
        # to reduce memory use, process one reaction at a time
        reaction_dependencies = {}
        dfs_preorder_nodes = networkx.algorithms.traversal.depth_first_search.dfs_preorder_nodes
        for dynamic_submodel in self.dynamic_submodels.values():
            for reaction in dynamic_submodel.reactions:

                net_stoichiometric_coefficients = collections.defaultdict(float)
                for participant in reaction.participants:
                    species = participant.species
                    net_stoichiometric_coefficients[species] += participant.coefficient
                for species, net_stoich_coeff in net_stoichiometric_coefficients.items():
                    if net_stoich_coeff < 0 or 0 < net_stoich_coeff:
                        dependency_graph.add_edge(reaction,
                                                  map_model_to_dynamic_model(species))

                # 4) traverse from each reaction to dependent expressions
                reaction_dependencies[reaction] = set()
                for node in dfs_preorder_nodes(dependency_graph, reaction):
                    if not isinstance(node, (wc_lang.Reaction, DynamicCompartment, DynamicSpecies)):
                        reaction_dependencies[reaction].add(node)

                # convert the dependency sets into lists, which are more than 3x smaller than sets
                reaction_dependencies[reaction] = list(reaction_dependencies[reaction])
                dependency_graph.remove_node(reaction)

        # 5) remove reactions that have no dependencies to conserve space
        rxns_to_remove = set()
        for rxn_id, dependencies in reaction_dependencies.items():
            if not dependencies:
                rxns_to_remove.add(rxn_id)
        for rxn_id in rxns_to_remove:
            del reaction_dependencies[rxn_id]

        return reaction_dependencies

    def continuous_reaction_dependencies(self):
        """ Get the expressions that depend on species used by reactions modeled by continuous submodels

        Caching uses these dependencies to determine the expressions that should be invalidated when
        species populations change or time advances.

        Returns:
            (:obj:`dict`): map from ids of continuous submodels to lists of expressions
                whose values depend on species with non-zero stoichiometry in reaction(s)
                modeled by the submodel
        """
        rxns_modeled_by_continuous_submodels = {}
        for submodel_id, dynamic_submodel in self.dynamic_submodels.items():
            if isinstance(dynamic_submodel, (wc_sim.submodels.dfba.DfbaSubmodel,
                                             wc_sim.submodels.odes.OdeSubmodel)):
                rxns_modeled_by_continuous_submodels[submodel_id] = set(dynamic_submodel.reactions)

        continuous_rxn_dependencies = {}
        for submodel_id, sm_reactions in rxns_modeled_by_continuous_submodels.items():
            continuous_rxn_dependencies[submodel_id] = set()
            # TODO (APG): OPTIMIZE DFBA CACHING: only include exchange and objective (biomass) reactions in dFBA submodels
            for reaction, dependencies in self.rxn_expression_dependencies.items():
                if reaction in sm_reactions:
                    continuous_rxn_dependencies[submodel_id].update(dependencies)
        # to save space convert values in continuous_rxn_dependencies from sets to lists
        for submodel_id, dependencies in continuous_rxn_dependencies.items():
            continuous_rxn_dependencies[submodel_id] = list(dependencies)
        return continuous_rxn_dependencies

    def prepare_dependencies(self, model):
        """ Initialize expression dependency attributes

        Args:
            model (:obj:`Model`): the description of the whole-cell model in `wc_lang`
        """
        self.rxn_expression_dependencies = self.obtain_dependencies(model)
        self.continuous_rxn_dependencies = self.continuous_reaction_dependencies()
        all_continuous_rxn_dependencies = set()
        for expression_keys in self.continuous_rxn_dependencies.values():
            all_continuous_rxn_dependencies.union(expression_keys)
        self.all_continuous_rxn_dependencies = tuple(all_continuous_rxn_dependencies)

    def _stop_caching(self):
        """ Disable caching; used for testing """
        self.cache_manager._stop_caching()

    def _start_caching(self):
        """ Enable caching; used for testing """
        self.cache_manager._start_caching()

    def flush_compartment_masses(self):
        """ If caching is enabled, invalidate cache entries for compartmental masses

        Run whenever populations change or time advances
        """
        if self.cache_manager.caching():
            # do nothing if the invalidation is EVENT_BASED invalidation, because invalidate() clears the cache
            if self.cache_manager.invalidation_approach() is InvalidationApproaches.REACTION_DEPENDENCY_BASED:
                dynamic_compartments = [dyn_compartment for dyn_compartment in self.dynamic_compartments.values()]
                self.cache_manager.invalidate(expressions=dynamic_compartments)

    def flush_after_reaction(self, reaction):
        """ If caching is enabled, invalidate cache entries when time advances and a reaction executes

        Args:
            reaction (:obj:`Reaction`): the reaction that executed
        """
        expressions = self.all_continuous_rxn_dependencies
        if reaction in self.rxn_expression_dependencies:
            expressions = itertools.chain(expressions, self.rxn_expression_dependencies[reaction])
        self.cache_manager.invalidate(expressions=expressions)
        self.flush_compartment_masses()

    def continuous_submodel_flush_after_populations_change(self, dynamic_submodel_id):
        """ Invalidate cache entries that depend on reactions modeled by a continuous submodel

        Only used when caching is enabled.
        Runs when a continuous submodel advances time or changes species populations.

        Args:
            dynamic_submodel_id (:obj:`str`): the id of the continuous submodel that's running
        """
        self.cache_manager.invalidate(expressions=self.continuous_rxn_dependencies[dynamic_submodel_id])
        self.flush_compartment_masses()


WC_LANG_MODEL_TO_DYNAMIC_MODEL = {
    wc_lang.Compartment: DynamicCompartment,
    wc_lang.DfbaObjective: DynamicDfbaObjective,
    wc_lang.Function: DynamicFunction,
    wc_lang.Observable: DynamicObservable,
    wc_lang.Parameter: DynamicParameter,
    wc_lang.RateLaw: DynamicRateLaw,
    wc_lang.Species: DynamicSpecies,
    wc_lang.StopCondition: DynamicStopCondition,
}


class InvalidationApproaches(Enum):
	REACTION_DEPENDENCY_BASED = auto()
	EVENT_BASED = auto()


class CachingEvents(Enum):
    """ Types of caching events, used to maintain caching statistics
    """
    HIT = auto()
    MISS = auto()
    HIT_RATIO = auto()
    FLUSH_HIT = auto()
    FLUSH_MISS = auto()
    FLUSH_HIT_RATIO = auto()


class CacheManager(object):
    """ Represent a RAM cache of `DynamicExpression.eval()` values

    This is a centralized cache for all `DynamicExpression` values and `DynamicCompartment` `accounted_mass` values.
    Caching may speed up a simulation substantially, or may slow it down.
    All caching is controlled by the multialgorithm configuration file.
    The `expression_caching` attribute determines whether caching is active.
    The `cache_invalidation` attribute selects the cache invalidation approach.

    The `event_based` invalidation approach invalidates (flushes) the entire cache at the start of
    each simulation event which changes species populations, that is, that executes a reaction. Thus,
    all expressions used during the event must be recalculated. This approach will boost performance
    if many expressions are used repeatedly during
    a single event, as occurs when many rate laws that share functions are evaluated.

    The `reaction_dependency_based` invalidation approach invalidates (flushes) individual cache
    entries that depend on the execution of a particular reaction.
    The dependencies of `DynamicExpression`\ s on species populations and the reactions that alter
    the populations are computed at initialization.

    Under the `reaction_dependency_based` approach,
    when a reaction executes all cached values of the `DynamicExpression`\ s that depend on
    the reaction are invalidated.
    This approach will be superior if a typical reaction execution changes populations of species
    that are used, directly or indirectly, by only a small fraction of the cached values of
    the `DynamicExpression`\ s.

    In addition, since the populations of species modeled by continuous integration algorithms,
    such as ODEs and dFBA,
    vary continuously, `DynamicExpression`\ s that depend on them must always be invalidated
    whenever simulation time advances.

    Attributes:
        caching_active (:obj:`bool`): whether caching is active
        cache_invalidation (:obj:`InvalidationApproaches`): the cache invalidation approach
        _cache (:obj:`dict`): cache of `DynamicExpression.eval()` values
        _cache_stats (:obj:`dict`): caching stats
    """
    def __init__(self, caching_active=None, cache_invalidation=None):
        """

        Args:
            caching_active (:obj:`bool`, optional): whether `DynamicExpression` values are cached
            cache_invalidation (:obj:`str`, optional): the cache invalidation approach:
                either reaction_dependency_based or event_based

        Raises:
            :obj:`MultialgorithmError`: if `cache_invalidation` is not `reaction_dependency_based`
            or `event_based`
        """
        config_multialgorithm = wc_sim.config.core.get_config()['wc_sim']['multialgorithm']
        self.caching_active = caching_active
        if caching_active is None:
            self.caching_active = config_multialgorithm['expression_caching']

        if self.caching_active:
            if cache_invalidation is None:
                cache_invalidation = config_multialgorithm['cache_invalidation']
            cache_invalidation = cache_invalidation.upper()
            if cache_invalidation not in InvalidationApproaches.__members__:
                raise MultialgorithmError(f"cache_invalidation '{cache_invalidation}' not in "
                                          f"{str(set(InvalidationApproaches.__members__))}")
            self.cache_invalidation = InvalidationApproaches[cache_invalidation]

        self._cache = dict()
        self._cache_stats = dict()
        for cls in itertools.chain(DynamicExpression.__subclasses__(), (DynamicCompartment,)):
            self._cache_stats[cls.__name__] = dict()
            for caching_event in list(CachingEvents):
                self._cache_stats[cls.__name__][caching_event] = 0

    def invalidation_approach(self):
        """ Provide the invalidation approach

        Returns:
            :obj:`InvalidationApproaches`: the invalidation approach
        """
        return self.cache_invalidation

    def get(self, expression):
        """ If caching is enabled, get the value of `expression` from the cache if it's stored

        Also maintain caching statistics.

        Args:
            expression (:obj:`object`): a dynamic expression

        Returns:
            :obj:`object`: the cached value of `expression`

        Raises:
            :obj:`MultialgorithmError`: if the cache does not contain an entry for `expression`
        """
        if self.caching_active:
            cls_name = expression.__class__.__name__
            if expression in self._cache:
                self._cache_stats[cls_name][CachingEvents.HIT] += 1
                return self._cache[expression]
            else:
                self._cache_stats[cls_name][CachingEvents.MISS] += 1
                raise MultialgorithmError(f"dynamic expression ({cls_name}.{expression.id}) "
                                          f"not in cache")

    def set(self, expression, value):
        """ If caching is enabled, set a value for `expression` in the cache

        Args:
            expression (:obj:`object`): a dynamic expression
            value (:obj:`object`): value of the dynamic expression
        """
        if self.caching_active:
            self._cache[expression] = value

    def flush(self, expressions):
        """ Invalidate the cache entries for all dynamic expressions in `expressions`

        Missing cache entries are ignored.

        Args:
            expressions (:obj:`list` of :obj:`obj`): iterator over dynamic expression instances
        """
        for expression in expressions:
            cls_name = expression.__class__.__name__
            try:
                del self._cache[expression]
                self._cache_stats[cls_name][CachingEvents.FLUSH_HIT] += 1
            except KeyError:
                self._cache_stats[cls_name][CachingEvents.FLUSH_MISS] += 1

    def clear_cache(self):
        """ Remove all cache entries """
        self._cache.clear()

    def invalidate(self, expressions=None):
        """ Invalidate the cache entries for all dynamic expressions in `expressions`

        Missing cache entries are ignored. Does nothing if caching is not enabled.

        Args:
            expressions (:obj:`set` of :obj:`obj`): iterator over dynamic expression instances
        """
        if self.caching_active: # coverage's claim that 'self.caching_active' is never False is wrong
            if self.cache_invalidation == InvalidationApproaches.REACTION_DEPENDENCY_BASED:
                expressions = tuple() if expressions is None else expressions
                self.flush(expressions)
            elif self.cache_invalidation == InvalidationApproaches.EVENT_BASED:
                self.clear_cache()
            else:
                pass    # pragma: no cover

    def caching(self):
        """ Is caching enabled?

        Returns:
            :obj:`bool`: return `True` if caching is enabled
        """
        return self.caching_active

    def set_caching(self, caching_active):
        """ Set the state of caching

        Args:
            caching_active (:obj:`bool`): `True` if caching should be enabled, otherwise `False`
        """
        self.caching_active = caching_active

    def __contains__(self, expression):
        """ Indicate whether the cache contains an expression

        Args:
            expression (:obj:`object`): a dynamic expression

        Returns:
            :obj:`bool`: return :obj:`True` if caching is enabled and `expression` is in the cache,
                :obj:`False` otherwise
        """
        return self.caching_active and expression in self._cache

    def size(self):
        """ Get the cache's size

        Returns:
            :obj:`int`: the number of entries in the cache
        """
        return len(self._cache)

    def empty(self):
        """ Determine whether the cache is empty

        Returns:
            :obj:`bool`: return :obj:`True` if the cache is empty, :obj:`False` otherwise
        """
        return self.size() == 0

    def _stop_caching(self):
        """ Disable caching; used for testing """
        self.set_caching(False)

    def _start_caching(self):
        """ Enable caching; used for testing """
        self.clear_cache()
        self.set_caching(True)

    def _add_hit_ratios(self):
        """ Add hit ratios to the cache stats dictionary
        """
        for _, stats in self._cache_stats.items():
            c_e = CachingEvents
            try:
                stats[c_e.HIT_RATIO] = stats[c_e.HIT] / (stats[c_e.HIT] + stats[c_e.MISS])
            except ZeroDivisionError:
                stats[c_e.HIT_RATIO] = float('nan')

            try:
                stats[c_e.FLUSH_HIT_RATIO] = \
                    stats[c_e.FLUSH_HIT] / (stats[c_e.FLUSH_HIT] + stats[c_e.FLUSH_MISS])
            except ZeroDivisionError:
                stats[c_e.FLUSH_HIT_RATIO] = float('nan')

    def cache_stats_table(self):
        """ Provide the caching stats

        Returns:
            :obj:`str`: the caching stats in a table
        """
        self._add_hit_ratios()
        rv = ['Class\t' + '\t'.join(caching_event.name for caching_event in list(CachingEvents))]
        for expression, stats in self._cache_stats.items():
            row = [expression]
            for caching_event in CachingEvents:
                val = stats[caching_event]
                if isinstance(val, float):
                    row.append(f'{val:.2f}')
                else:
                    row.append(str(val))
            rv.append('\t'.join(row))
        return '\n'.join(rv)

    def __str__(self):
        """ Readable cache state

        Returns:
            :obj:`str`: the caching stats in a table
        """
        rv = [f"caching_active: {self.caching_active}"]
        if self.caching_active:
            rv.append(f"cache_invalidation: {self.cache_invalidation}")
        rv.append('cache:')
        rv.append(pformat(self._cache))
        return '\n'.join(rv)
