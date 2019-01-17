""" Dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-02-07
:Copyright: 2017-2019, Karr Lab
:License: MIT
"""

import math
import numpy
import warnings
import wc_lang

from obj_model import utils
from wc_lang import Species, Compartment
from wc_sim.multialgorithm.dynamic_expressions import (DynamicComponent,
                                                       DynamicSpecies, DynamicObservable,
                                                       DynamicFunction,
                                                       DynamicRateLaw, DynamicDfbaObjective,
                                                       DynamicStopCondition, DynamicParameter)
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_utils.util.ontology import wcm_ontology


class DynamicCompartment(DynamicComponent):
    """ A dynamic compartment

    A `DynamicCompartment` tracks the dynamic aggregate state of a compartment, primarily its
    mass. A `DynamicCompartment` is created for each `wc_lang` `Compartment` in a whole-cell
    model.

    Attributes:
        id (:obj:`str`): id of this `DynamicCompartment`, copied from `compartment`
        name (:obj:`str`): name of this `DynamicCompartment`, copied from `compartment`
        init_volume (:obj:`float`): initial volume specified in the `wc_lang` model
        init_mass (:obj:`float`): initial mass
        species_population (:obj:`LocalSpeciesPopulation`): an object that represents
            the populations of species in this `DynamicCompartment`
        species_ids (:obj:`list` of :obj:`str`): the IDs of the species stored
            in this dynamic compartment; if `None`, use the IDs of all species in `species_population`
    """

    def __init__(self, dynamic_model, species_population, wc_lang_model, species_ids=None):
        """ Initialize this `DynamicCompartment`

        Args:
            dynamic_model
            species_population (:obj:`LocalSpeciesPopulation`): an object that represents
                the populations of species in this `DynamicCompartment`
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

        if wc_lang_model.distribution_init_volume == wcm_ontology['WCM:normal_distribution']:
            mean = wc_lang_model.mean_init_volume
            std = wc_lang_model.std_init_volume
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

    def mass(self):
        """ Provide the total current mass of all species in this `DynamicCompartment`

        Returns:
            :obj:`float`: this compartment's total current mass (g)
        """
        return self.species_population.compartmental_mass(self.id)

    def __str__(self):
        """ Provide a string representation of this `DynamicCompartment`

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
        dynamic_compartments (:obj:`dict`): map from compartment ID to `DynamicCompartment`; the simulation's
            `DynamicCompartment`s, one for each compartment in `model`
        cellular_dyn_compartments (:obj:`list`): list of the cellular compartments
        species_population (:obj:`LocalSpeciesPopulation`): an object that represents
            the populations of species in this `DynamicCompartment`
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
                the populations of species in this `DynamicCompartment`
            dynamic_compartments (:obj:`dict`): the simulation's `DynamicCompartment`s, one for each
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
        for dynamic_expression_group in [self.dynamic_observables, self.dynamic_functions,
                                         self.dynamic_stop_conditions]:
            for dynamic_expression in dynamic_expression_group.values():
                dynamic_expression.prepare()

    def cell_mass(self):
        """ Compute the cell's mass

        Sum the mass of all `DynamicCompartment`s that are not extracellular.
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

    def get_species_count_array(self, now):     # pragma no cover   not used
        """ Map current species counts into an numpy array

        Args:
            now (:obj:`float`): the current simulation time

        Returns:
            numpy array, #species x # compartments, containing count of specie in compartment
        """
        species_counts = numpy.zeros((len(model.species), len(model.compartments)))
        for species in model.species:
            for compartment in model.compartments:
                species_id = Species.gen_id_static(species.id, compartment.id)
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
