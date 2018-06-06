""" Dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-02-07
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
import numpy as np
import warnings
import math

from obj_model import utils
from wc_lang.core import Species, SpeciesType, Compartment
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm.observables import DynamicObservable, DynamicFunction, DynamicStopCondition


class DynamicCompartment(object):
    """ A dynamic compartment

    A `DynamicCompartment` tracks the dynamic aggregate state of a compartment, primarily its
    mass and volume. A `DynamicCompartment` is created for each `wc_lang` `Compartment` in a whole-cell
    model.

    Attributes:
        id (:obj:`str`): id of this `DynamicCompartment`, copied from `compartment`
        name (:obj:`str`): name of this `DynamicCompartment`, copied from `compartment`
        init_volume (:obj:`float`): initial volume specified in the `wc_lang` model
        species_population (:obj:`LocalSpeciesPopulation`): an object that represents
            the populations of species in this `DynamicCompartment`
        species_ids (:obj:`list` of `str`): the IDs of the species stored
            in this dynamic compartment; if `None`, use the IDs of all species in `species_population`
    """
    def __init__(self, compartment, species_population, species_ids=None):
        """ Initialize this `DynamicCompartment`

        Args:
            compartment (:obj:`Compartment`): the corresponding static `wc_lang` `Compartment`
            species_population (:obj:`LocalSpeciesPopulation`): an object that represents
                the populations of species in this `DynamicCompartment`
            species_ids (:obj:`list` of `str`, optional): the IDs of the species stored
                in this compartment; defaults to the IDs of all species in `species_population`

        Raises:
            :obj:`MultialgorithmError`: if `init_volume` is not a positive number
        """
        self.id = compartment.id
        self.name = compartment.name
        self.init_volume = compartment.initial_volume
        self.species_population = species_population
        self.species_ids = species_ids
        if math.isnan(self.init_volume):
            raise MultialgorithmError("DynamicCompartment {}: init_volume is NaN, but must be a positive "
                "number.".format(self.name))
        if self.init_volume<=0:
            raise MultialgorithmError("DynamicCompartment {}: init_volume ({}) must be a positive number.".format(
                self.name, self.init_volume))
        if 0 == self.mass():
            warnings.warn("DynamicCompartment '{}': initial mass is 0, so constant_density is 0, and "
                "volume will remain constant".format(self.name))
        self.constant_density = self.mass()/self.init_volume

    def mass(self):
        """ Provide the total current mass of all species in this `DynamicCompartment`

        Returns:
            :obj:`float`: this compartment's total current mass (g)
        """
        return self.species_population.compartmental_mass(self.id)

    def volume(self):
        """ Provide the current volume of this `DynamicCompartment`

        This compartment's density is assumed to be constant

        Returns:
            :obj:`float`: this compartment's current volume (L)
        """
        if self.constant_density == 0:
            return self.init_volume
        return self.mass()/self.constant_density

    def density(self):
        """ Provide the density of this `DynamicCompartment`, which is assumed to be constant

        Returns:
            :obj:`float`: this compartment's density (g/L)
        """
        return self.constant_density

    def __str__(self):
        """ Provide a string representation of this `DynamicCompartment`

        Returns:
            :obj:`str`: a string representation of this compartment
        """
        values = []
        values.append("ID: " + self.id)
        values.append("Name: " + self.name)
        values.append("Initial volume (L): {}".format(self.init_volume))
        values.append("Constant density (g/L): {}".format(self.constant_density))
        values.append("Current mass (g): {}".format(self.mass()))
        values.append("Current volume (L): {}".format(self.volume()))
        values.append("Fold change volume: {}".format(self.volume()/self.init_volume))
        return "DynamicCompartment:\n{}".format('\n'.join(values))

# TODO(Arthur): define these in config data, which may come from wc_lang
EXTRACELLULAR_COMPARTMENT_ID = 'e'
WATER_ID = 'H2O'


class DynamicModel(object):
    """ Represent the aggregate dynamics of a whole-cell model simulation

    A `DynamicModel` determines aggregate properties that are not provided
    by other, more specific, dynamical components like species populations, submodels, and
    dynamic compartments.

    Attributes:
        dynamic_compartments (:obj: `dict`): map from compartment ID to `DynamicCompartment`; the simulation's
            `DynamicCompartment`s, one for each compartment in `model`
        cellular_dyn_compartments (:obj:`list`): list of the cellular compartments
        dynamic_observables (:obj:`dict` of `DynamicObservable`): the simulation's dynamic observables,
            indexed by their ids
        dynamic_functions (:obj:`dict` of `DynamicFunction`): the simulation's dynamic functions,
            indexed by their ids
        dynamic_stop_conditions (:obj:`dict` of `DynamicStopCondition`): the simulation's stop conditions,
            indexed by their ids
        fraction_dry_weight (:obj:`float`): fraction of the cell's weight which is not water
            a constant
        water_in_model (:obj:`bool`): if set, the model represents water
    """
    def __init__(self, model, dynamic_compartments):
        """ Prepare a `DynamicModel` for a discrete-event simulation

        Args:
            model (:obj:`Model`): the description of the whole-cell model in `wc_lang`
            dynamic_compartments (:obj: `dict`): the simulation's `DynamicCompartment`s, one for each
                compartment in `model`
        """
        self.dynamic_compartments = dynamic_compartments
        self.dynamic_observables = {}
        self.dynamic_functions = {}
        self.dynamic_stop_conditions = {}

        # Classify compartments into extracellular and cellular; those which are not extracellular are cellular
        # Assumes at most one extracellular compartment
        extracellular_compartment = utils.get_component_by_id(model.get_compartments(),
            EXTRACELLULAR_COMPARTMENT_ID)

        self.cellular_dyn_compartments = []
        for dynamic_compartment in dynamic_compartments.values():
            if dynamic_compartment.id == EXTRACELLULAR_COMPARTMENT_ID:
                continue
            self.cellular_dyn_compartments.append( dynamic_compartment)

        # Does the model represent water?
        self.water_in_model = True
        for compartment in model.get_compartments():
            water_in_compartment_id = Species.gen_id(WATER_ID, compartment.id)
            if water_in_compartment_id not in [s.id() for s in compartment.species]:
                self.water_in_model = False
                break

        # cell dry weight
        self.fraction_dry_weight = utils.get_component_by_id(model.get_parameters(),
            'fractionDryWeight').value

    def cell_mass(self):
        """ Compute the cell's mass

        Sum the mass of all `DynamicCompartment`s that are not extracellular.
        Assumes compartment volumes are in L and concentrations in mol/L.

        Returns:
            :obj:`float`: the cell's mass (g)
        """
        # TODO(Arthur): how should water be treated in mass calculations?
        return sum([dynamic_compartment.mass() for dynamic_compartment in self.cellular_dyn_compartments])

    def cell_volume(self):
        """ Compute the cell's volume

        Sum the volume of all `DynamicCompartment`s that are not extracellular.

        Returns:
            :obj:`float`: the cell's volume (L)
        """
        return sum([dynamic_compartment.volume() for dynamic_compartment in self.cellular_dyn_compartments])

    def cell_dry_weight(self):
        """ Compute the cell's dry weight

        Returns:
            :obj:`float`: the cell's dry weight (g)
        """
        if self.water_in_model:
            return self.fraction_dry_weight * self.cell_mass()
        else:
            return self.cell_mass()

    def get_growth(self):
        """ Report the cell's growth in cell/s, relative to the cell's initial volume

        Returns:
            (:obj:`float`): growth in cell/s, relative to the cell's initial volume
        """
        # TODO(Arthur): implement growth
        pass

    def get_aggregate_state(self):
        """ Report the cell's aggregate state

        Returns:
            :obj:`dict`: the cell's aggregate state
        """
        aggregate_state = {
            'cell mass': self.cell_mass(),
            'cell volume': self.cell_volume()
        }

        compartments = {}
        for dynamic_compartment in self.cellular_dyn_compartments:
            compartments[dynamic_compartment.id] = {
                'name': dynamic_compartment.name,
                'mass': dynamic_compartment.mass(),
                'volume': dynamic_compartment.volume(),
            }
        aggregate_state['compartments'] = compartments
        return aggregate_state

    '''
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
    # TODO: complete or toss
    '''

    def eval_dynamic_observables(self, time, observables_to_eval=None):
        """ Evaluate some dynamic observables at time `time`

        Args:
            time (:obj:`float`): the simulation time
            observables_to_eval (:obj:`list` of `str`, optional): if provided, ids of the observables to
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

    def get_species_count_array(self, now):     # pragma no cover   not used
        """ Map current species counts into an np array

        Args:
            now (:obj:`float`): the current simulation time

        Returns:
            numpy array, #species x # compartments, containing count of specie in compartment
        """
        species_counts = np.zeros((len(model.species), len(model.compartments)))
        for species in model.species:
            for compartment in model.compartments:
                specie_id = Species.gen_id(species, compartment)
                species_counts[ species.index, compartment.index ] = \
                    model.local_species_population.read_one(now, specie_id)
        return species_counts
