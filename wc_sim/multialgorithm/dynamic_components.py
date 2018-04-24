""" Dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-02-07
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
import numpy as np

from obj_model import utils
from wc_lang.core import Species, SpeciesType, Compartment
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError


class DynamicCompartment(object):
    """ A dynamic compartment

    A `DynamicCompartment` tracks the dynamic aggregate state of a compartment, primarily its
    mass and volume. A `DynamicCompartment` is created for each `wc_lang` `Compartment`.

    Attributes:
        id (:obj:`str`): id of this `DynamicCompartment`, copied from the corresponding `Compartment`
        name (:obj:`str`): name of this `DynamicCompartment`, copied from the corresponding `Compartment`
        init_volume (:obj:`float`): initial volume specified in the `wc_lang` model
        species_population (:obj:`LocalSpeciesPopulation`): an object that represents
            the populations of species in this `DynamicCompartment`
        species_ids (:obj:`list` of `str`): the IDs of the species stored
            in this dynamic compartment; if `None`, use the IDs of all species in `species_population`
    """
    def __init__(self, compartment, species_population, species_ids=None):
        """ Initialize this `DynamicCompartment`

        Args:
            compartment (:obj:`Compartment`): the corresponding `wc_lang` `Compartment`
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
        if self.init_volume<=0:
            raise MultialgorithmError("DynamicCompartment: init_volume must be a positive number, "
                "but it is '{}'".format(self.init_volume))
        self.constant_density = self.mass()/self.init_volume

    def mass(self):
        """ Provide the total current mass of all species in this `DynamicCompartment`

        Returns:
            :obj:`float`: this compartment's total current mass (g)
        """
        return self.species_population.mass(species_ids=self.species_ids)

    def volume(self):
        """ Provide the current volume of this `DynamicCompartment`

        This compartment's density is assumed to be constant

        Returns:
            :obj:`float`: this compartment's current volume (L)
        """
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


# TODO(Arthur): figure out what goes in a DynamicModel
class DynamicModel(object):
    """ Represent the aggregate dynamics of a whole-cell model simulation

    A `DynamicModel` determines aggregate properties that are not provided
    by other, more specific, dynamical components like species populations, submodels, and
    dynamic compartments.

    Attributes:
        model (:obj:`Model`): the description of the whole-cell model in `wc_lang`
        # TODO(Arthur): initialize cellular_compartments
        cellular_compartments (:obj:`list` of `DynamicCompartment`): the cell's
            cellular compartments
        fraction_dry_weight (:obj:`float`): fraction of the cell's weight which is not water
            a constant
        water_in_model (:obj:`bool`): if set, the model represents water
        growth (:obj:`float`): growth in cell/s, relative to the cell's initial volume
    """

    def __init__(self, model):
        self.model = model

    def initialize(self):
        """ Prepare a `DynamicModel` for a discrete-event simulation
        """
        # Classify compartments into extracellular and cellular
        # Compartments which are not extracellular are cellular
        # Assumes exactly one extracellular compartment
        extracellular_compartment = utils.get_component_by_id(self.model.get_compartments(),
            EXTRACELLULAR_COMPARTMENT_ID)

        cellular_compartments = []
        for compartment in self.model.get_compartments():
            if compartment.id == EXTRACELLULAR_COMPARTMENT_ID:
                continue
            cellular_compartments.append(compartment)

        # Does the model represent water?
        self.water_in_model = True
        for compartment in self.model.get_compartments():
            water_in_compartment_id = Species.gen_id(WATER_ID, compartment.id)
            if water_in_compartment_id not in [s.id() for s in compartment.species]:
                self.water_in_model = False
                break

        # cell dry weight
        self.fraction_dry_weight = utils.get_component_by_id(self.model.get_parameters(),
            'fractionDryWeight').value

        # growth
        self.growth = np.nan

    def cell_mass(self):
        """ Compute the cell's mass

        Sum the mass of all `DynamicCompartment`s that are not extracellular.
        Assumes compartment volumes are in L and concentrations in mol/L.

        Returns:
            :obj:`float`: the cell's mass (g)
        """
        return sum([dynamic_compartment.mass() for dynamic_compartment in self.cellular_compartments])

    def cell_dry_weight(self):
        """ Compute the cell's dry weight

        Returns:
            :obj:`float`: the cell's dry weight (g)
        """
        if self.water_in_model:
            return self.fraction_dry_weight * self.cell_mass()
        else:
            return self.cell_mass()

    def get_species_count_array(self, now):
        """ Map current species counts into an np array

        Args:
            now (:obj:`float`): the current simulation time

        Returns:
            numpy array, #species x # compartments, containing count of specie in compartment
        """
        # TODO(Arthur): avoid wastefully converting between dictionary and array representations of copy numbers
        species_counts = np.zeros((len(model.species), len(model.compartments)))
        for species in model.species:
            for compartment in model.compartments:
                specie_id = Species.gen_id(species, compartment)
                species_counts[ species.index, compartment.index ] = \
                    model.local_species_population.read_one(now, specie_id)
        return species_counts
