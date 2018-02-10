""" Dynamic components of a multialgorithm simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-02-07
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
import numpy as np

from obj_model import utils
from wc_lang.core import Species, SpeciesType


class DynamicCompartment(object):
    """ A dynamic compartment

    A `DynamicCompartment` tracks the dynamic aggregate state of a compartment, primarily its
    mass and volume. A `DynamicCompartment` is created for each `wc_lang` `Compartment`.

    Attributes:
        id (:obj:`str`): unique id of the corresponding `wc_lang` `Compartment`
        name (:obj:`str`): name of the corresponding `wc_lang` `Compartment`
        init_volume (:obj:`float`): initial volume specified in the `wc_lang` model
        species_populations (:obj:`LocalSpeciesPopulation`): a shared store of the simulation's
            species populations
        species_ids (:obj:`list` of `str`): the IDs of the species stored
            in this compartment; this enables multiple `DynamicCompartment`s to share a
            `LocalSpeciesPopulation`
    """
    def __init__(self, id, name, init_volume, species_populations, species_ids=None):
        self.id = id
        self.name = name
        self.init_volume = init_volume
        self.species_populations = species_populations
        self.species_ids = species_ids
        self.initialize()

    def initialize(self):
        """ Initialize this `DynamicCompartment`
        """
        self.constant_density = self.mass()/self.init_volume

    def mass(self):
        """ Provide the total current mass of all species in this `DynamicCompartment`

        Returns:
            :obj:`float`: this compartment's total current mass (g)
        """
        return self.species_populations.mass(species_ids=self.species_ids)

    def volume(self):
        """ Provide the current volume of this `DynamicCompartment`

        This compartment's density is assumed to be constant

        Returns:
            :obj:`float`: this compartment's current volume (L)
        """
        return self.mass()/self.constant_density

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


EXTRACELLULAR_COMPARTMENT_ID = 'e'
WATER_ID = 'H2O'

class DynamicModel(object):
    """ Represent the aggregate dynamics of a whole-cell model simulation

    A `DynamicModel` provides methods for determining aggregate properties that are not provided
    by other, more specific, dynamical components suchs as species populations, submodels, and
    dynamic compartments.

    # TODO(Arthur): probably only need a model at initialization
    Attributes:
        model (:obj:`Model`): the description of the whole-cell model in `wc_lang`
        multialgorithm_simulation (:obj:`MultialgorithmSimulation`): the multialgorithm simulation
        # DC: remove
        volume (:obj:`float`): volume of the cell's cellular (cytoplasm) compartment
        extracellular_volume (:obj:`float`): volume of the cell's extra-cellular
        fraction_dry_weight (:obj:`float`): fraction of the cell's weight which is not water
            a constant
        dry_weight (:obj:`float`): a cell's dry weight
        density (:obj:`float`): cellular density, a constant
        growth (:obj:`float`): growth in cell/s, relative to the cell's initial volume
    """

    def __init__(self, model):
        self.model = model
        # TODO(Arthur): clarify & fix relationship between multialgorithm_simulation and DynamicModel
        # TODO(Arthur): perhaps remove next statement
        # self.multialgorithm_simulation = multialgorithm_simulation

    def initialize(self):
        """ Prepare a `DynamicModel` for a discrete-event simulation
        """
        # Classify compartments by cellular and extracellular
        # all others: cellular_compartments
        extracellular_compartment = utils.get_component_by_id(self.model.get_compartments(),
            EXTRACELLULAR_COMPARTMENT_ID)
        self.extracellular_volume = extracellular_compartment.initial_volume

        cellular_compartments = []
        for compartment in self.model.get_compartments():
            if compartment.id == EXTRACELLULAR_COMPARTMENT_ID:
                continue
            cellular_compartments.append(compartment)

        # DC: remove
        # volume: sum cellular compartment volumes
        self.volume = sum(
            [cellular_compartment.initial_volume for cellular_compartment in cellular_compartments])

        # does the model represent water?
        water_in_model = True
        for compartment in self.model.get_compartments():
            water_in_compartment_id = Species.gen_id(WATER_ID, compartment.id)
            if water_in_compartment_id not in [s.id() for s in compartment.species]:
                water_in_model = False
                break

        # cell mass
        # TODO(Arthur): next; DC: use DC mass?
        self.mass = self.initial_cell_mass([EXTRACELLULAR_COMPARTMENT_ID])
        self.fraction_dry_weight = utils.get_component_by_id(self.model.get_parameters(),
            'fractionDryWeight').value
        if water_in_model:
            self.dry_weight = self.fraction_dry_weight * self.mass
        else:
            self.dry_weight = self.mass

        # DC: remove
        # density
        self.density = self.mass / self.volume

        # growth
        self.growth = np.nan

    # DC: use DC masses
    def initial_cell_mass(self, extracellular_compartments):
        """ Compute the cell's initial mass from the model

        Sum the mass of all species not stored in an extracellular compartment.
        Assumes compartment volumes are in L and concentrations in mol/L.

        Args:
            extracellular_compartments (:obj:`list`): all extracellular compartments

        Returns:
            :obj:`float`: the cell's initial mass (g)
        """
        # sum over all initial concentrations in intracellular compartments
        total_mw = 0
        for concentration in self.model.get_concentrations():
            if concentration.species.compartment.id in extracellular_compartments:
                continue
            species_type = SpeciesType.objects.get_one(id=concentration.species.species_type.id)
            total_mw += concentration.value * \
                species_type.molecular_weight * \
                concentration.species.compartment.initial_volume
        return total_mw/Avogadro

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
