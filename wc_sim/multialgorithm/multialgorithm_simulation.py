""" Initialize a multialgorithm simulation from a language model and run-time parameters.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-07
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import re
import numpy as np
from scipy.constants import Avogadro
from collections import defaultdict
from six import iteritems, itervalues
from math import ceil, floor, exp, log, log10, isnan
import tokenize, token

from obj_model import utils
from wc_utils.util.list import difference
from wc_lang.core import (SubmodelAlgorithm, Model, ObjectiveFunction, SpeciesType, SpeciesTypeType,
    Species, Compartment, Reaction, ReactionParticipant, RateLawEquation, BiomassReaction)
from wc_lang.core import Submodel as LangSubmodel
from wc_lang.rate_law_utils import RateLawUtils

from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation, AccessSpeciesPopulations
from wc_sim.multialgorithm.submodels.submodel import Submodel as SimSubmodel
from wc_sim.multialgorithm.submodels.ssa import SsaSubmodel
from wc_sim.multialgorithm.submodels.fba import FbaSubmodel
from wc_sim.multialgorithm.species_populations import LOCAL_POP_STORE, Specie, SpeciesPopSimObject

from wc_utils.config.core import ConfigManager
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
config_multialgorithm = \
    ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']

# TODO(Arthur): use lists instead of sets to ensure deterministic behavior

EXTRACELLULAR_COMPARTMENT_ID = 'e'
WATER_ID = 'H2O'

"""
Design notes:

Inputs:

    * Static model in a wc_lang.core.Model
    * Command line parameters, including:

        * Num shared cell state objects

    * Optionally, extra config

Output:
    
    * Simulation partitioned into submodels and cell state, including:

        * Initialized submodel and state simulation objects
        * Initial simulation messages

DS:
    * language model
    * simulation objects
    * an initialized simulation

Algs:

    * Partition into Submodels and Cell State:
    
        #. Determine shared & private species
        #. Determine partition
        #. Create shared species object(s)
        #. Create submodels that contain private species and access shared species
        #. Have SimulationObjects send their initial messages
"""

"""
Density remains constant
Each submodel should compute:
    mass = Sum(counts * species_masses)
    volume = mass/density
    concentration = counts/volume
    counts = volume*concentration (stochastically rounded)

SimSubmodels may not share memory with each other.
What does a SimSubmodel need?:
    Part of a static WCmodel, with rate laws (but only part of it; could copy it and remove unnecessary parts):
        Write a filter, that removes specified parts of a Model.
        The subsets of these data that are involved in reactions modeled by the SimSubmodel:
            Species_types: id, molecular weight, empirical_formula
            Species: id, population, etc.
            Compartments: id,
            Reactions: participants, reversible
            RateLaw: reaction, direction, equation, etc.
            RateLawEquation: transcoded, modifiers
    Local attributes:
        id, density, local populations, reference(s) to shared population(s),
        network addresses of communicating SimObjects

Each SimulationObject must run either 1) on another processor, or 2) in another thread and not
share memory. How does ROSS handle various levels of parallelism -- multiple SimObjs in one thread;
multiple SimObjs on different cores or processors?

Direct exchange of species count changes for shared species through a shared membrane vs. exchange of
species copy number changes through shared population.
"""

class DynamicModel(object):
    """ Represent the aggregate dynamic state of a whole-cell model simulation

    The primary state of a model being simulated is its species counts, which each submodel accesses
    through its `AccessSpeciesPopulations`. A `DynamicModel` provides methods for
    determining aggregate properties, such as model and compartment volume and mass.

    Attributes:
        model (:obj:`Model`): the description of the whole-cell model in `wc_lang`
        multialgorithm_simulation (:obj:`MultialgorithmSimulation`): the multialgorithm simulation
        volume (:obj:`float`): volume of the cell's cellular (cytoplasm) compartment
        extracellular_volume (:obj:`float`): volume of the cell's extra-cellular
        fraction_dry_weight (:obj:`float`): fraction of the cell's weight which is not water
            a constant
        dry_weight (:obj:`float`): a cell's dry weight
        density (:obj:`float`): cellular density, a constant
        growth (:obj:`float`): growth in cell/s, relative to the cell's initial volume
    """

    def __init__(self, model, multialgorithm_simulation):
        self.model = model
        # TODO(Arthur): clarify & fix relationship between multialgorithm_simulation and DynamicModel
        # TODO(Arthur): perhaps remove next statement
        # self.multialgorithm_simulation = multialgorithm_simulation

    def initialize(self):
        """ Prepare a `DynamicModel` for a discrete-event simulation
        """
        # handle multiple cellular compartments
        # all others: cellular_compartments
        extracellular_compartment = utils.get_component_by_id(self.model.get_compartments(),
            EXTRACELLULAR_COMPARTMENT_ID)
        self.extracellular_volume = extracellular_compartment.initial_volume

        cellular_compartments = []
        for compartment in self.model.get_compartments():
            if compartment.id == EXTRACELLULAR_COMPARTMENT_ID:
                continue
            cellular_compartments.append(compartment)

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
        self.mass = self.initial_cell_mass([EXTRACELLULAR_COMPARTMENT_ID])
        self.fraction_dry_weight = utils.get_component_by_id(self.model.get_parameters(),
            'fractionDryWeight').value
        if water_in_model:
            self.dry_weight = self.fraction_dry_weight * self.mass
        else:
            self.dry_weight = self.mass

        # density
        self.density = self.mass / self.volume

        # growth
        self.growth = np.nan

    def initial_cell_mass(self, extracellular_compartments):
        """ Compute the cell's initial mass from the model

        Sum the mass of all species not stored in an extracellular compartment.
        Assumes compartment volumes are in L and concentrations in mol/L.

        Args:
            extracellular_compartments (:obj:`list`): all extracellular compartments

        Returns:
            :obj:`float`: the cell's initial mass (g)
        """
        # sum over all initial concentrations
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
        """Map current species counts into an np array.

        Args:
            now (float): the current simulation time

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


class MultialgorithmSimulation(object):
    """A multi-algorithmic simulation

    Simulate a model described by a `wc_lang.core` `Model`, using the `wc_sim.multialgorithm`
    simulator.

    Attributes:
        model (:obj:`Model`): a model description
        args (:obj:`Namespace`): parameters for the simulation
        init_populations (:obj: dict from species id to population): the initial populations of
            species, as specified by `model`
        simulation (:obj: `SimulationEngine`): the initialized simulation
        simulation_submodels (:obj: list of `SimSubmodel`): the simulation's submodels
        species_pop_objs (:obj: dict of id mapping to `SpeciesPopSimObject`): shared species
            populations stored in `SimulationObject`'s
        shared_specie_store_name (:obj:`str`): the name for the shared specie store
        dynamic_model (:obj: `DynamicModel`): the dynamic state of a model; aggregate
            state not available in `Model` or a simulation's `SimulationObject`'s
        private_species (:obj: `dict` of `set`): map from submodel to a set of the species
                modeled by only the submodel
        shared_species (:obj: `set`): the shared species
    """

    def __init__(self, model, args, shared_specie_store_name='shared_specie_store'):
        self.model = model
        self.args = args
        self.init_populations = {}
        self.simulation = SimulationEngine()
        self.simulation_submodels = {}
        self.species_pop_objs = {}
        self.shared_specie_store_name = shared_specie_store_name
        self.dynamic_model = DynamicModel(model, self)

    def initialize(self):
        """Initialize a simulation.
        """
        self.initialize_biological_state()
        (self.private_species, self.shared_species) = self.partition_species()
        RateLawUtils.transcode_rate_laws(self.model)
        self.species_pop_objs = self.create_shared_species_pop_objs()

    def build_simulation(self):
        """Build a simulation that has been initialized.

        Returns:
            `SimulationEngine`: an initialized simulation
        """
        self.sub_models = self.create_submodels()
        self.initialize_simulation()
        return self.simulation

    def initialize_biological_state(self):
        """Initialize the biological state of the simulation
        """
        # initialize species populations
        for species in self.model.get_species():
            if species.concentration is None:
                # TODO(Arthur): just init species with concentrations; have updates to count or
                # concentration of unknown species add them
                # make species.id() an alias for species.serialize()
                self.init_populations[species.serialize()] = 0
            else:
                # TODO(Arthur): confirm that truncating to int is OK here
                self.init_populations[species.serialize()] = \
                    int(species.concentration.value * species.compartment.initial_volume * Avogadro)

    def molecular_weights_for_species(self, species):
        """Obtain the molecular weights for specified species ids

        Args:
            species (:obj:`set`): a `set` of species ids

        Returns:
            `dict`: species_type_id -> molecular weight
        """
        specie_weights = {}
        for specie_id in species:
            (specie_type_id, _) = ModelUtilities.parse_specie_id(specie_id)
            specie_weights[specie_id] = SpeciesType.objects.get_one(id=specie_type_id).molecular_weight
        return specie_weights

    def partition_species(self):
        """Statically partition a `Model`'s `Species` into private species and shared species.

        Returns:
            (dict, set): tuple containing a dict mapping submodels to their private species, and a
                set of shared species
        """
        return (ModelUtilities.find_private_species(self.model, return_ids=True),
            ModelUtilities.find_shared_species(self.model, return_ids=True))

    def create_shared_species_pop_objs(self):
        """Create the shared species object.

        # TODO(Arthur): generalize to multiple `SpeciesPopSimObject` objects

        Returns:
            dict: `dict` mapping id to `SpeciesPopSimObject` objects for the simulation
        """
        species_pop_sim_obj = SpeciesPopSimObject(self.shared_specie_store_name,
            {specie_id:self.init_populations[specie_id] for specie_id in self.shared_species},
            molecular_weights=self.molecular_weights_for_species(self.shared_species))
        self.simulation.add_object(species_pop_sim_obj)
        return {self.shared_specie_store_name:species_pop_sim_obj}

    def create_access_species_pop(self, lang_submodel):
        """Create submodels that contain private species and access shared species

        Args:
            lang_submodel (:obj:`Submodel`): description of a submodel

        Returns:
            (:obj:`AccessSpeciesPopulations`): an `AccessSpeciesPopulations` for the `lang_submodel`
        """
        # make LocalSpeciesPopulations & molecular weights
        initial_population = {specie_id:self.init_populations[specie_id]
            for specie_id in self.private_species[lang_submodel.id]}
        molecular_weights = self.molecular_weights_for_species(self.private_species[lang_submodel.id])

        # DFBA submodels need initial fluxes
        if lang_submodel.algorithm == SubmodelAlgorithm.dfba:
            initial_fluxes = {specie_id:0 for specie_id in self.private_species[lang_submodel.id]}
        else:
            initial_fluxes = None
        local_species_population = LocalSpeciesPopulation(self.model,
            lang_submodel.id.replace('_', '_lsp_'),
            initial_population,
            molecular_weights,
            initial_fluxes=initial_fluxes)

        # make AccessSpeciesPopulations object
        access_species_population = AccessSpeciesPopulations(local_species_population,
            self.species_pop_objs)

        # configure species locations in the access_species_population
        access_species_population.add_species_locations(LOCAL_POP_STORE,
            self.private_species[lang_submodel.id])
        access_species_population.add_species_locations(self.shared_specie_store_name,
            self.shared_species)
        return access_species_population

    def create_submodels(self):
        """Create submodels that contain private species and access shared species

        Returns:
            dict mapping submodel.id to `SimSubmodel`: the simulation's submodels
        """
        # make submodels and their parts
        simulation_submodels = {}
        for lang_submodel in self.model.get_submodels():

            access_species_population = self.create_access_species_pop(lang_submodel)

            # make the simulation's submodels
            # todo: add a DynamicModel to each lang_submodel
            if lang_submodel.algorithm == SubmodelAlgorithm.dfba:

                # todo: use a more general method for finding the extracellular & cytoplasm compartments
                create_dfba_submodel_exchange_rxns(lang_submodel,
                    self.model.get_component('compartment', 'e'),
                    self.model.get_component('compartment', 'c'))
                simulation_submodels[lang_submodel.id] = FbaSubmodel(self.model,
                    lang_submodel.id,
                    access_species_population,
                    list(lang_submodel.reactions),
                    lang_submodel.get_species(),
                    lang_submodel.parameters,
                    self.args.FBA_time_step)

            elif lang_submodel.algorithm == SubmodelAlgorithm.ssa:
                simulation_submodels[lang_submodel.id] = SsaSubmodel(self.model,
                    lang_submodel.id,
                    access_species_population,
                    list(lang_submodel.reactions),
                    lang_submodel.get_species(),
                    lang_submodel.parameters)
            elif lang_submodel.algorithm == SubmodelAlgorithm.ode:
                # TODO(Arthur): incorporate an ODE lang_submodel; perhaps the one Eric & Catherine wrote
                raise ValueError("Need ODE implementation")
            else:
                raise ValueError("Unsupported lang_submodel algorithm '{}'".format(lang_submodel.algorithm))

            # connect the AccessSpeciesPopulations object to its affiliated Submodel
            access_species_population.set_submodel(simulation_submodels[lang_submodel.id])

            # add the submodel to the simulation
            self.simulation.add_object(simulation_submodels[lang_submodel.id])

        return simulation_submodels

    def initialize_simulation(self):
        all_object_types = set()
        all_object_types.add(SpeciesPopSimObject)
        for sub_model in self.sub_models.values():
            all_object_types.add(sub_model.__class__)
        print('self.simulation.register_object_types(all_object_types)', all_object_types)
        self.simulation.register_object_types(all_object_types)

        # have each simulation object send its initial event messages
        self.simulation.initialize()
