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
from wc_utils.util.list import difference, det_dedupe
from wc_lang.core import (SubmodelAlgorithm, Model, ObjectiveFunction, SpeciesType, SpeciesTypeType,
    Species, Compartment, Reaction, ReactionParticipant, RateLawEquation, BiomassReaction)
from wc_lang.core import Submodel as LangSubmodel
from wc_lang.rate_law_utils import RateLawUtils
from wc_sim.multialgorithm.dynamic_components import DynamicModel, DynamicCompartment
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation, AccessSpeciesPopulations
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm.submodels.ssa import SSASubmodel
from wc_sim.multialgorithm.submodels.fba import FbaSubmodel
from wc_sim.multialgorithm.species_populations import LOCAL_POP_STORE, Specie, SpeciesPopSimObject

from wc_sim.multialgorithm.config import core as config_core_multialgorithm
config_multialgorithm = \
    config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']

# TODO(Arthur): use lists instead of sets to ensure deterministic behavior


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
Each DynamicCompartment can compute:
    constant_density
    mass = Sum(counts * species_masses)
    volume = mass/density
    concentration = counts/volume
    counts = volume*concentration (stochastically rounded)

SimSubmodels only share memory with each other through read-only objects and a species population object.
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
        id, a dynamic compartment, reference to a shared population

Each SimulationObject must run either 1) on another processor, or 2) in another thread and not
share memory. How does ROSS handle various levels of parallelism -- multiple SimObjs in one thread;
multiple SimObjs on different cores or processors?

Direct exchange of species count changes for shared species through a shared membrane vs. exchange of
species copy number changes through shared population.
"""

class MultialgorithmSimulation(object):
    """ A multi-algorithmic simulation

    Simulate a model described by a `wc_lang` `Model`, using the `wc_sim.multialgorithm`
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
        self.dynamic_model = DynamicModel(model)

    def initialize(self):
        """ Initialize a simulation
        """
        self.init_populations = self.get_initial_species_pop(self.model)
        (self.private_species, self.shared_species) = self.partition_species()
        RateLawUtils.transcode_rate_laws(self.model)
        self.species_pop_objs = self.create_shared_species_pop_objs()

    def build_simulation(self):
        """ Build a simulation that has been initialized

        Returns:
            `SimulationEngine`: an initialized simulation
        """
        self.sub_models = self.create_submodels()
        self.initialize_simulation()
        return self.simulation

    @staticmethod
    def get_initial_species_pop(model):
        """ Obtain the initial species population

        Args:
            model (:obj:`Model`): a `wc_lang` model

        Returns:
            :obj:`dict`: a map specie_id -> population, for all species in `model`
        """
        init_populations = {}
        for specie in model.get_species():
            if specie.concentration is None:
                init_populations[specie.id()] = 0
            else:
                # TODO(Arthur): confirm that rounding down is OK here
                init_populations[specie.id()] = \
                    int(specie.concentration.value * specie.compartment.initial_volume * Avogadro)
        return init_populations

    def molecular_weights_for_species(self, species):
        """ Obtain the molecular weights for specified species ids

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
    # TODO(Arthur): make get_instance_attrs(base_models, attr) which returns the attr value (where attr
    # can be a method) for a collection of base_models would be handy; here we would call
    # get_instance_attrs(get_instance_attrs(species, 'species_type'), 'molecular_weight')
    # class ModelList(Model, list) could implement this so they could chain:
    # species_list.get_instance_attrs('species_type').get_instance_attrs('molecular_weight')

    def partition_species(self):
        """ Statically partition a `Model`'s `Species` into private species and shared species

        Returns:
            (dict, set): tuple containing a dict mapping submodels to their private species, and a
                set of shared species
        """
        return (ModelUtilities.find_private_species(self.model, return_ids=True),
            ModelUtilities.find_shared_species(self.model, return_ids=True))

    def create_shared_species_pop_objs(self):
        """ Create the shared species object.

        Returns:
            dict: `dict` mapping id to `SpeciesPopSimObject` objects for the simulation
        """
        species_pop_sim_obj = SpeciesPopSimObject(self.shared_specie_store_name,
            {specie_id:self.init_populations[specie_id] for specie_id in self.shared_species},
            molecular_weights=self.molecular_weights_for_species(self.shared_species))
        self.simulation.add_object(species_pop_sim_obj)
        return {self.shared_specie_store_name:species_pop_sim_obj}

    def create_access_species_pop(self, lang_submodel):
        """ Create submodels that contain private species and access shared species

        Args:
            lang_submodel (:obj:`Submodel`): description of a submodel

        Returns:
            :obj:`AccessSpeciesPopulations`: an `AccessSpeciesPopulations` for the `lang_submodel`
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
        local_species_population = LocalSpeciesPopulation(
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

    @staticmethod
    def create_dynamic_compartments_for_submodel(model, submodel, local_species_pop):
        """ Create the `DynamicCompartment`(s) that a `DynamicSubmodel` will use

        Args:
            model (:obj:`Model`): a `wc_lang` model
            submodel (:obj:`Submodel`): the `wc_lang` submodel being compiled into a `DynamicSubmodel`
            local_species_pop (:obj:`LocalSpeciesPopulation`): the store that maintains the
                species population for the `DynamicCompartment`(s)

        Returns:
            :obj:`dict`: mapping: compartment id -> `DynamicCompartment` for the
                `DynamicCompartment`(s) that a new `DynamicSubmodel` needs
        """
        # TODO(Arthur): make and use a det_set object, that has performance of a set and determinism of a list
        compartments = []
        for rxn in submodel.reactions:
            for participant in rxn.participants:
                compartments.append(participant.species.compartment)
        compartments = det_dedupe(compartments)
        # TODO(Arthur): log this
        # print("submodel {} needs these DynamicCompartment(s): {}".format(submodel.id, [c.id for c in compartments]))

        # make DynamicCompartments
        dynamic_compartments = {}
        for comp in compartments:
            # TODO(Arthur): make this cleaner:
            # a dynamic compartment must have the same id as the corresponding wc_lang compartment
            dynamic_compartments[comp.id] = DynamicCompartment(
                comp.id,
                comp.name,
                comp.initial_volume,
                local_species_pop)
        return dynamic_compartments

    @staticmethod
    def create_local_species_population(model, retain_history=True):
        """ Create a `LocalSpeciesPopulation` that contains all the species in a model

        Instantiate a `LocalSpeciesPopulation` as a single, centralized store of a model's population.

        Args:
            model (:obj:`Model`): a `wc_lang` model
            retain_history (:obj:`bool`, optional): whether the `LocalSpeciesPopulation` should
                retain species population history

        Returns:
            :obj:`LocalSpeciesPopulation`: a `LocalSpeciesPopulation` for the model
        """
        initial_population = MultialgorithmSimulation.get_initial_species_pop(model)

        molecular_weights = {}
        for specie in model.get_species():
            (specie_type_id, _) = ModelUtilities.parse_specie_id(specie.id())
            molecular_weights[specie.id()] = SpeciesType.objects.get_one(id=specie_type_id).molecular_weight

        # Species used by continuous time submodels (like DFBA and ODE) need initial fluxes
        # which indicate that the species is modeled by a continuous time submodel.
        # TODO(Arthur): support non-zero initial fluxes
        initial_fluxes = {}
        continuous_time_submodels = set([SubmodelAlgorithm.dfba, SubmodelAlgorithm.ode])
        for submodel in model.get_submodels():
            if submodel.algorithm in continuous_time_submodels:
                for specie in submodel.get_species():
                    initial_fluxes[specie.id()] = 0.0

        return LocalSpeciesPopulation(
            'LSP_' + model.id,
            initial_population,
            molecular_weights,
            initial_fluxes=initial_fluxes,
            retain_history=retain_history)

    def create_submodels(self):
        """ Create submodels that contain private species and access shared species

        Returns:
            dict mapping submodel.id to `SimSubmodel`: the simulation's submodels
        """
        # make submodels and their parts
        simulation_submodels = {}
        for lang_submodel in self.model.get_submodels():

            access_species_population = self.create_access_species_pop(lang_submodel)

            # make the simulation's submodels
            # TODO (Arthur): wc_lang should distinguish extracellular from intracellular compartments
            # TODO (Arthur): add a DynamicModel to each lang_submodel
            if lang_submodel.algorithm == SubmodelAlgorithm.ssa:
                simulation_submodels[lang_submodel.id] = SSASubmodel(self.model,
                    lang_submodel.id,
                    access_species_population,
                    list(lang_submodel.reactions),
                    lang_submodel.get_species(),
                    lang_submodel.parameters)

            elif lang_submodel.algorithm == SubmodelAlgorithm.dfba:
                continue
                simulation_submodels[lang_submodel.id] = FbaSubmodel(self.model,
                    lang_submodel.id,
                    access_species_population,
                    list(lang_submodel.reactions),
                    lang_submodel.get_species(),
                    lang_submodel.parameters,
                    self.args.FBA_time_step)

            elif lang_submodel.algorithm == SubmodelAlgorithm.ode:
                # TODO(Arthur): incorporate an ODE lang_submodel; perhaps the one Eric & Catherine wrote
                raise ValueError("Need ODE implementation")
            else:
                raise ValueError("Unsupported lang_submodel algorithm '{}'".format(lang_submodel.algorithm))

            # connect the AccessSpeciesPopulations object to its affiliated DynamicSubmodel
            access_species_population.set_submodel(simulation_submodels[lang_submodel.id])

            # add the submodel to the simulation
            self.simulation.add_object(simulation_submodels[lang_submodel.id])

        return simulation_submodels

    def initialize_simulation(self):
        """ Initialize a multialgorithmic simulation
        """
        all_object_types = set()
        all_object_types.add(SpeciesPopSimObject)
        for sub_model in self.sub_models.values():
            all_object_types.add(sub_model.__class__)
        print('self.simulation.register_object_types(all_object_types)', all_object_types)
        self.simulation.register_object_types(all_object_types)

        # have each simulation object send its initial event messages
        self.simulation.initialize()
