""" Initialize a multialgorithm simulation from a language model and run-time parameters.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-02-07
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from wc_lang import SubmodelAlgorithm, Model, Species
from wc_sim.multialgorithm.dynamic_components import DynamicModel, DynamicCompartment
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.multialgorithm_checkpointing import MultialgorithmicCheckpointingSimObj
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.species_populations import (LocalSpeciesPopulation, AccessSpeciesPopulations,
                                                       LOCAL_POP_STORE, SpeciesPopSimObject)
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm.submodels.fba import DfbaSubmodel
from wc_sim.multialgorithm.submodels.ssa import SsaSubmodel
from wc_utils.util.list import det_dedupe
from wc_utils.util.misc import obj_to_str

# TODO(Arthur): use lists instead of sets to ensure deterministic behavior
# TODO(Arthur): add logging
# TODO(Arthur): make get_instance_attrs(base_models, attr) which returns the attr value (where attr
# can be a method) for a collection of base_models would be handy; here we would call
# get_instance_attrs(get_instance_attrs(species, 'species_type'), 'molecular_weight')
# class ModelList(Model, list) could implement this so they could chain:
# species_list.get_instance_attrs('species_type').get_instance_attrs('molecular_weight')


"""
Design notes:

Inputs:

    * Static model in a wc_lang.Model
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

SimSubmodels only share memory with each other through read-only objects and a species population object.
What does a DynamicSubmodel need?:
    Part of a static WCmodel, with rate laws (but only part of it; could copy it and remove unnecessary parts):
        Write a filter, that removes specified parts of a Model.
        The subsets of these data that are involved in reactions modeled by the DynamicSubmodel:
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


# TODO (Arthur): put in config file
DEFAULT_VALUES = dict(
    shared_species_store='SHARED_SPECIE_STORE',
    checkpointing_sim_obj='CHECKPOINTING_SIM_OBJ'
)


class MultialgorithmSimulation(object):
    """ Initialize a multialgorithm simulation from a language model and run-time parameters

    Create a simulation from a model described by a `wc_lang` `Model`.

    Attributes:
        model (:obj:`Model`): a model description
        args (:obj:`dict`): parameters for the simulation; if results_dir is provided, then also
            must include checkpoint_period
        init_populations (:obj:`dict` from species id to population): the initial populations of
            species, as specified by `model`
        simulation (:obj:`SimulationEngine`): the initialized simulation
        simulation_submodels (:obj:`list` of :obj:`DynamicSubmodel`): the simulation's submodels
        checkpointing_sim_obj (:obj:`MultialgorithmicCheckpointingSimObj`): the checkpointing object;
            `None` if absent
        species_pop_objs (:obj:`dict` of `SpeciesPopSimObject`): shared species
            populations stored in `SimulationObject`'s
        shared_species_store_name (:obj:`str`): the name for the shared specie store
        dynamic_model (:obj:`DynamicModel`): the dynamic state of a model
        private_species (:obj:`dict` of `set`): map from `DynamicSubmodel` to a set of the species
                modeled by only the submodel
        shared_species (:obj:`set`): the shared species
        local_species_population (:obj:`LocalSpeciesPopulation`): a shared species population for the
            multialgorithm simulation
        dynamic_compartments (:obj:`dict`): the simulation's `DynamicCompartment`s, one for each
            compartment in `model`
    """

    def __init__(self, model, args, shared_species_store_name=DEFAULT_VALUES['shared_species_store']):
        """
        Args:
            model (:obj:`Model`): the model being simulated
            args (:obj:`dict`): parameters for the simulation
            shared_species_store_name (:obj:`str`, optional): the name of the shared species store
        """
        self.model = model
        self.args = args
        self.init_populations = {}
        self.simulation = SimulationEngine()
        self.simulation_submodels = {}
        self.checkpointing_sim_obj = None
        self.species_pop_objs = {}
        self.shared_species_store_name = shared_species_store_name
        self.local_species_population = self.make_local_species_pop(self.model)
        self.dynamic_compartments = self.create_dynamic_compartments(self.model, self.local_species_population)
        self.dynamic_model = None

    def build_simulation(self):
        """ Build a simulation

        Returns:
            :obj:`tuple` of (`SimulationEngine`, `DynamicModel`): an initialized simulation and its
                dynamic model
        """
        self.partition_species()
        self.dynamic_model = DynamicModel(self.model, self.local_species_population, self.dynamic_compartments)
        for comp in self.dynamic_compartments.values():
            comp.dynamic_model = self.dynamic_model
        self.dynamic_model.set_stop_condition(self.simulation)
        if 'results_dir' in self.args and self.args['results_dir']:
            self.checkpointing_sim_obj = self.create_multialgorithm_checkpointing(
                self.args['results_dir'],
                self.args['checkpoint_period'])
        self.simulation_submodels = self.create_dynamic_submodels()
        return (self.simulation, self.dynamic_model)

    def partition_species(self):
        """ Partition species populations for this model's submodels
        """
        self.init_populations = self.get_initial_species_pop(self.model)
        self.private_species = ModelUtilities.find_private_species(self.model, return_ids=True)
        self.shared_species = ModelUtilities.find_shared_species(self.model, return_ids=True)
        self.species_pop_objs = self.create_shared_species_pop_objs()

    def molecular_weights_for_species(self, species):
        """ Obtain the molecular weights for specified species ids

        Args:
            species (:obj:`set`): a `set` of species ids

        Returns:
            `dict`: species_type_id -> molecular weight
        """
        species_weights = {}
        for species_id in species:
            (species_type_id, _) = ModelUtilities.parse_species_id(species_id)
            species_weights[species_id] = self.model.species_types.get_one(id=species_type_id).molecular_weight
        return species_weights

    def create_shared_species_pop_objs(self):
        """ Create the shared species object.

        Returns:
            dict: `dict` mapping id to `SpeciesPopSimObject` objects for the simulation
        """
        species_pop_sim_obj = SpeciesPopSimObject(self.shared_species_store_name,
                                                  {species_id: self.init_populations[species_id] for species_id in self.shared_species},
                                                  molecular_weights=self.molecular_weights_for_species(self.shared_species))
        self.simulation.add_object(species_pop_sim_obj)
        return {self.shared_species_store_name: species_pop_sim_obj}

    # TODO(Arthur): test after MVP wc_sim done
    def create_access_species_pop(self, lang_submodel):   # pragma: no cover
        """ Create a `LocalSpeciesPopulations` for a submodel and wrap it in an `AccessSpeciesPopulations`

        Args:
            lang_submodel (:obj:`Submodel`): description of a submodel

        Returns:
            :obj:`AccessSpeciesPopulations`: an `AccessSpeciesPopulations` for the `lang_submodel`
        """
        # make LocalSpeciesPopulations & molecular weights
        initial_population = {species_id: self.init_populations[species_id]
                              for species_id in self.private_species[lang_submodel.id]}
        molecular_weights = self.molecular_weights_for_species(self.private_species[lang_submodel.id])

        # DFBA submodels need initial fluxes
        if lang_submodel.algorithm == SubmodelAlgorithm.dfba:
            initial_fluxes = {species_id: 0 for species_id in self.private_species[lang_submodel.id]}
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
        access_species_population.add_species_locations(self.shared_species_store_name,
                                                        self.shared_species)
        return access_species_population

    @staticmethod
    def get_initial_species_pop(model):
        """ Obtain the initial species population

        Args:
            model (:obj:`Model`): a `wc_lang` model

        Returns:
            :obj:`dict`: a map species_id -> population, for all species in `model`
        """
        init_populations = {}
        for species in model.get_species():
            init_populations[species.id] = ModelUtilities.concentration_to_molecules(
                species, species.compartment.mean_init_volume)
        return init_populations

    def get_dynamic_compartments(self, submodel):
        """ Get the `DynamicCompartment`s for Submodel `submodel`

        Args:
            submodel (:obj:`Submodel`): the `wc_lang` submodel being compiled into a `DynamicSubmodel`

        Returns:
            :obj:`dict`: mapping: compartment id -> `DynamicCompartment` for the
                `DynamicCompartment`(s) that a new `DynamicSubmodel` needs
        """
        dynamic_compartments = {}
        for comp in submodel.get_compartments():
            dynamic_compartments[comp.id] = self.dynamic_compartments[comp.id]
        return dynamic_compartments

    @staticmethod
    def create_dynamic_compartments(model, local_species_pop):
        """ Create the `DynamicCompartment`s for this simulation

        Args:
            model (:obj:`Model`): a `wc_lang` model
            local_species_pop (:obj:`LocalSpeciesPopulation`): the store that maintains the
                species population for the `DynamicCompartment`(s)

        Returns:
            :obj:`dict`: mapping: compartment id -> `DynamicCompartment` for the
                `DynamicCompartment`(s) used by this multialgorithmic simulation
        """
        # make DynamicCompartments
        dynamic_compartments = {}
        for compartment in model.get_compartments():
            dynamic_compartments[compartment.id] = DynamicCompartment(None, local_species_pop, compartment)
        return dynamic_compartments

    @staticmethod
    def make_local_species_pop(model, retain_history=True):
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
            (species_type_id, _) = ModelUtilities.parse_species_id(specie.id)
            # TODO(Arthur): make get_one more robust, or do linear search
            molecular_weights[specie.id] = model.species_types.get_one(id=species_type_id).molecular_weight

        # Species used by continuous time submodels (like DFBA and ODE) need initial fluxes
        # which indicate that the species is modeled by a continuous time submodel.
        # TODO(Arthur): support non-zero initial fluxes
        initial_fluxes = {}
        continuous_time_submodels = set([SubmodelAlgorithm.dfba, SubmodelAlgorithm.ode])
        for submodel in model.get_submodels():
            if submodel.algorithm in continuous_time_submodels:
                for specie in submodel.get_species():
                    initial_fluxes[specie.id] = 0.0

        return LocalSpeciesPopulation(
            'LSP_' + model.id,
            initial_population,
            molecular_weights,
            initial_fluxes=initial_fluxes,
            retain_history=retain_history)

    def create_multialgorithm_checkpointing(self, checkpoints_dir, checkpoint_period):
        """ Create a multialgorithm checkpointing object for this simulation

        Args:
            checkpoints_dir (:obj:`str`): the directory in which to save checkpoints
            checkpoint_period (:obj:`float`): interval between checkpoints, in simulated seconds

        Returns:
            :obj:`MultialgorithmicCheckpointingSimObj`: the checkpointing object
        """
        multialgorithm_checkpointing_sim_obj = MultialgorithmicCheckpointingSimObj(
            DEFAULT_VALUES['checkpointing_sim_obj'], checkpoint_period, checkpoints_dir,
            self.local_species_population, self.dynamic_model, self)

        # add the multialgorithm checkpointing object to the simulation
        self.simulation.add_object(multialgorithm_checkpointing_sim_obj)
        return multialgorithm_checkpointing_sim_obj

    def create_dynamic_submodels(self):
        """ Create dynamic submodels that access shared species

        Returns:
            :obj:`dict`: mapping `submodel.id` to `DynamicSubmodel`: the simulation's dynamic submodels

        Raises:
            :obj:`MultialgorithmError`: if a submodel cannot be created
        """

        # make the simulation's submodels
        simulation_submodels = []
        for lang_submodel in self.model.get_submodels():

            if lang_submodel.algorithm == SubmodelAlgorithm.ssa:
                simulation_submodel = SsaSubmodel(
                    lang_submodel.id,
                    self.dynamic_model,
                    list(lang_submodel.reactions),
                    lang_submodel.get_species(),
                    self.get_dynamic_compartments(lang_submodel),
                    self.local_species_population
                )

            elif lang_submodel.algorithm == SubmodelAlgorithm.dfba:
                # TODO(Arthur): make DFBA submodels work
                continue

                simulation_submodel = DfbaSubmodel(
                    lang_submodel.id,
                    self.dynamic_model,
                    list(lang_submodel.reactions),
                    lang_submodel.get_species(),
                    self.get_dynamic_compartments(lang_submodel),
                    self.local_species_population,
                    self.args['fba_time_step']
                )

            elif lang_submodel.algorithm == SubmodelAlgorithm.ode:
                # TODO(Arthur): incorporate an ODE lang_submodel; perhaps the one Eric & Catherine wrote
                raise MultialgorithmError("Need ODE implementation")
            else:
                raise MultialgorithmError("Unsupported lang_submodel algorithm '{}'".format(lang_submodel.algorithm))

            simulation_submodels.append(simulation_submodel)

            # add the submodel to the simulation
            self.simulation.add_object(simulation_submodel)

        return simulation_submodels

    def __str__(self):
        """ Provide a readable representation of this `MultialgorithmSimulation`

        Returns:
            :obj:`str`: a readable representation of this `MultialgorithmSimulation`
        """

        return obj_to_str(self, ['args', 'checkpointing_sim_obj', 'dynamic_compartments', 'dynamic_model',
                                 'init_populations', 'local_species_population', 'model', 'private_species', 'shared_species_store_name',
                                 'shared_species', 'simulation', 'simulation_submodels', 'species_pop_objs'])
