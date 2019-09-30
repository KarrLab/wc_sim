""" Initialize a multialgorithm simulation from a language model and run-time parameters.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-02-07
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from wc_lang import Model, Compartment, Species
from wc_sim.dynamic_components import DynamicModel, DynamicCompartment
from de_sim.simulation_engine import SimulationEngine
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_checkpointing import MultialgorithmicCheckpointingSimObj
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.species_populations import (LocalSpeciesPopulation, AccessSpeciesPopulations,
                                                       LOCAL_POP_STORE, SpeciesPopSimObject)
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.submodels.fba import DfbaSubmodel
from wc_sim.submodels.ssa import SsaSubmodel
from wc_utils.util.list import det_dedupe
from wc_utils.util.misc import obj_to_str
from wc_utils.util.ontology import are_terms_equivalent
from wc_onto import onto
from wc_utils.util.rand import RandomStateManager
import numpy.random

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
    Part of a static WCmodel, with rate laws:
        The subsets of these data that are involved in reactions modeled by the DynamicSubmodel:
            Species_types: id, molecular weight
            Species: id, population, etc.
            Compartments: id, initial volume, 
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
        args (:obj:`dict`): parameters for the simulation; if `results_dir` is provided, then also
            must include checkpoint_period
        shared_species_store_name (:obj:`str`): the name for the shared specie store
        simulation (:obj:`SimulationEngine`): the initialized simulation
        checkpointing_sim_obj (:obj:`MultialgorithmicCheckpointingSimObj`): the checkpointing object;
            `None` if absent
        random_state (:obj:`numpy.random.RandomState`): a random state
        species_pop_objs (:obj:`dict` of `SpeciesPopSimObject`): shared species
            populations used by `SimulationObject`\ 's; not currently used
        init_populations (:obj:`dict` from species id to population): the initial populations of
            species, derived from the specification in `model`
        local_species_population (:obj:`LocalSpeciesPopulation`): a shared species population for the
            multialgorithm simulation
        simulation_submodels (:obj:`list` of :obj:`DynamicSubmodel`): the simulation's submodels
        dynamic_model (:obj:`DynamicModel`): the dynamic state of a model
        private_species (:obj:`dict` of `set`): map from `DynamicSubmodel` to a set of the species
                modeled by only the submodel; not currently used
        shared_species (:obj:`set`): the shared species
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
        # initialize simulation infrastructure
        self.simulation = SimulationEngine()
        self.checkpointing_sim_obj = None
        # todo: check that this is being used correctly
        self.random_state = RandomStateManager.instance()
        self.shared_species_store_name = shared_species_store_name

        # create simulation attributes
        self.model = model
        self.args = args
        self.species_pop_objs = {}
        self.init_populations = {}
        self.local_species_population = self.make_local_species_pop(self.model, self.random_state)
        self.simulation_submodels = {}
        self.dynamic_model = None

        # self.dynamic_compartments = self.create_dynamic_compartments(self.model, self.local_species_population)

    def initialize_components(self, wc_lang_model):
        """ Initialize the biochemical components of a simulation

        Args:
            wc_lang_model (:obj:`Model`): the model being simulated

        Returns:
            :obj:`tuple` of (`SimulationEngine`, `DynamicModel`): 
        """
        '''
        init_volume(compartment) = normal(mu, sigma) # the distribution & its parameters are specified
        init_concentration(species) = normal(mu, sigma) # Lang specifies concentrations
        init_copy_nums(species) = init_volume * init_concentrations * NA
        init_accounted_mass(compartment) =  sum(init_copy_nums * mol_wts)
        init_mass(compartment) =  init_volume * init_density

        Link between initial and dynamical states (this part is still a bit awkward):
        init_accounted_density = init_accounted_mass / init_volume
        constant_density = init_density
        init_accounted_ratio = init_accounted_mass / init_mass
                             = init_accounted_density / init_density

        init_total_ration = init_mass / init_accounted_mass

        Dynamical trajectory:
        accounted_mass = sum(molecule_copy_number * molecule_mol_wt)
        accounted_vol = accounted_mass / init_density

        mass = accounted_mass / init_accounted_ratio
        vol = accounted_vol / init_accounted_ratio
        '''

        # 1. ✔ create DynamicCompartments, initializing the init_volume & init_density in each (create_dynamic_compartments())
        # 2. *** obtain the initial species populations by sampling their specified distributions (get_initial_species_pop())
        # 3. create a shared LocalSpeciesPopulation with all species (make_local_species_pop())
        # 4. todo: initialize with non-zero fluxes
        # 5. finish initializing DynamicCompartments (initialize_mass_and_density(), but with species_population)
        # 6. finish initializing DynamicModel
        # 7. create submodels
        # 8. start simulation

    # todo: split into initialize_components() & initialize_infrastructure; run initialize_infrastructure
    # at end of __init__
    def build_simulation(self):
        """ Build a simulation

        Returns:
            :obj:`tuple` of (`SimulationEngine`, `DynamicModel`): an initialized simulation and its
                dynamic model
        """
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

    # not currently used
    def partition_species(self):
        """ Partition species populations for this model's submodels
        """
        self.init_populations = self.get_initial_species_pop(self.model, self.random_state)
        # self.private_species not currently used
        self.private_species = ModelUtilities.find_private_species(self.model, return_ids=True)
        self.shared_species = ModelUtilities.find_shared_species(self.model, return_ids=True)
        # self.species_pop_objs not currently used
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
            species_type_id, _ = Species.parse_id(species_id)
            species_type = self.model.species_types.get_one(id=species_type_id)
            if species_type.structure:
                species_weights[species_id] = species_type.structure.molecular_weight
            else:
                species_weights[species_id] = float('nan')
        return species_weights

    # not currently used
    def create_shared_species_pop_objs(self):
        """ Create the shared species object.

        Returns:
            dict: `dict` mapping id to `SpeciesPopSimObject` objects for the simulation
        """
        species_pop_sim_obj = SpeciesPopSimObject(
            self.shared_species_store_name,
            {species_id: self.init_populations[species_id] for species_id in self.shared_species},
            molecular_weights=self.molecular_weights_for_species(self.shared_species),
            random_state=self.random_state)
        self.simulation.add_object(species_pop_sim_obj)
        return {self.shared_species_store_name: species_pop_sim_obj}

    # TODO(Arthur): not currently used; test after MVP wc_sim done
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
        if are_terms_equivalent(lang_submodel.framework, onto['WC:dynamic_flux_balance_analysis']):
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
    def get_initial_species_pop(model, random_state):
        """ Obtain the initial species population

        Args:
            model (:obj:`Model`): a `wc_lang` model
            random_state (:obj:`numpy.random.RandomState`): random state

        Returns:
            :obj:`dict`: a map species_id -> population, for all species in `model`
        """
        init_populations = {}
        for species in model.get_species():
            init_populations[species.id] = ModelUtilities.concentration_to_molecules(
                # todo: use actual dynamic compartment volume, not mean expected volume
                species, species.compartment.init_volume.mean, random_state)
        return init_populations

    def get_dynamic_compartments(self, submodel):
        """ Get the :obj:`DynamicCompartment`\ s for Submodel `submodel`

        Args:
            submodel (:obj:`Submodel`): the `wc_lang` submodel being compiled into a `DynamicSubmodel`

        Returns:
            :obj:`dict`: mapping: compartment id -> `DynamicCompartment` for the
                `DynamicCompartment`(s) that a new `DynamicSubmodel` needs
        """
        dynamic_compartments = {}
        for comp in submodel.get_children(kind='submodel', __type=Compartment):
            dynamic_compartments[comp.id] = self.dynamic_compartments[comp.id]
        return dynamic_compartments

    def create_dynamic_compartments(self):
        """ Create the :obj:`DynamicCompartment`\ s for this simulation
        """
        # create DynamicCompartments
        self.dynamic_compartments = {}
        for compartment in self.model.get_compartments():
            self.dynamic_compartments[compartment.id] = DynamicCompartment(None, self.random_state,
                                                                           compartment)

    def initialize_dynamic_compartments(self):
        """ Initialize the :obj:`DynamicCompartment`\ s for this simulation
        """
        # initialize all DynamicCompartments
        for dynamic_compartment in self.dynamic_compartments.values():
            dynamic_compartment.initialize_mass_and_density(self.local_species_population)

    @staticmethod
    def make_local_species_pop(model, random_state, retain_history=True):
        """ Create a `LocalSpeciesPopulation` that contains all the species in a model

        Instantiate a `LocalSpeciesPopulation` as a single, centralized store of a model's population.

        Args:
            model (:obj:`Model`): a `wc_lang` model
            retain_history (:obj:`bool`, optional): whether the `LocalSpeciesPopulation` should
                retain species population history

        Returns:
            :obj:`LocalSpeciesPopulation`: a `LocalSpeciesPopulation` for the model
        """
        initial_population = MultialgorithmSimulation.get_initial_species_pop(model, random_state)

        molecular_weights = {}
        for specie in model.get_species():
            (species_type_id, _) = Species.parse_id(specie.id)
            # TODO(Arthur): make get_one more robust, or do linear search
            species_type = model.species_types.get_one(id=species_type_id)
            if species_type.structure:
                molecular_weights[specie.id] = species_type.structure.molecular_weight
            else:
                molecular_weights[specie.id] = float('nan')

        # Species used by continuous time submodels (like DFBA and ODE) need initial fluxes
        # which indicate that the species is modeled by a continuous time submodel.
        # TODO(Arthur): support non-zero initial fluxes; calculate them with initial runs of dFBA and ODE submodels
        initial_fluxes = {}
        for submodel in model.get_submodels():
            if not are_terms_equivalent(submodel.framework, onto['WC:ordinary_differential_equations']) and \
                    not are_terms_equivalent(submodel.framework, onto['WC:dynamic_flux_balance_analysis']):
                for specie in submodel.get_children(kind='submodel', __type=Species):
                    initial_fluxes[specie.id] = 0.0

        return LocalSpeciesPopulation(
            'LSP_' + model.id,
            initial_population,
            molecular_weights,
            initial_fluxes=initial_fluxes,
            random_state=random_state,
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

            if are_terms_equivalent(lang_submodel.framework, onto['WC:stochastic_simulation_algorithm']):
                simulation_submodel = SsaSubmodel(
                    lang_submodel.id,
                    self.dynamic_model,
                    list(lang_submodel.reactions),
                    lang_submodel.get_children(kind='submodel', __type=Species),
                    self.get_dynamic_compartments(lang_submodel),
                    self.local_species_population
                )

            elif are_terms_equivalent(lang_submodel.framework, onto['WC:dynamic_flux_balance_analysis']):
                # TODO(Arthur): make DFBA submodels work
                continue

                simulation_submodel = DfbaSubmodel(
                    lang_submodel.id,
                    self.dynamic_model,
                    list(lang_submodel.reactions),
                    lang_submodel.get_children(kind='submodel', __type=Species),
                    self.get_dynamic_compartments(lang_submodel),
                    self.local_species_population,
                    self.args['fba_time_step']
                )

            elif are_terms_equivalent(lang_submodel.framework, onto['WC:ordinary_differential_equations']):
                # TODO(Arthur): add ODE submodels from wc_sim fall 2019
                raise MultialgorithmError("Need ODE implementation")
            else:
                raise MultialgorithmError("Unsupported lang_submodel framework '{}'".format(lang_submodel.framework))

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
                                 'init_populations', 'local_species_population', 'model', 'private_species',
                                 'shared_species_store_name', 'shared_species', 'simulation',
                                 'simulation_submodels', 'species_pop_objs'])
