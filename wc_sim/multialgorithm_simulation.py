""" Initialize a multialgorithm simulation from a language model and run-time parameters

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-02-07
:Copyright: 2016-2019, Karr Lab
:License: MIT
"""

from pprint import pprint
import numpy.random
import warnings

from de_sim.simulation_object import SimObjClassPriority
from wc_lang import Model, Compartment, Species
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.dynamic_components import DynamicModel, DynamicCompartment
from de_sim.simulation_engine import SimulationEngine
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_checkpointing import MultialgorithmicCheckpointingSimObj
from wc_sim.multialgorithm_errors import MultialgorithmError, MultialgorithmWarning
from wc_sim.species_populations import LocalSpeciesPopulation
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.submodels.fba import DfbaSubmodel
from wc_sim.submodels.odes import OdeSubmodel
from wc_sim.submodels.ssa import SsaSubmodel
from wc_sim.submodels.testing.deterministic_simulation_algorithm import DsaSubmodel
from wc_utils.util.list import det_dedupe
from wc_utils.util.misc import obj_to_str
from wc_utils.util.ontology import are_terms_equivalent
from wc_onto import onto
from wc_utils.util.rand import RandomStateManager

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']

# TODO(Arthur): use lists instead of sets to ensure deterministic behavior
# TODO(Arthur): add logging
# TODO (Arthur): initialize with non-zero fluxes, by running continuous submodels first

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
CHECKPOINTING_SIM_OBJ = config_multialgorithm['checkpointing_sim_obj_name']


class MultialgorithmSimulation(object):
    """ Initialize a multialgorithm simulation from a language model and run-time parameters

    Create a simulation from a model described by a `wc_lang` `Model`.

    Attributes:
        model (:obj:`Model`): a model description
        args (:obj:`dict`): parameters for the simulation; if `results_dir` is an entry in `args`,
            then `checkpoint_period` must also be included
        simulation (:obj:`SimulationEngine`): the initialized simulation
        checkpointing_sim_obj (:obj:`MultialgorithmicCheckpointingSimObj`): the checkpointing object;
            `None` if absent
        random_state (:obj:`numpy.random.RandomState`): a random state
        init_populations (:obj:`dict` from species id to population): the initial populations of
            species, derived from the specification in `model`
        local_species_population (:obj:`LocalSpeciesPopulation`): a shared species population for the
            multialgorithm simulation
        dynamic_model (:obj:`DynamicModel`): the dynamic state of a model being simulated
        temp_dynamic_compartments (:obj:`dict`): the simulation's `DynamicCompartment`s, one for each
            compartment in `model`; temporary attribute used until :obj:`DynamicModel` is made
    """

    def __init__(self, model, args):
        """
        Args:
            model (:obj:`Model`): the model being simulated
            args (:obj:`dict`): parameters for the simulation
        """
        # initialize simulation infrastructure
        self.simulation = SimulationEngine()
        self.random_state = RandomStateManager.instance()

        # create simulation attributes
        self.model = model
        self.args = args or []

        # a model without submodels cannot be simulated
        if not self.model.get_submodels():
            raise MultialgorithmError(f"model {self.model.id} cannot be simulated because it contains"
                                      f" no submodels")

    def build_simulation(self):
        """ Prepare a multialgorithm simulation

        Returns:
            :obj:`tuple` of (`SimulationEngine`, `DynamicModel`): an initialized simulation and its
                dynamic model
        """
        self.set_simultaneous_execution_priorities()
        self.initialize_components()
        self.initialize_infrastructure()
        return (self.simulation, self.dynamic_model)

    def initialize_components(self):
        """ Initialize the biological components of a simulation
        """
        self.create_dynamic_compartments()
        self.init_species_pop_from_distribution()
        self.local_species_population = self.make_local_species_population()
        self.prepare_dynamic_compartments()

    def initialize_infrastructure(self):
        """ Initialize the infrastructure of a simulation
        """
        self.dynamic_model = DynamicModel(self.model, self.local_species_population,
                                          self.temp_dynamic_compartments)
        self.temp_dynamic_compartments = None
        for comp in self.dynamic_model.dynamic_compartments.values():
            comp.dynamic_model = self.dynamic_model
        self.dynamic_model.set_stop_condition(self.simulation)
        # print('self.args')
        # pprint(self.args)
        if 'results_dir' in self.args and self.args['results_dir']:
            self.checkpointing_sim_obj = self.create_multialgorithm_checkpointing(
                self.args['results_dir'],
                self.args['checkpoint_period'])
        self.dynamic_model.dynamic_submodels = self.create_dynamic_submodels()

    def molecular_weights_for_species(self, species=None):
        """ Obtain the molecular weights for species with specified ids

        A weight of `NaN` is returned for species whose species_types do not have a structure.

        Args:
            species (:obj:`iterator`, optional): an iterator over species ids; if not initialized,
                obtain weights for all species in the model

        Returns:
            :obj:`dict`: species_type_id -> molecular weight
        """
        species_weights = {}
        species = species or [species.id for species in self.model.get_species()]
        for species_id in species:
            species_type_id, _ = Species.parse_id(species_id)
            species_type = self.model.species_types.get_one(id=species_type_id)
            if species_type.structure:
                species_weights[species_id] = species_type.structure.molecular_weight
            else:
                species_weights[species_id] = float('nan')
        return species_weights

    def init_species_pop_from_distribution(self):
        """ Initialize the species populations

        Uses the dynamic compartment volume previously sampled from its distribution
        """
        self.init_populations = {}
        for species in self.model.get_species():
            dynamic_compartment = self.temp_dynamic_compartments[species.compartment.id]
            self.init_populations[species.id] = ModelUtilities.sample_copy_num_from_concentration(
                species, dynamic_compartment.init_volume, self.random_state)

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
            dynamic_compartments[comp.id] = self.dynamic_model.dynamic_compartments[comp.id]
        return dynamic_compartments

    def create_dynamic_compartments(self):
        """ Create the :obj:`DynamicCompartment`\ s for this simulation
        """
        # create DynamicCompartments
        self.temp_dynamic_compartments = {}
        for compartment in self.model.get_compartments():
            self.temp_dynamic_compartments[compartment.id] = DynamicCompartment(None, self.random_state,
                                                                           compartment)

    def prepare_dynamic_compartments(self):
        """ Prepare the :obj:`DynamicCompartment`\ s for this simulation
        """
        for dynamic_compartment in self.temp_dynamic_compartments.values():
            dynamic_compartment.initialize_mass_and_density(self.local_species_population)

    def make_local_species_population(self, retain_history=True):
        """ Create a :obj:`LocalSpeciesPopulation` that contains all the species in a model

        Instantiate a :obj:`LocalSpeciesPopulation` as the centralized store of a model's species population.

        Args:
            retain_history (:obj:`bool`, optional): whether the :obj:`LocalSpeciesPopulation` should
                retain species population history

        Returns:
            :obj:`LocalSpeciesPopulation`: a :obj:`LocalSpeciesPopulation` for the model
        """
        molecular_weights = self.molecular_weights_for_species()

        # Species used by continuous time submodels (like DFBA and ODE) need initial population slopes
        # which indicate that the species is modeled by a continuous time submodel.
        # TODO(Arthur): support non-zero initial population slopes; calculate them with initial runs of dFBA and ODE submodels
        init_pop_slopes = {}
        for submodel in self.model.get_submodels():
            if are_terms_equivalent(submodel.framework, onto['WC:ordinary_differential_equations']) or \
                    are_terms_equivalent(submodel.framework, onto['WC:dynamic_flux_balance_analysis']):
                # todo: understand why get_children() of submodel needs kind=submodel
                for species in submodel.get_children(kind='submodel', __type=Species):
                    init_pop_slopes[species.id] = 0.0

        return LocalSpeciesPopulation(
            'LSP_' + self.model.id,
            self.init_populations,
            molecular_weights,
            initial_population_slopes=init_pop_slopes,
            random_state=self.random_state,
            retain_history=retain_history)

    def set_simultaneous_execution_priorities(self):
        """ Assign simultaneous execution priorities for all simulation objects and submodels
        """
        # simulation objects and submodels executing at the same simulation time will run in this order:
        SimObjClassPriority.assign_decreasing_priority([SsaSubmodel,
                                                        DsaSubmodel,
                                                        DfbaSubmodel,
                                                        OdeSubmodel,
                                                        MultialgorithmicCheckpointingSimObj])

    def create_multialgorithm_checkpointing(self, checkpoints_dir, checkpoint_period):
        """ Create a multialgorithm checkpointing object for this simulation

        Args:
            checkpoints_dir (:obj:`str`): the directory that will contain checkpoints
            checkpoint_period (:obj:`float`): interval between checkpoints, in simulated seconds

        Returns:
            :obj:`MultialgorithmicCheckpointingSimObj`: the checkpointing object
        """
        multialgorithm_checkpointing_sim_obj = MultialgorithmicCheckpointingSimObj(
            CHECKPOINTING_SIM_OBJ, checkpoint_period, checkpoints_dir,
            self.local_species_population, self.dynamic_model, self)
        # print(f'multialgorithm_checkpointing_sim_obj: {multialgorithm_checkpointing_sim_obj}')

        # add the multialgorithm checkpointing object to the simulation
        self.simulation.add_object(multialgorithm_checkpointing_sim_obj)
        return multialgorithm_checkpointing_sim_obj

    def create_dynamic_submodels(self):
        """ Create dynamic submodels that access shared species

        Returns:
            :obj:`list`: list of the simulation's `DynamicSubmodel`\ s

        Raises:
            :obj:`MultialgorithmError`: if a submodel cannot be created
        """
        # make the simulation's submodels
        simulation_submodels = {}
        for lang_submodel in self.model.get_submodels():

            # don't create a submodel with no reactions
            if not lang_submodel.reactions:
                warnings.warn(f"not creating submodel '{lang_submodel.id}': no reactions provided",
                              MultialgorithmWarning)
                continue

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
                simulation_submodel = DfbaSubmodel(
                    lang_submodel.id,
                    self.dynamic_model,
                    list(lang_submodel.reactions),
                    lang_submodel.get_children(kind='submodel', __type=Species),
                    self.get_dynamic_compartments(lang_submodel),
                    self.local_species_population,
                    self.args['dfba_time_step']
                )

            elif are_terms_equivalent(lang_submodel.framework, onto['WC:ordinary_differential_equations']):
                simulation_submodel = OdeSubmodel(
                    lang_submodel.id,
                    self.dynamic_model,
                    list(lang_submodel.reactions),
                    lang_submodel.get_children(kind='submodel', __type=Species),
                    self.get_dynamic_compartments(lang_submodel),
                    self.local_species_population,
                    self.args['ode_time_step']
                )

            elif are_terms_equivalent(lang_submodel.framework, onto['WC:deterministic_simulation_algorithm']):
                # a deterministic simulation algorithm, used for testing
                simulation_submodel = DsaSubmodel(
                    lang_submodel.id,
                    self.dynamic_model,
                    list(lang_submodel.reactions),
                    lang_submodel.get_children(kind='submodel', __type=Species),
                    self.get_dynamic_compartments(lang_submodel),
                    self.local_species_population
                )

            else:
                raise MultialgorithmError(f"Unsupported lang_submodel framework '{lang_submodel.framework}'")

            simulation_submodels[simulation_submodel.id] = simulation_submodel

            # add the submodel to the simulation
            self.simulation.add_object(simulation_submodel)

        return simulation_submodels

    def __str__(self):
        """ Provide a readable representation of this `MultialgorithmSimulation`

        Returns:
            :obj:`str`: a readable representation of this `MultialgorithmSimulation`
        """
        return obj_to_str(self, ['args', 'simulation', 'dynamic_compartments', 'dynamic_model',
                                 'init_populations', 'local_species_population', 'model',
                                  'checkpointing_sim_obj', 'simulation_submodels'])
