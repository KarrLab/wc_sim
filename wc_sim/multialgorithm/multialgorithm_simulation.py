'''Initialize a multialgorithm simulation from a language model and run-time parameters.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-07
:Copyright: 2016-2017, Karr Lab
:License: MIT
'''

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
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation, AccessSpeciesPopulations
from wc_sim.multialgorithm.submodels.submodel import Submodel as SimSubmodel
from wc_sim.multialgorithm.submodels.ssa import SsaSubmodel
from wc_sim.multialgorithm.submodels.fba import FbaSubmodel
from wc_sim.multialgorithm.utils import species_compartment_name
from wc_sim.multialgorithm.species_populations import LOCAL_POP_STORE, Specie, SpeciesPopSimObject
from wc_utils.config.core import ConfigManager
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm

# TODO(Arthur): use lists instead of sets to ensure deterministic behavior

'''
Design notes:

Inputs:
    Static model in a wc_lang.core.Model
    Command line parameters, including:
        Num shared cell state objects
    Optionally, extra config

Output:
    Simulation partitioned into submodels and cell state, including:
        Initialized submodel and state simulation objects
        Initial simulation messages

DS:
    language model
    simulation objects
    an initialized simulation

Algs:
    Partition into Submodels and Cell State:
        1a. Determine shared & private species
        1b. Determine partition
        2. Create shared species object(s)
        3. Create submodels that contain private species and access shared species
        4. Have SimulationObjects send their initial messages

todo: also need model generator
'''

'''
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
'''

class DynamicModel(object):
    '''Represent the aggregate dynamic state of a whole-cell model simulation.

    The primary state of a model being simulated are species counts, which each submodel accesses
    through its `AccessSpeciesPopulations`. A `DynamicModel` provides methods for
    determining aggregate properties, such as model and compartment volume and mass.

    Attributes:
        model (:obj:`wc_lang.core.Model`): the `Model`
        multialgorithm_simulation (:obj:`MultialgorithmSimulation`): the multialgorithm simulation
        fraction_dry_weight (:obj:`float`): fraction of the cell's weight which is not water
            a constant
        volume (:obj:`float`): volume of the cell's cellular (cytoplasm) compartment
        extracellular_volume (:obj:`float`): volume of the cell's extra-cellular
        dry_weight (:obj:`float`): a cell's dry weight
        density (:obj:`float`): cellular density, a constant
        growth (:obj:`float`): growth in cell/s, relative to the cell's initial volume
    '''
    def __init__(self, model, multialgorithm_simulation):
        '''Create a `DynamicModel`
        '''
        self.model = model
        self.multialgorithm_simulation = multialgorithm_simulation

    def initialize(self):
        '''Prepare a `DynamicModel` for a discrete-event simulation
        '''
        # handle multiple cellular compartments
        # e: extracellular_compartment
        # all others: cellular_compartments
        extracellular_compartment_id = 'e'
        extracellular_compartment = utils.get_component_by_id(self.model.get_compartments(),
            extracellular_compartment_id)
        self.extracellular_volume = extracellular_compartment.initial_volume

        cellular_compartments = []
        for compartment in self.model.get_compartments():
            if compartment.id == extracellular_compartment_id:
                continue
            cellular_compartments.append(compartment)

        # volume: sum cellular compartment volumes
        self.volume = sum(
            [cellular_compartment.initial_volume for cellular_compartment in cellular_compartments])

        # cell mass, assuming that the species in the model include water
        # TODO(Arthur): generalize to handle models that do not contain water
        self.mass = self.initial_cell_mass([extracellular_compartment_id])
        self.fraction_dry_weight = utils.get_component_by_id(self.model.get_parameters(),
            'fractionDryWeight').value
        self.dry_weight = self.fraction_dry_weight * self.mass

        # density
        self.density = self.mass / self.volume

        # growth
        self.growth = np.nan

    def initial_cell_mass(self, extracellular_compartments):
        '''Compute the cell's initial mass from the model.

        Sum the mass of all species not stored in an extracellular compartment.

        Args:
            extracellular_compartments (`list`): all extracellular compartments

        Returns:
            `float`: the cell's initial mass (g)
        '''
        # This assumes compartment volumes are in L and concentrations in mol/L
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
        '''Map current species counts into an np array.

        Args:
            now (float): the current simulation time

        Returns:
            numpy array, #species x # compartments, containing count of specie in compartment
        '''
        # TODO(Arthur): avoid wastefully converting between dictionary and array representations of copy numbers
        species_counts = np.zeros((len(model.species), len(model.compartments)))
        for species in model.species:
            for compartment in model.compartments:
                specie_name = species_compartment_name(species, compartment)
                species_counts[ species.index, compartment.index ] = \
                    model.local_species_population.read_one( now, specie_name )
        return species_counts


class MultialgorithmSimulation(object):
    '''A multi-algorithmic simulation

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
            populations stored in `SimulationObject`s
        shared_specie_store_name (:obj:`str`): the name for the shared specie store
        dynamic_model (:obj: `DynamicModel`): the dynamic state of a model; aggregate
            state not available in `Model` or a simulation's `SimulationObject`s
        private_species (:obj: `dict` of `set`): map from submodel to a set of the species
                modeled by only the submodel
        shared_species (:obj: `set`): the shared species
    '''

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
        '''Initialize a simulation.
        '''
        self.check_model()
        self.initialize_biological_state()
        (self.private_species, self.shared_species) = self.partition_species()
        self.transcode_rate_laws()
        self.species_pop_objs = self.create_shared_species_pop_objs()

    def build_simulation(self):
        '''Build a simulation that has been initialized.

        Returns:
            `SimulationEngine`: an initialized simulation
        '''
        self.sub_models = self.create_submodels()
        self.initialize_simulation()
        return self.simulation

    def check_model(self):
        CheckModel(self.model).run()

    def initialize_biological_state(self):
        '''Initialize the biological state of the simulation
        '''
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
        '''Obtain the molecular weights for specified species ids

        Args:
            species (:obj:`set`): a `set` of species ids

        Returns:
            `dict`: species_type_id -> molecular weight
        '''
        specie_weights = {}
        for specie_id in species:
            (specie_type_id, _) = ModelUtilities.parse_specie_id(specie_id)
            specie_weights[specie_id] = SpeciesType.objects.get_one(id=specie_type_id).molecular_weight
        return specie_weights

    def partition_species(self):
        '''Statically partition a `Model`'s `Species` into private species and shared species.

        Returns:
            (dict, set): tuple containing a dict mapping submodels to their private species, and a
                set of shared species
        '''
        return (ModelUtilities.find_private_species(self.model, return_ids=True),
            ModelUtilities.find_shared_species(self.model, return_ids=True))

    def transcode_rate_laws(self):
        '''Transcode all rate law expressions into Python expressions

        Raises:
            ValueError: If a rate law cannot be transcoded
        '''
        for lang_submodel in self.model.get_submodels():
            for reaction in lang_submodel.reactions:
                for rate_law in reaction.rate_laws:
                    rate_law.equation.transcoded = ModelUtilities.transcode(rate_law,
                        lang_submodel.get_species())

    def create_shared_species_pop_objs(self):
        '''Create the shared species object.

        # TODO(Arthur): generalize to multiple `SpeciesPopSimObject` objects

        Returns:
            dict: `dict` mapping id to `SpeciesPopSimObject` objects for the simulation
        '''
        species_pop_sim_obj = SpeciesPopSimObject(self.shared_specie_store_name,
            {specie_id:self.init_populations[specie_id] for specie_id in self.shared_species},
            molecular_weights=self.molecular_weights_for_species(self.shared_species))
        self.simulation.add_object(species_pop_sim_obj)
        return {self.shared_specie_store_name:species_pop_sim_obj}

    def create_access_species_pop(self, lang_submodel):
        '''Create submodels that contain private species and access shared species

        Args:
            lang_submodel (:obj:`Submodel`): description of a submodel

        Returns:
            (:obj:`AccessSpeciesPopulations`): an `AccessSpeciesPopulations` for the `lang_submodel`
        '''
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

    # TODO IMPT COMPLETE THIS
    def create_dfba_submodel_exchange_rxns(self, submodel, extracellular, cytoplasm):
        '''Generate extracellular exchange reactions for a dfba submodel

        Ensure that a DFBA submodel has exchange reactions for all species that are in both the
        extracellular compartment and the cytoplasm.

        Args:
            submodel (`LangSubmodel`): a DFBA submodel
            extracellular (`Compartment`): the model's extracellular compartment
            cytoplasm (`Compartment`): the model's cytoplasm compartment
        '''
        AUTOMATICALLY_CREATED_EXCHANGE_RXN_ID_PREFIX = '__generated_reaction_'
        AUTOMATICALLY_CREATED_EXCHANGE_RXN_NAME_PREFIX = '__generated_reaction_name_'

        reaction_id_num = 1
        specie_types = 'test1 test2 test3'.split()
        # for each specie make Reaction with Participants with Species with Concentration (of 0)
        for specie_type in specie_types:
            rxn_id = "{}_{}".format(AUTOMATICALLY_CREATED_EXCHANGE_RXN_ID_PREFIX, reaction_id_num)
            rxn_name = "{}_{}".format(AUTOMATICALLY_CREATED_EXCHANGE_RXN_NAME_PREFIX, reaction_id_num)
            reaction_id_num += 1
            specie_extra = Specie(species_type=specie_type, compartment=extracellular)
            concentration = Concentration(species=specie_extra, value=0)
            specie_cyto = Specie(species_type=specie_type, compartment=cytoplasm)
            part_extra = P
            submodel.name
            # direction = forward
            participants = "".format()

    # TODO: call this
    def confirm_dfba_submodel_obj_func(self, submodel):
        '''Ensure that a dFBA submodel has an objective function

        If the submodel definition does not provide an objective function, then use the
        biomass reaction.

        Args:
            submodel (`LangSubmodel`): a dFBA submodel

        Raises:
            ValueError: if `submodel` is not a dFBA submodel
            ValueError: if `submodel` cannot use its biomass reaction '{}' as an objective function
        '''
        if submodel.algorithm != SubmodelAlgorithm.dfba:
            raise ValueError("submodel '{}' not a dfba submodel".format(submodel.name))

        if not submodel.objective_function is None:
            return

        # use the biomass reaction
        obj_func_expression = submodel.biomass_reaction.id
        # deserialize the expression
        attr = ObjectiveFunction.Meta.attributes['expression']
        # deserialize needs the biomass reaction and all the Reactions
        objs = {}
        objs[BiomassReaction] = {submodel.biomass_reaction.id:submodel.biomass_reaction}
        objs[Reaction] = dict(zip([rxn.id for rxn in submodel.reactions], submodel.reactions))
        (of, invalid_attribute) = ObjectiveFunction.deserialize(attr, obj_func_expression, objs)
        if invalid_attribute:
            raise ValueError("submodel '{}' cannot use biomass reaction '{}' as an objective function: "
                "{}".format(submodel.name, submodel.biomass_reaction.id, invalid_attribute.messages[0]))
        submodel.objective_function = of

    # TODO: call this
    # TODO: doublecheck Sphinx formatting
    def default_dfba_submodel_flux_bounds(self, submodel):
        ''' Apply default flux bounds to a dFBA submodel's reactions

        The FBA optimizer needs min and max flux bounds for each dFBA submodel reaction.
        If bounds are not provided in some reactions, and default bounds are provided in a config file,
        then apply the defaults to the reactions.
        Specifically, min and max default bounds are applied as follows:
            reversible reactions:
                min_flux = -default_max_flux_bound
                max_flux = default_max_flux_bound
            irreversible reactions:
                min_flux = default_min_flux_bound
                max_flux = default_max_flux_bound

        Args:
            submodel (`LangSubmodel`): a dFBA submodel

        Raises:
            ValueError: if `submodel` is not a dFBA submodel

        Returns:
            :obj:`tuple` of (`int`,`int`): counts of min and max flux bounds set
        '''
        if submodel.algorithm != SubmodelAlgorithm.dfba:
            raise ValueError("submodel '{}' not a dfba submodel".format(submodel.name))

        need_default_flux_bounds = False
        for rxn in submodel.reactions:
            need_default_flux_bounds = need_default_flux_bounds or isnan(rxn.min_flux) or isnan(rxn.max_flux)
        if not need_default_flux_bounds:
            # all reactions have flux bounds
            return (0,0)

        # Are default flux bounds available? They cannot be negative.
        try:
            config_multialgorithm = \
                ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']
            default_min_flux_bound = config_multialgorithm['default_min_flux_bound']
            default_max_flux_bound = config_multialgorithm['default_max_flux_bound']
        except KeyError as e:
            raise ValueError("cannot obtain default_min_flux_bound and default_max_flux_bound=")
        if not 0 <= default_min_flux_bound <= default_max_flux_bound:
            raise ValueError("default flux bounds violate 0 <= default_min_flux_bound <= default_max_flux_bound:\n"
            "default_min_flux_bound={}; default_max_flux_bound={}".format(default_min_flux_bound,
                default_max_flux_bound))

        # Apply default flux bounds to reactions in submodel
        num_default_min_flux_bounds = 0
        num_default_max_flux_bounds = 0
        for rxn in submodel.reactions:
            if isnan(rxn.min_flux):
                num_default_min_flux_bounds += 1
                if rxn.reversible:
                    rxn.min_flux = -default_max_flux_bound
                else:
                    rxn.min_flux = default_min_flux_bound
            if isnan(rxn.max_flux):
                num_default_max_flux_bounds += 1
                rxn.max_flux = default_max_flux_bound
        return (num_default_min_flux_bounds, num_default_max_flux_bounds)

    def create_submodels(self):
        '''Create submodels that contain private species and access shared species

        Returns:
            dict mapping submodel.id to `SimSubmodel`: the simulation's submodels
        '''
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

class CheckModel(object):
    '''Statically check a model

    Checked properties:
        DFBA submodels contain a biomass reaction and an objective function
        Rate laws transcode and evaluate without error
        All reactants in each submodel's reactions are in the submodel's compartment

    Other properties to check:
        The model does not contain dead-end species which are only consumed or produced
        Reactions are balanced
        Reactions in dynamic submodels contain fully specified rate laws
        Consider the reactions modeled by a submodel -- all modifier species used by the rate laws
            for the reactions participate in at least one reaction in the submodel

    # TODO: implement these, and expand the list of properties
    '''

    def __init__(self, model):
        self.model = model

    def run(self):
        self.errors = []
        for submodel in self.model.get_submodels():
            if submodel.algorithm == SubmodelAlgorithm.dfba:
                self.errors.extend(self.check_dfba_submodel(submodel))
            if submodel.algorithm in [SubmodelAlgorithm.ssa, SubmodelAlgorithm.ode]:
                self.errors.extend(self.check_dynamic_submodel(submodel))
        self.errors.extend(self.check_rate_law_equations())
        if self.errors:
            raise ValueError('\n'.join(self.errors))

    def check_dfba_submodel(self, submodel):
        '''Check the inputs to a DFBA submodel

        Ensure that:
            * All regular DFBA reactions have min flux and max flux with appropriate values
            * The DFBA submodel contains a biomass reaction
            TODO * The DFBA submodel contains an objective function

        Args:
            submodel (`LangSubmodel`): a DFBA submodel

        Returns:
            :obj:`list` of `str`: if no errors, returns an empty `list`, otherwise a `list` of
                error messages
        '''
        errors = []
        for reaction in submodel.reactions:
            for attr in ['min_flux', 'max_flux']:
                if not hasattr(reaction, attr) or isnan(getattr(reaction, attr)):
                    errors.append("Error: no {} for reaction '{}' in submodel '{}'".format(
                        attr, reaction.name, submodel.name))
                    continue

            if hasattr(reaction, 'min_flux') and hasattr(reaction, 'max_flux'):
                if reaction.max_flux < reaction.min_flux:
                    errors.append("Error: max_flux < min_flux ({} < {}) for reaction '{}' in submodel '{}'".format(
                        reaction.max_flux, reaction.min_flux, reaction.name, submodel.name))
                if reaction.reversible and 0 < reaction.min_flux:
                    errors.append("Error: 0 < min_flux ({}) for reversible reaction '{}' in submodel '{}'".format(
                        reaction.min_flux, reaction.name, submodel.name))

        if submodel.biomass_reaction is None or not submodel.biomass_reaction.biomass_components:
            errors.append("Error: submodel '{}' uses dfba but lacks a biomass reaction".format(submodel.name))

        return errors

    def check_dynamic_submodel(self, submodel):
        '''Check the inputs to a dynamic submodel

        Ensure that:
            * All reactions have rate laws for the appropriate directions

        Args:
            submodel (`LangSubmodel`): a dynamic (SSA or ODE) submodel

        Returns:
            :obj:`list` of :obj:`str` if no errors, returns an empty `list`, otherwise a `list` of
                error messages
        '''
        errors = []
        for reaction in submodel.reactions:
            direction_types = set()
            for rate_law in reaction.rate_laws:
                direction_types.add(rate_law.direction.name)
            if not direction_types:
                errors.append("Error: reaction '{}' in submodel '{}' has no "
                    "rate law specified".format(reaction.name, submodel.name))
            if reaction.reversible:     # reversible is redundant with a reaction's rate laws
                if direction_types.symmetric_difference(set(('forward', 'backward'))):
                    errors.append("Error: reaction '{}' in submodel '{}' is reversible but has only "
                        "a '{}' rate law specified".format(reaction.name, submodel.name,
                        direction_types.pop()))
            else:
                if direction_types.symmetric_difference(set(('forward',))):
                    errors.append("Error: reaction '{}' in submodel '{}' is not reversible but has "
                        "a 'backward' rate law specified".format(reaction.name, submodel.name))
        return errors

    def check_rate_law_equations(self):
        '''Transcode and evaluate all rate law equations in a model

        Ensure that all rate law equations can be transcoded and evaluated.
        This method is deliberately redundant with `MultialgorithmSimulation.transcode_rate_laws()`,
        which does not report errors.

        Returns:
            :obj:`list` of `str`: if no errors, returns an empty `list`, otherwise a `list` of
                error messages
        '''
        errors = []
        species = self.model.get_species()
        species_concentrations = ModelUtilities.initial_specie_concentrations(self.model)
        for reaction in self.model.get_reactions():
            for rate_law in reaction.rate_laws:
                if getattr(rate_law, 'equation', None) is None:
                    continue
                try:
                    rate_law.equation.transcoded = ModelUtilities.transcode(rate_law, species)
                except Exception as error:
                    errors.append(str(error))
            try:
                rates = ModelUtilities.eval_rate_laws(reaction, species_concentrations)
            except Exception as error:
                errors.append(str(error))
        return errors

    def verify_reactant_compartments(self):
        '''Verify that all reactants in each submodel's reactions are in the submodel's compartment

        Returns:
            :obj:`list` of `str`: if no errors, returns an empty `list`, otherwise a `list` of
                error messages
        '''
        errors = []
        for lang_submodel in self.model.get_submodels():
            compartment = lang_submodel.compartment
            if compartment is None:
                errors.append("submodel '{}' must contain a compartment attribute".format(lang_submodel.id))
                continue
            for reaction in lang_submodel.reactions:
                for participant in reaction.participants:
                    if participant.coefficient < 0:     # select reactants
                        if participant.species.compartment != compartment:
                            error = "submodel '{}' models compartment {}, but its reaction {} uses "\
                            "specie {} in another compartment: {}".format(lang_submodel.id,
                                compartment.id, reaction.id, participant.species.species_type.id,
                                participant.species.compartment.id)
                            errors.append(error)
        return errors
