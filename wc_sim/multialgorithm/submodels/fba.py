""" A Flux Balance Analysis (FBA) sub-model that represents a set of reactions

:Author: Jonathan Karr, karr@mssm.edu
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-07-14
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import sys
import numpy as np
import warnings
from scipy.constants import Avogadro

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cobra import Metabolite as CobraMetabolite
    from cobra import Model as CobraModel
    from cobra import Reaction as CobraReaction

from wc_sim.core.simulation_object import SimulationObject
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel
from wc_utils.util.misc import isclass_by_name


class FbaSubmodel(DynamicSubmodel):
    """
    FbaSubmodel employs Flux Balance Analysis to predict the reaction fluxes of
    a set of chemical species in a 'well-mixed' container constrained by maximizing
    biomass increase.

    # TODO(Arthur): expand description
    # TODO(Arthur): change variable names to lower_with_under style

    Attributes:
        time_step: float; time between FBA executions
        metabolismProductionReaction
        exchangedSpecies
        cobraModel
        thermodynamicBounds
        exchangeRateBounds
        defaultFbaBound
        reactionFluxes
        Plus see superclasses.

    Event messages:
        RunFba
        # messages after future enhancement
        AdjustPopulationByContinuousSubmodel
        GetPopulation
        GivePopulation
    """

    # Message types sent by FbaSubmodel
    SENT_MESSAGE_TYPES = [
        message_types.RunFba,
        message_types.AdjustPopulationByContinuousSubmodel,
        message_types.GetPopulation,
    ]

    # At any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [
        message_types.GivePopulation,
        message_types.RunFba,
    ]

    def __init__(self, id, dynamic_model, reactions, species, parameters, dynamic_compartment,
        local_species_population, time_step):
        """ Initialize an FBA submodel

        # TODO(Arthur): expand description

        Args:
            See pydocs of super classes.
            dynamic_model (:obj: `DynamicModel`): the aggregate state of a simulation
            time_step: float; time between FBA executions
        """
        super().__init__(id, dynamic_model, reactions, species, parameters, dynamic_compartment, local_species_population)
        self.algorithm = 'FBA'
        if time_step <= 0:
            raise MultialgorithmError("time_step must be positive, but is {}".format(time_step))
        self.time_step = time_step

        # log initialization data
        self.log_with_time("init: id: {}".format(id))
        self.log_with_time("init: time_step: {}".format(str(time_step)))

        self.metabolismProductionReaction = None
        self.exchangedSpecies = None

        self.cobraModel = None
        self.thermodynamicBounds = None
        self.exchangeRateBounds = None

        self.defaultFbaBound = 1e15
        self.reactionFluxes = np.zeros(0)

        self.set_up_fba_submodel()

    def set_up_fba_submodel(self):
        """Set up this FBA submodel for simulation.

        Setup species fluxes, reaction participant, enzyme counts matrices.
        Create initial event for this FBA submodel.
        """

        cobraModel = CobraModel(self.id)
        self.cobraModel = cobraModel

        # setup metabolites
        cbMets = []
        for species in self.species:
            cbMets.append(CobraMetabolite(id=species.serialize(), name=species.species_type.name))
        cobraModel.add_metabolites(cbMets)

        # setup reactions
        for rxn in self.reactions:
            cbRxn = CobraReaction(
                id=rxn.id,
                name=rxn.name,
                lower_bound=-self.defaultFbaBound if rxn.reversible else 0,
                upper_bound=self.defaultFbaBound,
                objective_coefficient=1 if rxn.id == 'MetabolismProduction' else 0,
            )
            cobraModel.add_reaction(cbRxn)

            cbMets = {}
            for part in rxn.participants:
                cbMets[part.species.serialize()] = part.coefficient
            cbRxn.add_metabolites(cbMets)

        # add external exchange reactions
        self.exchangedSpecies = []
        for i_species, species in enumerate(self.species):
            if species.compartment.id == 'e':
                cbRxn = CobraReaction(
                    id='{}Ex'.format(species.serialize()),
                    name='{} exchange'.format(species.serialize()),
                    lower_bound=-self.defaultFbaBound,
                    upper_bound=self.defaultFbaBound,
                    objective_coefficient=0)
                cobraModel.add_reaction(cbRxn)
                cbRxn.add_metabolites({species.serialize(): 1})

                self.exchangedSpecies.append(ExchangedSpecies(
                    id=species.serialize(),
                    species_index=i_species,
                    fba_reaction_index=cobraModel.reactions.index(cbRxn),
                    is_carbon_containing=species.species_type.is_carbon_containing()))

        # add biomass exchange reaction
        cbRxn = CobraReaction(
            id='BiomassEx',
            name='Biomass exchange',
            lower_bound=0,
            upper_bound=self.defaultFbaBound,
            objective_coefficient=0,
        )
        cobraModel.add_reaction(cbRxn)
        # TODO(Arthur): generalize this: the biomass reaction may not be named 'Biomass', and may
        # model compartments other than 'c'
        cbRxn.add_metabolites({'biomass[c]': -1})

        """Bounds"""
        # thermodynamic
        arrayCobraModel = cobraModel.to_array_based_model()
        self.thermodynamicBounds = {
            'lower': np.array(arrayCobraModel.lower_bounds.tolist()),
            'upper': np.array(arrayCobraModel.upper_bounds.tolist()),
        }

        # exchange reactions
        carbonExRate = self.get_component_by_id('carbonExchangeRate', 'parameters').value
        nonCarbonExRate = self.get_component_by_id('nonCarbonExchangeRate', 'parameters').value
        self.exchangeRateBounds = {
            'lower': np.full(len(cobraModel.reactions), -np.nan),
            'upper': np.full(len(cobraModel.reactions), np.nan),
        }

        for exSpecies in self.exchangedSpecies:
            if self.get_component_by_id(exSpecies.id, 'species').species.is_carbon_containing():
                self.exchangeRateBounds['lower'][exSpecies.fba_reaction_index] = -carbonExRate
                self.exchangeRateBounds['upper'][exSpecies.fba_reaction_index] = carbonExRate
            else:
                self.exchangeRateBounds['lower'][exSpecies.fba_reaction_index] = -nonCarbonExRate
                self.exchangeRateBounds['upper'][exSpecies.fba_reaction_index] = nonCarbonExRate

        """Setup reactions"""
        self.metabolismProductionReaction = {
            'index': cobraModel.reactions.index(cobraModel.reactions.get_by_id('MetabolismProduction')),
            'reaction': self.get_component_by_id('MetabolismProduction', 'reactions'),
        }

        self.schedule_next_FBA_analysis()

    def schedule_next_FBA_analysis(self):
        """Schedule the next analysis by this FBA submodel.
        """
        self.send_event(self.time_step, self, message_types.RunFba())

    def calcReactionFluxes(self):
        """calculate growth rate.
        """

        """
        assertion because
            arrCbModel = self.cobraModel.to_array_based_model()
            arrCbModel.lower_bounds = lowerBounds
            arrCbModel.upper_bounds = upperBounds
        was assigning a list to the bound for each reaction
        """
        for r in self.cobraModel.reactions:
            assert (isinstance(r.lower_bound, np.float64) and isinstance(r.upper_bound, np.float64))

        self.cobraModel.optimize()
        self.reactionFluxes = self.cobraModel.solution.x
        self.model.growth = self.reactionFluxes[
            self.metabolismProductionReaction['index']]  # fraction cell/s

    def updateMetabolites(self):
        """Update species (metabolite) counts and fluxes.
        """
        # biomass production
        adjustments = {}
        local_fluxes = {}
        # TODO(Arthur): important, adjustments and local_fluxes need to be converted into copy number values
        for participant in self.metabolismProductionReaction['reaction'].participants:
            # was: self.speciesCounts[part.id] -= self.model.growth * part.coefficient * timeStep
            adjustments[participant.id] = (-self.model.growth * participant.coefficient * self.time_step,
                                           -self.model.growth * participant.coefficient)

        # external nutrients
        for exSpecies in self.exchangedSpecies:
            # was: self.speciesCounts[exSpecies.id] +=
            # self.reactionFluxes[exSpecies.fba_reaction_index] * timeStep
            adjustments[exSpecies.id] = (self.reactionFluxes[exSpecies.fba_reaction_index] * self.time_step,
                                         self.reactionFluxes[exSpecies.fba_reaction_index])

        self.model.local_species_population.adjust_continuously(self.time, adjustments)

    def calcReactionBounds(self):
        """Compute FBA reaction bounds.
        """
        # thermodynamics
        lowerBounds = self.thermodynamicBounds['lower'].copy()
        upperBounds = self.thermodynamicBounds['upper'].copy()

        # rate laws
        # DC: use DC volume
        upperBounds[0:len(self.reactions)] = np.fmin(
            upperBounds[0:len(self.reactions)],
            DynamicSubmodel.calc_reaction_rates(self.reactions)
            * self.model.volume * Avogadro)

        # external nutrients availability
        specie_counts = self.get_specie_counts()
        for exSpecies in self.exchangedSpecies:
            upperBounds[exSpecies.fba_reaction_index] = max(0,
                                                            np.minimum(
                                                                upperBounds[
                                                                    exSpecies.fba_reaction_index], specie_counts[
                                                                    exSpecies.id])
                                                            / self.time_step)

        # exchange bounds
        lowerBounds = np.fmin(lowerBounds, self.model.dryWeight / 3600 * Avogadro
                              * 1e-3 * self.exchangeRateBounds['lower'])
        upperBounds = np.fmin(upperBounds, self.model.dryWeight / 3600 * Avogadro
                              * 1e-3 * self.exchangeRateBounds['upper'])

        for i_rxn, rxn in enumerate(self.cobraModel.reactions):
            rxn.lower_bound = lowerBounds[i_rxn]
            rxn.upper_bound = upperBounds[i_rxn]

    # todo: restructure
    def handle_event(self, event_list):
        """Handle a FbaSubmodel simulation event.

        In this shared-memory FBA, the only event is RunFba, and event_list should
        always contain one event.

        Args:
            event_list: list of event messages to process
        """
        # call handle_event() in class SimulationObject which performs generic
        # tasks on the event list
        SimulationObject.handle_event(self, event_list)
        if not self.num_events % 100:
            print("{:7.1f}: submodel {}, event {}".format(self.time, self.id, self.num_events))

        for event in event_list:
            if isclass_by_name(event.message, message_types.GivePopulation):

                pass
                # TODO(Arthur): add this functionality; currently, handling RunFba
                # accesses memory directly

                # population_values is a GivePopulation body attribute
                population_values = event.message.population

                self.log_with_time("GivePopulation: {}".format(str(event.message)))
                # store population_values in some local cache ...

            elif isclass_by_name(event.message, message_types.RunFba):

                self.log_with_time("submodel '{}' executing".format(self.id))

                # run the FBA analysis
                self.calcReactionBounds()
                self.calcReactionFluxes()
                self.updateMetabolites()
                self.schedule_next_FBA_analysis()

            else:
                assert False, "Error: the 'if' statement should handle "\
                    "event.message '{}'".format(event.message)


class ExchangedSpecies(object):
    """ Represents an exchanged species and its exchange reaction

    Attributes:
        id (:obj:`str`): id
        species_index (:obj:`int`): index of exchanged species within list of species
        fba_reaction_index (:obj:`int`): index of species' exchange reaction within list of cobra model reactions
        is_carbon_containing(:obj:`bool`): indicates if exchanged species contains carbon
    """

    def __init__(self, id, species_index, fba_reaction_index, is_carbon_containing):
        """ Construct an object to represent an exchanged species and its exchange reaction

        Args:
            id (:obj:`str`): id
            species_index (:obj:`int`): index of exchanged species within list of species
            fba_reaction_index (:obj:`int`): index of species' exchange reaction within list of cobra model reactions
            is_carbon_containing(:obj:`bool`): indicates if exchanged species contains carbon
        """
        self.id = id
        self.species_index = species_index
        self.fba_reaction_index = fba_reaction_index
        self.is_carbon_containing = is_carbon_containing
