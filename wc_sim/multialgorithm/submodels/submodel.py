'''A generic submodel - multiple submodel are combined into a multi-algorithmic model.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Author: Jonathan Karr, karr@mssm.edu
:Date: 2016-03-22
:Copyright: 2016, Karr Lab
:License: MIT
'''
#Represents a submodel
# TODO(Arthur): IMPORTANT: unittest all these methods
# TODO(Arthur): IMPORTANT: document with Sphinx

from itertools import chain

import math
import numpy as np
import re

from scipy.constants import Avogadro
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.multialgorithm.utils import species_compartment_name
from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation
from wc_sim.multialgorithm.debug_logs import logs as debug_logs

class Submodel(SimulationObject):
    '''
    Attributes:
        private_cell_state: a LocalSpeciesPopulation that stores the copy numbers of the species involved in reactions
            that are modeled only by this SsaSubmodel instance.
        shared_cell_state: a list of LocalSpeciesPopulation that store the copy numbers of
            the species that are modeled by this Submodel AND other Submodel instances.
    # TODO(Arthur): start using private_cell_state

    '''

    def __init__(self, model, name, id, private_cell_state, shared_cell_state, reactions, species,
        parameters ):
        '''Initialize a submodel.

        Args:
            model (`Model`): the model to which this submodel belongs
            name (str): the name of this submodel / simulation object
            id (type): unique identifier for this submodel
            private_cell_state (`LocalSpeciesPopulation`): a LocalSpeciesPopulation that stores the
                copy numbers of the species that are modeled only by this submodel.
            shared_cell_state (`LocalSpeciesPopulation`): a LocalSpeciesPopulation that stores the
                copy numbers of species that are collectively modeled by this and other submodels.
            reactions (list): the reactions modeled by this submodel.
            species (list): the species that participate in the reactions modeled by this submodel.
            parameters (list): the model's parameters
        '''
        self.model = model  # the model which this Submodel belongs to
        self.name = name
        self.id = id
        self.private_cell_state = private_cell_state
        self.shared_cell_state = shared_cell_state
        self.reactions = reactions
        self.species = species
        self.parameters = parameters
        SimulationObject.__init__( self, name )

    def get_specie_counts(self):
        '''Get a dictionary of current species counts for this submodel.

        Returns:
            Current species count, in a dict: species_id -> count
        '''
        species_ids = set([s.id for s in self.species])
        return self.shared_cell_state.read(self.time, species_ids)

    def get_specie_concentrations(self):
        '''Get a dictionary of current species concentrations for this submodel.

        Returns:
            Current species concentrations, in a dict: species_id -> concentration
        '''
        counts = self.get_specie_counts()
        ids = [ s.id for s in self.species ]
        return { specie_id:(counts[specie_id] / self.model.volume) / Avogadro for specie_id in ids }

    @staticmethod
    def calc_reaction_rates(reactions, species_concentrations):
        '''Calculate the rates for a submodel's reactions.

        Rates computed by eval'ing reactions provided in the model definition,
        with species concentrations obtained by lookup from the dict
        `species_concentrations`.

        Args:
            reactions (list): reactions modeled by a calling submodel
            species_concentrations (dict): `species_id` -> `Species()` object; current
                concentration of species participating in reactions

        Returns:
            A numpy array of reaction rates, indexed by reaction index.
        '''
        rates = np.full(len(reactions), np.nan)
        for iRxn, rxn in enumerate(reactions):
            if rxn.rate_law:
                rates[iRxn] = Submodel.eval_rate_law(rxn, species_concentrations)
        return rates

    @staticmethod
    def eval_rate_law(reaction, species_concentrations):
        '''Evaluate a reaction's rate law with respect to the given species concentrations.

        Args:
            reaction (:obj:`Reaction`): A Reaction instance.
            species_concentrations (:obj:`dict` of :obj:`species_id` -> :obj:`Species`):
                A dictionary of species concentrations.

        Returns:
            float: The reaction's rate for the given species concentrations.

        Raises:
            AttributeError: If the reaction's rate law has not been transcoded.
            ValueError: If the reaction's rate law has a syntax error.
            NameError: If the rate law references a specie whose concentration is not provided in
                `species_concentrations`.
        '''
        rate_law = reaction.rate_law
        try:
            transcoded_reaction = rate_law.transcoded
        except AttributeError as error:
            raise AttributeError("Error: reaction '{}' must have rate law '{}' transcoded.".format(
                reaction.id, reaction.rate_law.law ))
        try:
            # the empty '__builtins__' reduces security risks; see "Eval really is dangerous"
            # return eval(transcoded_reaction, {'__builtins__': {}},
            return eval(transcoded_reaction, {},
                {'species_concentrations': species_concentrations,
                'Vmax': reaction.vmax, 'Km': reaction.km})
        except SyntaxError as error:
            raise ValueError( "Error: reaction '{}' has syntax error in rate law '{}'.".format(
                reaction.id, reaction.rate_law.law ) )
        except NameError as error:
            raise NameError( "Error: NameError in rate law '{}' of reaction '{}': '{}'".format(
                reaction.rate_law.law, reaction.id, error) )

    def enabled_reaction(self, reaction):
        '''Determine whether the cell state has adequate specie counts to run a reaction.

        Args:
            reaction (:obj:`Reaction`): The reaction to evaluate.

        Returns:
            True if reaction is stoichiometrically enabled
        '''
        for participant in reaction.participants:
            species_id = species_compartment_name( participant.species, participant.compartment )
            count = self.model.local_species_population.read_one( self.time, species_id)
            # 'participant.coefficient < 0' constrains the test to reactants
            if participant.coefficient < 0 and count < -participant.coefficient:
                return False
        return True

    def identify_enabled_reactions(self, propensities):
        '''Determine reactions in a propensity array which have adequate specie counts to run.

        A reaction's mass action kinetics, as computed by calc_reaction_rates()
        may be positive when insufficient species are available to execute the reaction.
        identify reactions with inadequate specie counts. Ignore ignore species with
        propensities that are already 0.

        Args:
            propensities: np array; the current propensities for these reactions

        Returns:
            A numpy array with 0 indicating reactions without adequate species counts
                and 1 indicating those with adequate counts, indexed by reaction index.
        '''
        enabled = np.full(len(self.reactions), 1)
        counts = self.get_specie_counts()
        for iRxn, rxn in enumerate(self.reactions):
            # compare each reaction with its stoichiometry,
            # reaction disabled if the specie count of any lhs participant is less than its coefficient
            if propensities[iRxn] <= 0:
                enabled[iRxn] = 0
                continue

            if not self.enabled_reaction(rxn):
                enabled[iRxn] = 0

                log = debug_logs.get_log( 'wc.debug.file' )
                log.debug(
                    "reaction: {} of {}: insufficient counts".format( iRxn, len(self.reactions) ),
                    sim_time=self.time )

        return enabled

    def execute_reaction(self, local_species_population, reaction):
        '''Update species counts based on a single reaction.

        Typically called by an SSA submodel.

        Args:
            speciesCounts: a LocalSpeciesPopulation object; the state storing
                the reaction's participant species
            reaction: a Reaction object; the reaction being executed

        '''
        # TODO(Arthur): move to SsaSubmodel.py, unless other submodels would use
        adjustments={}
        for participant in reaction.participants:
            adjustments[participant.id] = participant.coefficient
        try:
            local_species_population.adjust_discretely( self.time, adjustments )
        except ValueError as e:
            raise ValueError( "{:7.1f}: submodel {}: reaction: {}: {}".format(self.time, self.name,
                reaction.id, e) )

    def get_component_by_id(self, id, component_type=''):
        ''' Find model component with id.

        Args:
            id (:obj:`str`): id of component to find
            component_type (:obj:`str`, optional): type of component to search for; if empty search over all components

        Returns:
            :obj:`object`: component with id, or `None` if there is no component with the id
        '''

        # components to search over
        if component_type in ['species', 'reactions', 'parameters']:
            components = getattr(self, component_type)
        elif not component_type:
            components = chain(self.species, self.reactions, self.parameters)
        else:
            raise Exception('Invalid component type "{}"'.format(component_type))

        # find component
        return next((component for component in components if component.id == id), None)

