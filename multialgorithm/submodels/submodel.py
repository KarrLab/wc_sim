#Represents a submodel
# TODO(Arthur): IMPORTANT: unittest all these methods
# TODO(Arthur): IMPORTANT: document with Sphinx

''' 
@author Jonathan Karr, karr@mssm.edu
@date 3/22/2016
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
'''

from itertools import chain
from numpy import random
from collections import namedtuple

import math
import numpy as np
import re

import Sequential_WC_Simulator.core.utilities
from Sequential_WC_Simulator.core.utilities import N_AVOGADRO
from Sequential_WC_Simulator.multialgorithm.model_representation import Model, ExchangedSpecies
from Sequential_WC_Simulator.multialgorithm.shared_cell_state import SharedMemoryCellState
from Sequential_WC_Simulator.core.SimulationObject import SimulationObject

class Submodel(SimulationObject):
    """
    Attributes:
        private_cell_state: a CellState that stores the copy numbers of the species involved in reactions
            that are modeled only by this simple_SSA_submodel instance.
        shared_cell_states: a list of CellStates that store the copy numbers of
            the species that are modeled by this Submodel AND other Submodel instances.
    # TODO(Arthur): start using private_cell_state

    """
    
    def __init__(self, model, name, id, private_cell_state, shared_cell_states,
        reactions, species, write_plot_output=False):
        """Initialize a submodel.

        Args:
            model: reference; the model to which this submodel belongs
            name: string; name of this submodel / simulation object
            id: string; unique identifier name of this submodel
            private_cell_state: a CellState that stores the copy numbers of the species involved in reactions
                that are modeled only by this submodel.
            shared_cell_states: a list of CellStates that store the copy numbers of
                the species that are collectively modeled by this Submodel and other Submodel instances.
            reactions: list; reactions modeled by this submodel
            species: list; species that participate in the reactions modeled by this submodel
            plot_output: boolean; produce output for plotting
        """
        self.model = model  # the model which this Submodel belongs to
        self.name = name
        self.id = id
        self.private_cell_state = private_cell_state
        self.shared_cell_states = shared_cell_states
        self.reactions = reactions
        self.species = species
        SimulationObject.__init__( self, name, plot_output=write_plot_output )

    '''
    def set_up_submodel(self):
        """Set up this submodel for simulation.
        """
        #initialize species counts dictionary
        # self.speciesCounts = { species.id : SpeciesCount( species.id, 0) for species in self.species}
        pass
    '''
    
    #sets local species counts from global species counts
    def updateLocalCellState(self):
        # DES PLAN: eliminate
        '''
        for species in self.species:
            self.speciesCounts[species.id] = model.speciesCounts[species.species.index, species.compartment.index]
        '''
        self.volume = self.model.volume
        self.extracellularVolume = self.model.extracellularVolume
    
    def get_specie_counts(self, rounded=True):
        """Get a dictionary of current species counts for this submodel.
        
        Args:
            rounded: boolean; if False, do not stochastically round the counts
            # TODO(Arthur): IMPORTANT: make this work
            
        Return:
            Current species count, in a dict: species_id -> count
        """
        ids = map( lambda s: s.id, self.species )
        return self.model.the_SharedMemoryCellState.read( self.time, ids )

    def get_specie_concentrations(self):
        """Get a dictionary of current species concentrations for this submodel.
        
        Return:
            Current species concentrations, in a dict: species_id -> concentration
        """
        counts = self.get_specie_counts()
        ids = map( lambda s: s.id, self.species )
        return { specie_id:(counts[specie_id] / self.model.volume)/N_AVOGADRO for specie_id in ids }

    '''
    # TODO(Arthur): IMPORTANT: discard code I'm not using
    #get species concentrations
    def getSpeciesConcentrations(self, now):
        # DES PLANNING COMMENT(Arthur): DES can use this; just access SpeciesCounts objects
        # DES PLAN: eliminate; just use global concentrations
        return self.model.getSpeciesConcentrationsDict( now )
        volumes = self.getSpeciesVolumes()
        concs = {}
        for species in self.species:
            concs[species.id] = (self.speciesCounts[species.id] / volumes[species.id]) / N_AVOGADRO
        return concs
    '''
        
    #get container volume. for each species
    # TODO(Arthur): IMPORTANT: this doesn't work; is it used?
    def getSpeciesVolumes(self, now):
        # DES PLANNING COMMENT(Arthur): redundant
        return getSpeciesConcentrationsDict( now )
        '''
        volumes = {}
        for species in self.species:
            if species.compartment.id == 'c':
                volumes[species.id] = self.volume
            else:
                volumes[species.id] = self.extracellularVolume
        return volumes
        '''
        
    # TODO(Arthur): make this an instance method, and drop the arguments
    @staticmethod
    def calcReactionRates(reactions, speciesConcentrations):
        """Calculate the rates for a submodel's reactions.
        
        Rates computed by eval'ing reactions provided in the model definition,
        with species concentrations obtained by lookup from the dict 
        speciesConcentrations.

        Args:
            reactions: list; reactions modeled by a calling submodel
            speciesConcentrations: dict: species_id -> Species() object; current 
                concentration of species participating in reactions
            
        Returns:
            A numpy array of reaction rates, indexed by reaction index.
        """
        rates = np.full(len(reactions), np.nan)
        for iRxn, rxn in enumerate(reactions):          
            if rxn.rateLaw:
                try:
                    rates[iRxn] = eval(rxn.rateLaw.transcoded, {}, 
                        {'speciesConcentrations': speciesConcentrations, \
                        'Vmax': rxn.vmax, 'Km': rxn.km})
                except SyntaxError as error:
                    raise ValueError( "Error: reaction '{}' has syntax error in rate law '{}'.".format(
                        rxn.id, rxn.rateLaw.native ) )

        return rates
               
    def identify_enabled_reactions(self, propensities):
        """Determine reactions with adequate specie counts to run.
        
        A reaction's mass action kinetics, as computed by calcReactionRates()
        may be positive when insufficient species are available to execute the reaction. 
        identify reactions with inadequate specie counts. Ignore ignore species with 
        propensities that are already 0.
        
        Args:
            propensities: np array; the current propensities for these reactions
            
        Returns:
            A numpy array with 0 indicating reactions without adequate species counts
                and 1 indicating those with adequate counts, indexed by reaction index.
        """
        enabled = np.full(len(self.reactions), 1)
        counts = self.get_specie_counts()
        for iRxn, rxn in enumerate(self.reactions):
            # compare each reaction with its stoichiometry, 
            # reaction disabled if the specie count of any lhs participant is less than its coefficient
            if propensities[iRxn] <= 0:
                enabled[iRxn] = 0
                continue

            for participant in rxn.participants:
                species_id = Model.species_compartment_name( participant.species, participant.compartment )
                if participant.coefficient < 0 and counts[ species_id ] < -participant.coefficient:
                    '''
                    print "reaction: {} of {}: insufficient counts for {}; available: {}".format( 
                        iRxn, len(self.reactions), participant, counts[ species_id ] )
                    '''
                    enabled[iRxn] = 0
                    continue
        return enabled
               
    def executeReaction(self, the_SharedMemoryCellState, reaction):
        """Update species counts based on a single reaction.
        
        Typically called by an SSA submodel.
        
        Args:
            speciesCounts: a SharedMemoryCellState object; the state storing 
                the reaction's participant species
            reaction: a Reaction object; the reaction being executed
            
        """
        # TODO(Arthur): move to simple_SSA_submodel.py, unless other submodels would use
        adjustments={}
        for participant in reaction.participants:
            adjustments[participant.id] = participant.coefficient
        try:
            the_SharedMemoryCellState.adjust_discretely( self.time, adjustments )
        except ValueError as e:
            raise ValueError( "{:7.1f}: submodel {}: reaction: {}: {}".format(self.time, self.name, reaction.id, e) )

    def getComponentById(self, id, components = None):
        # DES PLANNING COMMENT(Arthur): DES can use this
        if not components:
            components = chain(self.species, self.reactions, self.parameters)
        
        for component in components:
            if component.id == id:
                return component
        
