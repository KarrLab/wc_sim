"""

@author Jonathan Karr, karr@mssm.edu
Created 2016/07/14
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""
    
import sys
import logging
import numpy as np

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import MessageTypesRegistry
from Sequential_WC_Simulator.multialgorithm.submodels.submodel import Submodel

from MessageTypes import (MessageTypes, 
    GET_POPULATION_body, 
    GIVE_POPULATION_body )
    
#Represents an FBA submodel
class FbaSubmodel(Submodel):
    # COMMENT(Arthur): I want to understand this better.
    metabolismProductionReaction = None 
    exchangedSpecies = None
    
    cobraModel = None
    thermodynamicBounds = None
    exchangeRateBounds = None
    
    defaultFbaBound = 1e15
    
    dryWeight = np.nan
    reactionFluxes = np.zeros(0)
    growth = np.nan
    
    def __init__(self, *args, **kwargs):        
        Submodel.__init__(self, *args, **kwargs)
        self.algorithm = 'FBA'
        
    def setupSimulation(self):
        '''setup reaction participant, enzyme counts matrices'''
        # Submodel.setupSimulation(self)        
                
        '''Setup FBA'''
        cobraModel = CobraModel(self.id)
        self.cobraModel = cobraModel
            
        #setup metabolites
        cbMets = []
        for species in self.species:
            cbMets.append(CobraMetabolite(id = species.id, name = species.name))
        cobraModel.add_metabolites(cbMets)
        
        #setup reactions
        for rxn in self.reactions:            
            cbRxn = CobraReaction(
                id = rxn.id,
                name = rxn.name,
                lower_bound = -self.defaultFbaBound if rxn.reversible else 0,
                upper_bound =  self.defaultFbaBound,
                objective_coefficient = 1 if rxn.id == 'MetabolismProduction' else 0,
                )
            cobraModel.add_reaction(cbRxn)

            cbMets = {}
            for part in rxn.participants:
                cbMets[part.id] = part.coefficient
            cbRxn.add_metabolites(cbMets)            
        
        #add external exchange reactions
        self.exchangedSpecies = []
        for species in self.species:
            if species.compartment.id == 'e':                
                cbRxn = CobraReaction(
                    id = '%sEx' % species.species.id,
                    name = '%s exchange' % species.species.name,
                    lower_bound = -self.defaultFbaBound,
                    upper_bound =  self.defaultFbaBound,
                    objective_coefficient = 0,
                    )
                cobraModel.add_reaction(cbRxn)
                cbRxn.add_metabolites({species.id: 1})
                
                self.exchangedSpecies.append(ExchangedSpecies(id = species.id, reactionIndex = cobraModel.reactions.index(cbRxn)))
        
        #add biomass exchange reaction
        cbRxn = CobraReaction(
            id = 'BiomassEx',
            name = 'Biomass exchange',
            lower_bound = 0,
            upper_bound = self.defaultFbaBound,
            objective_coefficient = 0,
            )
        cobraModel.add_reaction(cbRxn)
        cbRxn.add_metabolites({'Biomass[c]': -1})
        
        '''Bounds'''
        #thermodynamic       
        arrayCobraModel = cobraModel.to_array_based_model()
        self.thermodynamicBounds = {
            'lower': np.array(arrayCobraModel.lower_bounds.tolist()),
            'upper': np.array(arrayCobraModel.upper_bounds.tolist()),
            }
        
        #exchange reactions
        carbonExRate = self.getComponentById('carbonExchangeRate', self.parameters).value
        nonCarbonExRate = self.getComponentById('nonCarbonExchangeRate', self.parameters).value
        self.exchangeRateBounds = {
            'lower': np.full(len(cobraModel.reactions), -np.nan),
            'upper': np.full(len(cobraModel.reactions),  np.nan),
            }
        for exSpecies in self.exchangedSpecies:
            if self.getComponentById(exSpecies.id, self.species).species.containsCarbon():
                self.exchangeRateBounds['lower'][exSpecies.reactionIndex] = -carbonExRate
                self.exchangeRateBounds['upper'][exSpecies.reactionIndex] =  carbonExRate
            else:
                self.exchangeRateBounds['lower'][exSpecies.reactionIndex] = -nonCarbonExRate
                self.exchangeRateBounds['upper'][exSpecies.reactionIndex] =  nonCarbonExRate
            
        '''Setup reactions'''
        self.metabolismProductionReaction = {
            'index': cobraModel.reactions.index(cobraModel.reactions.get_by_id('MetabolismProduction')),
            'reaction': self.getComponentById('MetabolismProduction', self.reactions),
            }
            
    def updateLocalCellState(self):
        Submodel.updateLocalCellState(self)
        self.dryWeight = self.model.dryWeight
        
    def updateGlobalCellState(self, model):
        # Submodel.updateGlobalCellState(self, model)
        model.growth = self.growth
                        
    def calcReactionFluxes(self, timeStep = 1):
        '''calculate growth rate'''
        self.cobraModel.optimize()
        
        self.reactionFluxes = self.cobraModel.solution.x
        self.growth = self.reactionFluxes[self.metabolismProductionReaction['index']] #fraction cell/s
        
    def updateMetabolites(self, timeStep = 1):
        # DES PLANNING COMMENT(Arthur): HERE IS the population ADJUSTMENT we expect in Specie()
        #biomass production
        for part in self.metabolismProductionReaction['reaction'].participants:
            # DES PLAN: directly reference global state
            self.speciesCounts[part.id] -= self.growth * part.coefficient * timeStep
        
        #external nutrients
        for exSpecies in self.exchangedSpecies:
            # DES PLAN: directly reference global state
            self.speciesCounts[exSpecies.id] += self.reactionFluxes[exSpecies.reactionIndex] * timeStep
        
    def calcReactionBounds(self,  timeStep = 1):
        #thermodynamics
        lowerBounds = self.thermodynamicBounds['lower'].copy()
        upperBounds = self.thermodynamicBounds['upper'].copy()
        
        #rate laws
        upperBounds[0:len(self.reactions)] = utilities.nanminimum(
            upperBounds[0:len(self.reactions)], 
            self.calcReactionRates(self.reactions, self.getSpeciesConcentrations()) * self.volume * N_AVOGADRO,
            )
        
        #external nutrients availability
        for exSpecies in self.exchangedSpecies:
            # DES PLAN: directly reference global state
            upperBounds[exSpecies.reactionIndex] = max(0, np.minimum(upperBounds[exSpecies.reactionIndex], self.speciesCounts[exSpecies.id]) / timeStep)
        
        #exchange bounds
        lowerBounds = utilities.nanminimum(lowerBounds, self.dryWeight / 3600 * N_AVOGADRO * 1e-3 * self.exchangeRateBounds['lower'])
        upperBounds = utilities.nanminimum(upperBounds, self.dryWeight / 3600 * N_AVOGADRO * 1e-3 * self.exchangeRateBounds['upper'])
        
        #return
        arrCbModel = self.cobraModel.to_array_based_model()
        arrCbModel.lower_bounds = lowerBounds
        arrCbModel.upper_bounds = upperBounds
