''' 
Reads models specified in Excel into a Python object

@author Jonathan Karr, karr@mssm.edu
@date 3/22/2016
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
'''

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cobra import Metabolite as CobraMetabolite
    from cobra import Model as CobraModel
    from cobra import Reaction as CobraReaction
from itertools import chain
from numpy import random
from Sequential_WC_Simulator.core.utilities import N_AVOGADRO
import Sequential_WC_Simulator.core.utilities
import math
import numpy as np
import re
from Sequential_WC_Simulator.multialgorithm.model_representation import Model, ExchangedSpecies


class SpeciesCount(object):
    '''
    operations:
        val = read()
        write( val )
    All operations occur at the current simulation time.
    '''
    __slots__ = 'id val time'.split()
    def __init__( self, id, val = None ):
        if val:
            self.val = val
        else:
            self.val = 0
        self.time = 0
        # report val to CellState
    
    def read( self, now ):
        if self.time == now:
            return self.val
        else:
            # get val from CellState
            pass
            
    def write( self, now, val ):
        if self.time == now and self.val = val:
            # nothing to update
            return
        # report val to CellState
            
    
#Represents a submodel
# DES PLANNING COMMENT(Arthur): should be a SimulationObject
class Submodel(object):
    
    def __init__(self, id = '', name = '', reactions = [], species = []):
        # COMMENT(Arthur): make required args positional, to ensure they're provided
        self.id = id
        self.name = name
        self.reactions = reactions
        self.species = species
        
    def setupSimulation(self):
        #initialize species counts dictionary
        '''
        DES PLANNING COMMENT(Arthur):
        Plan for species counts:
        0) prototype: all species counts in local SpeciesCounts objects that wrap simulation scheduled accesses to CellState
        1) locality optimization: local species counts distinguish between shared and private species
            shared species accesses all mapped into scheduled accesses to CellState
            private species accesses simply access local species counts
        '''
        # self.speciesCounts = { species.id : 0 for species in self.species}
        self.speciesCounts = { species.id : SpeciesCount( species.id, 0) for species in self.species}
        
    #sets local species counts from global species counts
    def updateLocalCellState(self, model):
        # DES PLANNING COMMENT(Arthur): DES must replace this with SpeciesCounts objects
        for species in self.species:
            self.speciesCounts[species.id] = model.speciesCounts[species.species.index, species.compartment.index]
        self.volume = model.volume
        self.extracellularVolume = model.extracellularVolume
    
    #sets global species counts from local species counts 
    # COMMENT(Arthur): so this just overwrites global counts with local counts
    def updateGlobalCellState(self, model):
        # DES PLANNING COMMENT(Arthur): DES must replace this with SpeciesCounts objects
        for species in self.species:
            model.speciesCounts[species.species.index, species.compartment.index] = self.speciesCounts[species.id]
            
    #get species concentrations
    def getSpeciesConcentrations(self):
        # DES PLANNING COMMENT(Arthur): DES can use this; just access SpeciesCounts objects
        volumes = self.getSpeciesVolumes()
        concs = {}
        for species in self.species:
            concs[species.id] = self.speciesCounts[species.id] / volumes[species.id] / N_AVOGADRO
        return concs
        
    #get container volumes for each species
    def getSpeciesVolumes(self):
        # DES PLANNING COMMENT(Arthur): DES can use this
        volumes = {}
        for species in self.species:
            if species.compartment.id == 'c':
                volumes[species.id] = self.volume
            else:
                volumes[species.id] = self.extracellularVolume
        return volumes
        
    #calculate reaction rates
    @staticmethod
    def calcReactionRates(reactions, speciesConcentrations):
        # DES PLANNING COMMENT(Arthur): DES can use this
        rates = np.full(len(reactions), np.nan)
        for iRxn, rxn in enumerate(reactions):          
            if rxn.rateLaw:
                # COMMENT(Arthur): nice
                # COMMENT(Arthur): would be good to catch SyntaxError exceptions, which may not get detected
                # until here
                rates[iRxn] = eval(rxn.rateLaw.transcoded, {}, {'speciesConcentrations': speciesConcentrations, 'Vmax': rxn.vmax, 'Km': rxn.km})
        return rates
               
    #update species counts based on a reaction
    @staticmethod
    def executeReaction(speciesCounts, reaction):
        # DES PLANNING COMMENT(Arthur): DES can use this
        for part in reaction.participants:
            speciesCounts[part.id] += part.coefficient
        return speciesCounts
    
    def getComponentById(self, id, components = None):
        # DES PLANNING COMMENT(Arthur): DES can use this
        if not components:
            components = chain(self.species, self.reactions, self.parameters)
        
        for component in components:
            if component.id == id:
                return component
        
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
        Submodel.setupSimulation(self)        
                
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
            
    def updateLocalCellState(self, model):
        Submodel.updateLocalCellState(self, model)
        self.dryWeight = model.dryWeight
        
    def updateGlobalCellState(self, model):
        Submodel.updateGlobalCellState(self, model)
        model.growth = self.growth
                        
    def calcReactionFluxes(self, timeStep = 1):
        '''calculate growth rate'''
        self.cobraModel.optimize()
        
        self.reactionFluxes = self.cobraModel.solution.x
        self.growth = self.reactionFluxes[self.metabolismProductionReaction['index']] #fraction cell/s
        
    def updateMetabolites(self, timeStep = 1):
        #biomass production
        for part in self.metabolismProductionReaction['reaction'].participants:
            self.speciesCounts[part.id] -= self.growth * part.coefficient * timeStep
        
        #external nutrients
        for exSpecies in self.exchangedSpecies:
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
            upperBounds[exSpecies.reactionIndex] = max(0, np.minimum(upperBounds[exSpecies.reactionIndex], self.speciesCounts[exSpecies.id]) / timeStep)
        
        #exchange bounds
        lowerBounds = utilities.nanminimum(lowerBounds, self.dryWeight / 3600 * N_AVOGADRO * 1e-3 * self.exchangeRateBounds['lower'])
        upperBounds = utilities.nanminimum(upperBounds, self.dryWeight / 3600 * N_AVOGADRO * 1e-3 * self.exchangeRateBounds['upper'])
        
        #return
        arrCbModel = self.cobraModel.to_array_based_model()
        arrCbModel.lower_bounds = lowerBounds
        arrCbModel.upper_bounds = upperBounds
        
#Represents an SSA submodel
class SsaSubmodel(Submodel):
    def __init__(self, *args, **kwargs):
        Submodel.__init__(self, *args, **kwargs)
        self.algorithm = 'SSA'
        
    def setupSimulation(self):
        Submodel.setupSimulation(self)
            
    @staticmethod
    def stochasticSimulationAlgorithm(speciesCounts, speciesVolumes, reactions, volume, timeMax):
        # COMMENT(Arthur): strange statement; can't reactions be kept in right data structure?
        if len(reactions) >= 1 and not isinstance(reactions[0], list):
            reactions = [reactions]
            
        nSubmodels = len(reactions)

        time = 0
        while time < timeMax:
            #calculate concentrations
            speciesConcentrations = {}
            for id, cnt in speciesCounts.iteritems():
                speciesConcentrations[id] = speciesCounts[id] / speciesVolumes[id] / N_AVOGADRO
        
            #calculate propensities
            totalPropensities = np.zeros(nSubmodels)
            reactionPropensities = []
            for iSubmodel in range(nSubmodels):
                # COMMENT(Arthur): each submodel separate in OO DES
                # COMMENT(Arthur): I understand physicality of concentrations and propensities, 
                # but wasteful to divide by volume & N_AVOGADRO and then multiply by them

                # COMMENT(Arthur): optimization: only calculate new reaction rates for species whose 
                # speciesConcentrations (counts) change
                p = np.maximum(0, Submodel.calcReactionRates(reactions[iSubmodel], speciesConcentrations) * volume * N_AVOGADRO)
                totalPropensities[iSubmodel] = np.sum(p)
                reactionPropensities.append(p)
            
            #Select time to next reaction from exponential distribution
            dt = random.exponential(1/np.sum(totalPropensities))
            # COMMENT(Arthur): OO DES avoids this if statement, as timeMax will be end of simulation
            if time + dt > timeMax:
                if random.rand() > (timeMax - time) / dt:                
                    break
                else:
                    dt = timeMax - time
            
            #Select next reaction
            # COMMENT(Arthur): OO DES executes each submodel separately
            iSubmodel = random.choice(nSubmodels, p = totalPropensities / np.sum(totalPropensities))                    
            iRxn = random.choice(len(reactionPropensities[iSubmodel]), p = reactionPropensities[iSubmodel] / totalPropensities[iSubmodel])

            #update time and execute reaction
            time += dt
            speciesCounts = Submodel.executeReaction(speciesCounts, reactions[iSubmodel][iRxn])
                
        return speciesCounts
