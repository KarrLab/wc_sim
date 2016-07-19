''' 
Represents a model as a set of objects.

@author Jonathan Karr, karr@mssm.edu
@date 3/22/2016
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
'''

from itertools import chain
import math
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cobra import Metabolite as CobraMetabolite
    from cobra import Model as CobraModel
    from cobra import Reaction as CobraReaction

from Sequential_WC_Simulator.core.utilities import N_AVOGADRO

#Represents a model (submodels, compartments, species, reactions, parameters, references)
# COMMENT(Arthur): tabs were providing indentation, which confused the parser; converted them to spaces
class Model(object):
    
    def __init__(self, submodels = [], compartments = [], species = [], reactions = [], parameters = [], references = []):
        self.submodels = submodels
        self.compartments = compartments
        self.species = species
        self.reactions = reactions
        self.parameters = parameters
        self.references = references       
        
    def setupSimulation(self):
        # COMMENT(Arthur): minor point, but as we discussed fractionDryWeight isn't standard
        # python naming for a variable; here's the Google Python Style Guide
        # https://google.github.io/styleguide/pyguide.html
        self.fractionDryWeight = self.getComponentById('fractionDryWeight', self.parameters).value
    
        for subModel in self.submodels:
            subModel.setupSimulation()

        self.calcInitialConditions()
            
    def calcInitialConditions(self):
        cellComp = self.getComponentById('c', self.compartments)
        extrComp = self.getComponentById('e', self.compartments)
        
        #volume
        self.volume = cellComp.initialVolume
        self.extracellularVolume = extrComp.initialVolume
        
        #species counts
        self.speciesCounts = np.zeros((len(self.species), len(self.compartments)))
        for species in self.species:
            for conc in species.concentrations:
                self.speciesCounts[species.index, conc.compartment.index] = conc.value * conc.compartment.initialVolume * N_AVOGADRO
        
        #cell mass
        self.calcMass()                
         
        #density
        self.density = self.mass / self.volume
        
        #growth
        self.growth = np.nan
        
        #sync submodels
        for subModel in self.submodels:
            subModel.updateLocalCellState(self)
        
    def calcMass(self):
        for comp in self.compartments:
            if comp.id == 'c':
                iCellComp = comp.index
    
        mass = 0.
        for species in self.species:
            # COMMENT(Arthur): isn't a weight of None an error, hopefully caught earlier
            if species.molecularWeight is not None:
                mass += self.speciesCounts[species.index, iCellComp] * species.molecularWeight
        mass /= N_AVOGADRO
        
        self.mass = mass
        self.dryWeight = self.fractionDryWeight * mass
    
    def calcVolume(self):
        self.volume = self.mass / self.density
        
    # COMMENT(Arthur): rather than add an index to each object, if everything has a id
    # one can create a map from id to index and then use that; 
    def setComponentIndices(self):
        for index, obj in enumerate(self.submodels):
            obj.index = index
        for index, obj in enumerate(self.compartments):
            obj.index = index
        for index, obj in enumerate(self.species):
            obj.index = index
        for index, obj in enumerate(self.reactions):
            obj.index = index
        for index, obj in enumerate(self.parameters):
            obj.index = index
        for index, obj in enumerate(self.references):
            obj.index = index
        
    #get species counts as dictionary
    def getSpeciesCountsDict(self):           
        speciesCountsDict = {}
        for species in self.species:
            for compartment in self.compartments:
                speciesCountsDict['%s[%s]' % (species.id, compartment.id)] = self.speciesCounts[species.index, compartment.index]                
        return speciesCountsDict
    
    #set species counts from dictionary
    def setSpeciesCountsDict(self, speciesCountsDict):
        for species in self.species:
            for compartment in self.compartments:
                self.speciesCounts[species.index, compartment.index] = speciesCountsDict['%s[%s]' % (species.id, compartment.id)]
                
    #get species concentrations
    def getSpeciesConcentrations(self):
        # COMMENT(Arthur): I added parens so one doesn't need to know 
        # Python operator precedence to understand the code
        return (self.speciesCounts / self.getSpeciesVolumes()) / N_AVOGADRO
        
    #get species concentrations
    def getSpeciesConcentrationsDict(self):
        concs = self.getSpeciesConcentrations()
        speciesConcsDict = {}
        for species in self.species:
            for compartment in self.compartments:
                # COMMENT(Arthur): as it's not mutable, one can use a tuple 
                # (in general, any 'hashable') as a dict key
                # e.g., speciesConcsDict[ (species.id, compartment.id) ]
                speciesConcsDict['%s[%s]' % (species.id, compartment.id)] = concs[species.index, compartment.index]                
        return speciesConcsDict
    
    #get container volumes for each species
    def getSpeciesVolumes(self):
        cellComp = self.getComponentById('c', self.compartments)
        extracellularComp = self.getComponentById('e', self.compartments)
        
        volumes = np.zeros((len(self.species), len(self.compartments)))
        volumes[:, cellComp.index] = self.volume
        volumes[:, extracellularComp.index] = self.extracellularVolume
        return volumes
        
    #get species volumes as dictionary
    def getSpeciesVolumesDict(self):  
        volumes = self.getSpeciesVolumes()         
        volumesDict = {}
        for species in self.species:
            for compartment in self.compartments:
                volumesDict['%s[%s]' % (species.id, compartment.id)] = volumes[species.index, compartment.index]                
        return volumesDict
    
    #get total RNA number        
    def getTotalRnaCount(self):
        cellComp = self.getComponentById('c', self.compartments)
        tot = 0
        for species in self.species:
            if species.type == 'RNA':
               tot += self.speciesCounts[species.index, cellComp.index]
        return tot
         
    #get total protein copy number
    def getTotalProteinCount(self):
        cellComp = self.getComponentById('c', self.compartments)
        tot = 0
        for species in self.species:
            if species.type == 'Protein':
               tot += self.speciesCounts[species.index, cellComp.index]
        return tot
    
    def getComponentById(self, id, components = None):
        if not components:
            components = chain(self.submodels, self.compartments, self.species, self.reactions, self.parameters, self.references)
            
        for component in components:
            if component.id == id:
                return component
    
    # COMMENT(Arthur): it's often helpful for an object to return a string rep of itself
    # usually this is written in the __str__() method, but here we want a reassuring summary
    def summary( self ):
        counts=[]
        for attr in 'submodels compartments species reactions parameters references'.split():
            counts.append( "{}: {}".format( attr, len( getattr( self, attr ) ) ) )
        return "Model contains:\n{}".format( '\n'.join( counts ) )
        
#Represents a compartment
class Compartment(object):
    index = None
    id = ''
    name = ''
    initialVolume = None
    comments = ''
    
    def __init__(self, id = '', name = '', initialVolume = None, comments = ''):
        self.id = id
        self.name = name
        self.initialVolume = initialVolume
        self.comments = comments
        
#Represents a species
class Species(object):
    index = None
    id = ''
    name = ''
    structure = ''
    empiricalFormula = ''
    molecularWeight = None
    charge = None
    type = ''
    concentrations = []
    crossRefs = []
    comments = ''
    
    def __init__(self, id = '', name = '', structure = '', empiricalFormula = '', molecularWeight = None, 
        charge = None, type = '', concentrations = [], crossRefs = [], comments = ''):
        
        self.id = id    
        self.name = name
        self.structure = structure
        self.empiricalFormula = empiricalFormula
        self.molecularWeight = molecularWeight
        self.charge = charge
        self.type = type
        self.concentrations = concentrations
        self.crossRefs = crossRefs
        
    def containsCarbon(self):
        if self.empiricalFormula:
            return self.empiricalFormula.upper().find('C') != -1
        return False

#Represents a reaction
class Reaction(object):
    index = None
    id = ''
    name = ''
    submodel = ''
    reversible = None
    participants = []
    enzyme = ''
    rateLaw = None
    vmax = None
    km = None
    crossRefs = []
    comments = ''

    # COMMENT(Arthur): for debugging would be nice to retain the initial Stoichiometry text
    def __init__(self, id = '', name = '', submodel = '', reversible = None, participants = [], 
        enzyme = '', rateLaw = '', vmax = None, km = None, crossRefs = [], comments = ''):
        
        if vmax:
            vmax = float(vmax)
        if km:
            km = float(km)
        
        self.id = id    
        self.name = name
        self.submodel = submodel
        self.reversible = reversible
        self.participants = participants
        self.enzyme = enzyme
        self.rateLaw = rateLaw
        self.vmax = vmax
        self.km = km
        self.crossRefs = crossRefs
        self.comments = comments
        
    #convert rate law to python        
    def getStoichiometryString(self):
        globalComp = self.participants[0].compartment
        for part in self.participants:
            if part.compartment != globalComp:
                globalComp = None
                break
        
        lhs = []
        rhs = []
        for part in self.participants:
            if part.coefficient < 0:
                partStr = ''
                if part.coefficient != -1:
                    if math.ceil(part.coefficient) == part.coefficient: 
                        partStr += '(%d) ' % -part.coefficient
                    else:
                        partStr += '(%e) ' % -part.coefficient
                partStr += part.species.id
                if globalComp is None:
                    partStr += '[%s]' % part.compartment.id
                lhs.append(partStr)
            else:
                partStr = ''
                if part.coefficient != 1:
                    if math.ceil(part.coefficient) == part.coefficient: 
                        partStr += '(%d) ' % part.coefficient
                    else:
                        partStr += '(%e) ' % part.coefficient
                partStr += part.species.id
                if globalComp is None:
                    partStr += '[%s]' % part.compartment.id
                rhs.append(partStr)
            
        stoichStr = ''
        if globalComp is not None:
            stoichStr += '[%s]: ' % globalComp.id
        stoichStr += '%s %s==> %s' % (' + '.join(lhs), '<' if self.reversible else '', ' + '.join(rhs))
        
        return stoichStr
        
#Represents a model parameter
class Parameter(object):
    index = None
    id = ''
    name = ''
    submodel = None
    value = None
    units = ''
    comments = ''
    
    def __init__(self, id = '', name = '', submodel = '', value = None, units = '', comments = ''):
        self.id = id
        self.name = name
        self.submodel = submodel
        self.value = value
        self.units = units
        self.comments = comments
            
#Represents a reference
class Reference(object):
    index = None
    id = ''
    name = ''
    crossRefs = []
    comments = ''
    
    def __init__(self, id = '', name = '', crossRefs = [], comments = ''):
        self.id = id
        self.name = name
        self.crossRefs = crossRefs
        self.comments = comments

#Represents a concentration in a compartment
class Concentration(object):
    compartment = ''
    value = None
    
    def __init__(self, compartment = '', value = None):
        self.compartment = compartment
        self.value = value
    
#Represents a participant in a submodel
class SpeciesCompartment(object):
    index = None    
    species = ''
    compartment = ''
    
    id = ''
    name = ''
    
    def __init__(self, index = None, species = '', compartment = ''):
        self.index = index
        self.species = species
        self.compartment = compartment    
        
    def calcIdName(self):
        self.id = '%s[%s]' % (self.species.id, self.compartment.id)
        self.name = '%s (%s)' % (self.species.name, self.compartment.name)
      
#Represents an external 
class ExchangedSpecies(object):
    id = ''
    reactionIndex = None
    
    def __init__(self, id = '', reactionIndex = None):
        self.id = id
        self.reactionIndex = reactionIndex
    
#Represents a participant in a reaction
class ReactionParticipant(object):
    species = ''
    compartment = ''
    coefficient = None
    
    id = ''
    name = ''
    
    def __init__(self, species = '', compartment = '', coefficient = None):
        self.species = species
        self.compartment = compartment    
        self.coefficient = coefficient
        
    def calcIdName(self):
        self.id = '%s[%s]' % (self.species.id, self.compartment.id)
        self.name = '%s (%s)' % (self.species.name, self.compartment.name)

#Represents a rate law
class RateLaw(object):
    native = ''
    transcoded = ''
    
    def __init__(self, native = ''):
        self.native = native or ''
        
    #get modifiers of rate law
    def getModifiers(self, species, compartments):
        modifiers = []        
        for spec in species:
            for comp in compartments:
                id = '%s[%s]' % (spec.id, comp.id)
                if self.native.find(id) != -1:
                    modifiers.append(id)
        return modifiers
        
    #transcoded for python
    def transcode(self, species, compartments):
        self.transcoded = self.native
        
        for spec in species:
            for comp in compartments:
                id = '%s[%s]' % (spec.id, comp.id)
                self.transcoded = self.transcoded.replace(id, "speciesConcentrations['%s']" % id)
        
#Represents a cross reference to an external database
class CrossReference(object):
    source = ''
    id = ''    

    def __init__(self, source = '', id = ''):
        self.source = source
        self.id = id

