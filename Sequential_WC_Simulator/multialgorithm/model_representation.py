''' 
Represents a model as a set of objects.

@author Jonathan Karr, karr@mssm.edu
@date 3/22/2016
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
'''

# TODO(Arthur): clean up comments

from itertools import chain
import math
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cobra import Metabolite as CobraMetabolite
    from cobra import Model as CobraModel
    from cobra import Reaction as CobraReaction

from wc_utilities.util.MiscUtilities import N_AVOGADRO
from Sequential_WC_Simulator.multialgorithm.utilities import species_compartment_name
from Sequential_WC_Simulator.multialgorithm.config_constants_old import WC_SimulatorConfig
from Sequential_WC_Simulator.multialgorithm.shared_memory_cell_state import SharedMemoryCellState
from wc_utilities.util.decorate_default_data_struct import default_mutable_params

# logging
from Sequential_WC_Simulator.multialgorithm.config.setup_local_debug_log import debug_log
debugging_logger = debug_log.get_logger( 'wc.debug.file' )


class Model(object):
    """Model represents all the data in a whole-cell model.
    
    Currently, a model is instantiated and loaded from a stored representation in an Excel by
    model_loader.getModelFromExcel().

    Attributes:
        name: type; description
        submodels: list; submodel specifications and references
        compartments: list; compartment specifications 
        species: list; species specifications 
        reactions: list; reactions specifications 
        parameters: type; description
        references: list; submodel specifications 
        fractionDryWeight: type; description
        volume: type; description
        extracellularVolume: type; description
        density: type; description
        growth: type; description
        the_SharedMemoryCellState: type; description
        mass: type; description
        dryWeight: type; description
        speciesCounts: type; description
        debug: boolean; debug status
    # TODO(Arthur): expand this attribute documentation
    """
    
    @default_mutable_params( ['submodels_list', 'compartments_list', 'species_list',
        'reactions_list', 'parameters_list', 'references_list'] )
    def __init__(self, submodels_list = None, compartments_list = None, species_list = None,
        reactions_list = None, parameters_list = None, references_list = None ):
        self.name = 'temp_name'         # TODO(Arthur): get real name
        self.submodels = submodels_list
        self.compartments = compartments_list
        self.species = species_list
        self.reactions = reactions_list
        self.parameters = parameters_list
        self.references = references_list

        # make a logger for this Model
        # write initialization data
        debugging_logger.debug( "init: species: {}".format( str([str(s.name) for s in self.species]) ),
            sim_time=0 )
        debugging_logger.debug( "init: reactions: {}".format( str([str(r.name) for r in self.reactions]) ),
            sim_time=0 )
        
    def setupSimulation(self):
        """Set up a discrete-event simulation from the specification.
        """
        self.fractionDryWeight = self.getComponentById('fractionDryWeight', self.parameters).value
    
        '''
        for subModel in self.submodels:
            subModel.setupSimulation()
        '''
        self.calcInitialConditions()
            
    def calcInitialConditions(self):
        """Set up the initial conditions for a simulation. 
        
        Prepare data that has been loaded into a model.
        The primary simulation data are species counts, stored in a SharedMemoryCellState().
        """
        cellComp = self.getComponentById('c', self.compartments)
        extrComp = self.getComponentById('e', self.compartments)
        
        #volume
        self.volume = cellComp.initialVolume
        self.extracellularVolume = extrComp.initialVolume
        
        #species counts
        self.the_SharedMemoryCellState = SharedMemoryCellState( self, "CellState", {}, 
            retain_history=True )
        for species in self.species:
            for conc in species.concentrations:
                # initializing all fluxes to 0 so that continuous adjustments can be made
                # TODO(Arthur): just initialize species that participate in continuous models
                self.the_SharedMemoryCellState.init_cell_state_specie( 
                    species_compartment_name( species, conc.compartment ), 
                    conc.value * conc.compartment.initialVolume * N_AVOGADRO,
                    initial_flux_given = 0 )
        
        #cell mass
        self.calcMass( 0. )  # initial conditiona at now=0
         
        #density
        self.density = self.mass / self.volume
        
        #growth
        self.growth = np.nan
        
    def getSpeciesCountArray(self, now):
        """Map current species counts into an np array.
        
        Args:
            now: float; the current time
            
        Return:
            numpy array, #species x # compartments, containing count of specie in compartment
        """
        # TODO(Arthur): avoid wastefully converting between dictionary and array representations of copy numbers
        speciesCounts = np.zeros((len(self.species), len(self.compartments)))
        for species in self.species:
            for compartment in self.compartments:
                specie_name = species_compartment_name(species, compartment)
                speciesCounts[ species.index, compartment.index ] = \
                    self.the_SharedMemoryCellState.read( now, [specie_name] )[specie_name]
        return speciesCounts
        
    def calcMass(self, now):
        # now: the current simulation time
        for comp in self.compartments:
            if comp.id == 'c':
                iCellComp = comp.index
                the_cytoplasm = comp
    
        mass = 0.
        speciesCounts = self.getSpeciesCountArray( now )
        for species in self.species:
            # COMMENT(Arthur): isn't a weight of None an error, hopefully caught earlier
            if species.molecularWeight is not None:
                mass += speciesCounts[species.index, iCellComp] * species.molecularWeight
                
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
    # DES PLAN: not used, so do not worry
    """
    def getSpeciesCountsDict(self):           
        speciesCountsDict = {}
        for species in self.species:
            for compartment in self.compartments:
                speciesCountsDict['{0}[{1}]'.format(species.id, compartment.id)] = self.speciesCounts[species.index, compartment.index]                
        return speciesCountsDict
    """
    
    #set species counts from dictionary
    # DES PLAN: also not used, so do not worry
    """
    def setSpeciesCountsDict(self, speciesCountsDict):
        for species in self.species:
            for compartment in self.compartments:
                self.speciesCounts[species.index, compartment.index] = speciesCountsDict['{0}[{1}]'.format(species.id, compartment.id)]
    """
    
    #get species concentrations
    def getSpeciesConcentrations(self, now ):
        speciesCounts = self.getSpeciesCountArray( now )
        return ( speciesCounts / self.getSpeciesVolumes() ) / N_AVOGADRO
        
    #get species concentrations
    def getSpeciesConcentrationsDict(self, now ):
        concs = self.getSpeciesConcentrations( now )
        speciesConcsDict = {}
        for species in self.species:
            for compartment in self.compartments:
                # COMMENT(Arthur): as it's not mutable, one can use a tuple 
                # (in general, any 'hashable') as a dict key
                # e.g., speciesConcsDict[ (species.id, compartment.id) ]
                speciesConcsDict['{1}[{0}]'.format(species.id, compartment.id)] = concs[species.index, compartment.index]                
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
                volumesDict['{0}[{1}]'.format(species.id, compartment.id)] = volumes[species.index, compartment.index]                
        return volumesDict
    
    #get total RNA number        
    def getTotalRnaCount(self):
        cellComp = self.getComponentById('c', self.compartments)
        tot = 0
        # DES PLAN: use a temporary speciesCounts array
        speciesCounts = self.getSpeciesCountArray( now )
        for species in self.species:
            if species.type == 'RNA':
                tot += speciesCounts[species.index, cellComp.index]
        return tot
         
    #get total protein copy number
    def getTotalProteinCount(self):
        cellComp = self.getComponentById('c', self.compartments)
        tot = 0
        # DES PLAN: use a temporary speciesCounts array
        speciesCounts = self.getSpeciesCountArray( now )
        for species in self.species:
            if species.type == 'Protein':
                tot += speciesCounts[species.index, cellComp.index]
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
        
    # TODO(Arthur): need a consistency checker
    

# represent a submodel
class SubmodelSpecification(object):
    """Specification for a submodel, obtained from the input spec.

    Attributes:
        id: string; a unique identifier for the submodel
        name: string; a unique name for the submodel
        algorithm: string; the algorithm used to integrate the submodel
        the_submodel: reference; the Submodel object, after it has been instantiated
    """
    
    def __init__(self, id, name, algorithm ):
        self.id = id
        self.name = name
        self.algorithm = algorithm
        self.the_submodel = None
    
#Represents a compartment
class Compartment(object):
    """Specification for a Compartment, obtained from the input spec.

    Attributes:
        index = None
        id = ''
        name = ''
        initialVolume = None
        comments = ''
    # TODO(Arthur): expand this attribute documentation
    """
    
    def __init__(self, id = '', name = '', initialVolume = None, comments = ''):
        self.id = id
        self.name = name
        self.initialVolume = initialVolume
        self.comments = comments
        
#Represents a species
class Species(object):
    """Specification for a molecular specie, obtained from the input spec.
    
    # TODO(Arthur): fix singular and plural use of specie & species; e.g., this should be a Specie.

    Attributes:
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
    # TODO(Arthur): expand this attribute documentation
    """
    
    @default_mutable_params( ['concentrations_list', 'crossRefs_list'] )
    def __init__(self, id = '', name = '', structure = '', empiricalFormula = '', molecularWeight = None, 
        charge = None, type = '', concentrations_list = None, crossRefs_list = None, comments = ''):
        
        self.id = id    
        self.name = name
        self.structure = structure
        self.empiricalFormula = empiricalFormula
        self.molecularWeight = molecularWeight
        self.charge = charge
        self.type = type
        self.concentrations = concentrations_list
        self.crossRefs = crossRefs_list
        
    def containsCarbon(self):
        if self.empiricalFormula:
            return self.empiricalFormula.upper().find('C') != -1
        return False

#Represents a reaction
class Reaction(object):
    """Specification for a Reaction, obtained from the input spec.

    Attributes:
        index = None
        id = ''
        name = ''
        submodel_spec = None
        reversible = None
        participants = []
        enzyme = ''
        rateLaw = None
        vmax = None
        km = None
        crossRefs = []
        comments = ''
    # TODO(Arthur): expand this attribute documentation
    """

    # COMMENT(Arthur): for debugging would be nice to retain the initial reaction text
    @default_mutable_params( ['participants_list', 'crossRefs_list'] )
    def __init__(self, id = '', name = '', submodel_spec = None, reversible = None, participants_list = None, 
        enzyme = '', rateLaw = '', vmax = None, km = None, crossRefs_list = None, comments = ''):
        
        if vmax:
            vmax = float(vmax)
        if km:
            km = float(km)
        
        self.id = id    
        self.name = name
        self.submodel_spec = submodel_spec
        self.reversible = reversible
        self.participants = participants_list
        self.enzyme = enzyme
        self.rateLaw = rateLaw
        self.vmax = vmax
        self.km = km
        self.crossRefs = crossRefs_list
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
                        partStr += '({:d}) '.format(-part.coefficient)
                    else:
                        partStr += '({:e}) '.format(-part.coefficient)
                partStr += part.species.id
                if globalComp is None:
                    partStr += '[{}]'.format(part.compartment.id)
                lhs.append(partStr)
            else:
                partStr = ''
                if part.coefficient != 1:
                    if math.ceil(part.coefficient) == part.coefficient: 
                        partStr += '({:d}) '.format(part.coefficient)
                    else:
                        partStr += '({:e}) '.format(part.coefficient)
                partStr += part.species.id
                if globalComp is None:
                    partStr += '[{}]'.format(part.compartment.id)
                rhs.append(partStr)
            
        stoichStr = ''
        if globalComp is not None:
            stoichStr += '{}: '.format(globalComp.id)
        stoichStr += '{0} {1}==> {2}'.format(' + '.join(lhs), '<' if self.reversible else '', ' + '.join(rhs))
        
        return stoichStr
        
#Represents a model parameter
class Parameter(object):
    """Specification for a Parameter, obtained from the input spec.

    Attributes:
        index = None
        id = ''
        name = ''
        submodel_spec = None
        value = None
        units = ''
        comments = ''
    # TODO(Arthur): expand this attribute documentation
    """
    
    def __init__(self, id = '', name = '', submodel_spec = '', value = None, units = '', comments = ''):
        self.id = id
        self.name = name
        self.submodel_spec = submodel_spec
        self.value = value
        self.units = units
        self.comments = comments
            
#Represents a reference
class Reference(object):
    """Specification for a Reference, obtained from the input spec.

    Attributes:
        index = None
        id = ''
        name = ''
        crossRefs = None
        comments = ''
    # TODO(Arthur): expand this attribute documentation
    """
    
    @default_mutable_params( ['crossRefs_list'] )
    def __init__(self, id = '', name = '', crossRefs_list = None, comments = ''):
        self.id = id
        self.name = name
        self.crossRefs = crossRefs_list
        self.comments = comments

#Represents a concentration in a compartment
class Concentration(object):
    """Specification for a Concentration, obtained from the input spec.

    Attributes:
        compartment = ''
        value = None
    # TODO(Arthur): expand this attribute documentation
    """
    
    def __init__(self, compartment = '', value = None):
        self.compartment = compartment
        self.value = value
    
#Represents a participant in a submodel
class SpeciesCompartment(object):
    """Specification for a SpeciesCompartment, obtained from the input spec.

    Attributes:
        index = None    
        species = ''
        compartment = ''
    # TODO(Arthur): expand this attribute documentation
    """
    
    def __init__(self, index = None, species = '', compartment = ''):
        self.index = index
        self.species = species
        self.compartment = compartment    
        
    def calcIdName(self):
        self.id = '{0}[{1}]'.format(self.species.id, self.compartment.id)
        self.name = '{0} ({1})'.format(self.species.name, self.compartment.name)
      
#Represents an external 
class ExchangedSpecies(object):
    """Specification for a ExchangedSpecies, obtained from the input spec.

    Attributes:
        id = ''
        reactionIndex = None
    # TODO(Arthur): expand this attribute documentation
    """
    
    def __init__(self, id = '', reactionIndex = None):
        self.id = id
        self.reactionIndex = reactionIndex
    
class ReactionParticipant(object):
    """Specification for a ReactionParticipant, obtained from the input spec.

    Attributes:
        species = ''
        compartment = ''
        coefficient = None
        id
        name
    # TODO(Arthur): expand this attribute documentation
    """
    
    def __init__(self, species = '', compartment = '', coefficient = None):
        self.species = species
        self.compartment = compartment    
        self.coefficient = coefficient
        
    def calcIdName(self):
        # TODO(Arthur): IMPORTANT: dangerous: need to call calcIdName whenever self.species.id or 
        # self.compartment.id changes; much safer to not precompute these, and just create them dynamically
        # applies to other calcIdName() methods too; alternatively, make self.species.id and self.compartment.id
        # private (__id), enable updates through a method, and have that trigger this method
        self.id = '{0}[{1}]'.format(self.species.id, self.compartment.id)
        self.name = '{0} ({1})'.format(self.species.name, self.compartment.name)
        
    def __str__(self):
        """Provide a string representation of this ReactionParticipant.
        
        ReactionParticipant is a strange class, as attributes species and compartment have
        different types ast different times. Hence, the unusual type checking in this method.
        """

        if (isinstance(self.species, str) and isinstance(self.compartment, str) and 
            isinstance(self.coefficient, float) ):
            return "ReactionParticipant of strings: specie: {}; compartment: {}; coefficient: {:6.3f}".format( 
                self.species, self.compartment, self.coefficient )
        if (isinstance(self.species, Species) and isinstance(self.compartment, Compartment) and 
            isinstance(self.coefficient, float) ):
            return "ReactionParticipant of objects: specie: {}; compartment: {}; coefficient: {:6.3f}".format( 
                self.species.id, self.compartment.id, self.coefficient )
        raise ValueError("ReactionParticipant not properly instantiated.".format())
                
#Represents a rate law
class RateLaw(object):
    """Specification for a RateLaw, obtained from the model spec.

    Attributes:
        native = ''
        transcoded = ''
    # TODO(Arthur): expand this attribute documentation
    """
    
    def __init__(self, native = ''):
        self.native = native or ''
        
    #get modifiers of rate law
    def getModifiers(self, species, compartments):
        modifiers = []        
        for spec in species:
            for comp in compartments:
                id = '{0}[{1}]'.format(spec.id, comp.id)
                if self.native.find(id) != -1:
                    modifiers.append(id)
        return modifiers
        
    #transcoded for python
    def transcode(self, species, compartments):
        self.transcoded = self.native
        
        for spec in species:
            for comp in compartments:
                id = '{0}[{1}]'.format(spec.id, comp.id)
                self.transcoded = self.transcoded.replace(id, "speciesConcentrations['{}']".format(id))
                
    def __str__(self):
        return "native: {}\ntranscoded for python: {}".format(self.native, self.transcoded) 

class CrossReference(object):
    """Specification for a CrossReference, obtained from the input spec.

    Attributes:
        source = ''
        id = ''    
    # TODO(Arthur): expand this attribute documentation
    """

    def __init__(self, source = '', id = ''):
        self.source = source
        self.id = id

