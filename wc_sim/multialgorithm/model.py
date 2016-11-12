'''An in-memory representation of a WC model.

:Author: Jonathan Karr, karr@mssm.edu
:Date: 2016-10-10
:Copyright: 2016, Karr Lab
:License: MIT
'''
from scipy.constants import Avogadro
from wc_sim.multialgorithm.shared_memory_cell_state import SharedMemoryCellState
from wc_sim.multialgorithm.utils import species_compartment_name
import numpy as np

no_longer_used

class Model(object):

    def __init__(self, model_def):
        self.submodels = [Submodel(submodel) for submodel in model_def.submodels]
        self.compartments = model_def.compartments
        self.species = model_def.species
        self.reactions = [Reaction(rxn) for rxn in model_def.reactions]
        self.parameters = model_def.parameters
        self.references = model_def.references

        for submodel in self.submodels:
            for i_rxn, rxn in enumerate(submodel.reactions):
                submodel.reactions[i_rxn] = self.get_component_by_id(rxn.id, 'reactions')

    def setupSimulation(self):
        """Set up a discrete-event simulation from the specification. """

        '''Transcode rate laws'''
        for rxn in self.reactions:
            if rxn.rate_law:
                rxn.rate_law.transcode(self.species, self.compartments)

        self.fractionDryWeight = self.get_component_by_id('fractionDryWeight', 'parameters').value

        self.calc_initial_conditions()

    def calc_initial_conditions(self):
        """Set up the initial conditions for a simulation. 

        Prepare data that has been loaded into a model.
        The primary simulation data are species counts, stored in a SharedMemoryCellState().
        """
        cellComp = self.get_component_by_id('c', 'compartments')
        extrComp = self.get_component_by_id('e', 'compartments')

        # volume
        self.volume = cellComp.initial_volume
        self.extracellular_volume = extrComp.initial_volume

        # species counts
        self.shared_memory_cell_state = SharedMemoryCellState(self, "CellState", {},
                                                               retain_history=True)
        for species in self.species:
            for conc in species.concentrations:
                # initializing all fluxes to 0 so that continuous adjustments can be made
                # TODO(Arthur): just initialize species that participate in continuous models
                self.shared_memory_cell_state.init_cell_state_specie(
                    species_compartment_name(species, conc.compartment),
                    conc.value * conc.compartment.initial_volume * Avogadro,
                    initial_flux_given=0)

        # cell mass
        self.calc_mass(0.)  # initial condition at time=0

        # density
        self.density = self.mass / self.volume

        # growth
        self.growth = np.nan

    def get_species_counts(self, time):
        """Map current species counts into an np array.

        Args:
            time: float; the current time

        Return:
            numpy array, #species x # compartments, containing count of specie in compartment
        """
        # TODO(Arthur): avoid wastefully converting between dictionary and array representations of copy numbers
        speciesCounts = np.zeros((len(self.species), len(self.compartments)))
        for i_species, species in enumerate(self.species):
            for i_compartment, compartment in enumerate(self.compartments):
                specie_name = species_compartment_name(species, compartment)
                speciesCounts[ i_species, i_compartment ] = \
                    self.shared_memory_cell_state.read(time, [specie_name])[specie_name]
        return speciesCounts

    def calc_mass(self, time):
        # time: the current simulation time
        i_cell_comp = next(index for index, comp in enumerate(self.compartments) if comp.id == 'c')

        mass = 0.
        speciesCounts = self.get_species_counts(time)
        for i_species, species in enumerate(self.species):
            # COMMENT(Arthur): isn't a weight of None an error, hopefully caught earlier
            if not np.isnan(species.molecular_weight):
                mass += speciesCounts[i_species, i_cell_comp] * species.molecular_weight

        mass /= Avogadro

        self.mass = mass
        self.dryWeight = self.fractionDryWeight * mass

    def get_component_by_id(self, id, component_type=''):
        """ Find model component with id.

        Args:
            id (:obj:`str`): id of component to find
            component_type (:obj:`str`, optional): type of component to search for; if empty search over all components

        Returns:
            :obj:`object`: component with id, or `None` if there is no component with the id
        """

        # components to search over
        if component_type in ['compartments', 'species', 'submodels', 'reactions', 'parameters', 'references']:
            components = getattr(self, component_type)
        elif not component_type:
            components = chain(self.compartments, self.species, self.submodels,
                               self.reactions, self.parameters, self.references)
        else:
            raise Exception('Invalid component type "{}"'.format(component_type))

        # find component
        return next((component for component in components if component.id == id), None)


class Submodel(object):
    """ Represents a submodel.

    Attributes:
        id (:obj:`str`): unique id
        name (:obj:`str`): name
        algorithm (:obj:`str`): algorithm
        species (:obj:`list`): list of species in submodel
        reactions (:obj:`list`): list of reactions in submodel
        parameters (:obj:`list`): list of parameers in submodel
    """

    def __init__(self, submodel_def):
        """ Construct a submodel.

        Args:
            submodel_def. (:obj:`str`): unique id
        """

        self.id = submodel_def.id
        self.name = submodel_def.name
        self.algorithm = submodel_def.algorithm
        self.species = submodel_def.species
        self.reactions = submodel_def.reactions
        self.parameters = submodel_def.parameters

        self.the_submodel = None  # todo: remove

    def get_component_by_id(self, id, component_type=''):
        """ Find model component with id.

        Args:
            id (:obj:`str`): id of component to find
            component_type (:obj:`str`, optional): type of component to search for; if empty search over all components

        Returns:
            :obj:`object`: component with id, or `None` if there is no component with the id
        """

        # components to search over
        if component_type in ['species', 'reactions', 'parameters']:
            components = getattr(self, component_type)
        elif not component_type:
            components = chain(self.species, self.reactions, self.parameters)
        else:
            raise Exception('Invalid component type "{}"'.format(component_type))

        # find component
        return next((component for component in components if component.id == id), None)


class Reaction(object):
    """ Represents a reaction. 

    Attributes:
        id            (:obj:`str`): id
        name          (:obj:`str`): name
        submodel      (:obj:`wc_lang.core.Submodel`): submodel that reaction belongs to
        reversible    (:obj:`bool`): indicates if reaction is thermodynamically reversible
        participants  (:obj:`list`): list of reaction participants
        enzyme        (:obj:`wc_lang.core.SpeciesCompartment`): enzyme
        rate_law      (:obj:`RateLaw`): rate law
        vmax          (:obj:`float`): vmax
        km            (:obj:`float`): km
        cross_refs    (:obj:`list`): list of cross references to external databases
        comments      (:obj:`str`): comments
    """

    def __init__(self, reaction_def):
        """ Construct a reaction.

        Args:
            reaction_def (:obj:`str`): id
        """

        self.id = reaction_def.id
        self.name = reaction_def.name
        self.submodel = reaction_def.submodel
        self.reversible = reaction_def.reversible
        self.participants = reaction_def.participants
        self.enzyme = reaction_def.enzyme
        if reaction_def.rate_law:
            self.rate_law = RateLaw(reaction_def.rate_law)
        else:
            self.rate_law = None
        self.vmax = reaction_def.vmax
        self.km = reaction_def.km
        self.cross_refs = reaction_def.cross_refs
        self.comments = reaction_def.comments


class RateLaw(object):

    def __init__(self, reaction_def):
        self.law = reaction_def.law
        self.transcoded = None

    def transcode(self, species, compartments):
        self.transcoded = self.law

        for spec in species:
            for comp in compartments:
                id = '{0}[{1}]'.format(spec.id, comp.id)
                self.transcoded = self.transcoded.replace(id, "speciesConcentrations['{}']".format(id))


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
