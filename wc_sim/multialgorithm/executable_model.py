'''Modify a Model so it can be used for multi-algorithmic simulation.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-11-10
:Copyright: 2016, Karr Lab
:License: MIT
'''

from wc_lang.io import Excel
from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation
from wc_sim.multialgorithm.submodels.submodel import Submodel
from wc_sim.multialgorithm.utils import species_compartment_name
from scipy.constants import Avogadro
from collections import defaultdict
import numpy as np

class ExecutableModel(object):
    '''A set of methods that modify a Model so it can be used is a multi-algorithmic simulation.'''

    @staticmethod
    def set_up_simulation(model):
        '''Set up a model for a discrete-event simulation.

        Prepare data that has been loaded into a model.
        The primary simulation data are species counts, stored in a LocalSpeciesPopulation().

        Args:
            model (:obj:`wc_lang.core.Model`): The `Model` instance.
        '''

        model.fraction_dry_weight = model.get_component_by_id('fractionDryWeight', 'parameters').value
        cellular_compartment = model.get_component_by_id('c', 'compartments')
        extracellular_compartment = model.get_component_by_id('e', 'compartments')

        #volume
        model.volume = cellular_compartment.initial_volume
        model.extracellular_volume = extracellular_compartment.initial_volume

        #species counts
        model.cell_state = LocalSpeciesPopulation( model, "CellState", {}, retain_history=True )
        for species in model.species:
            for conc in species.concentrations:
                # TODO(Arthur): just init species that have positive concentrations; have state dynamically
                # add and gc species
                # initializing all fluxes to 0 so that continuous adjustments can be made
                # TODO(Arthur): just initialize fluxes in species that participate in continuous models
                model.cell_state.init_cell_state_specie(
                    species_compartment_name( species, conc.compartment ),
                    conc.value * conc.compartment.initial_volume * Avogadro,
                    initial_flux_given = 0 )

        # transcoded rate laws
        ExecutableModel.transcode_rate_laws(model)

        #cell mass
        ExecutableModel.calc_mass(model, 0)     # initial conditions

        #density
        model.density = model.mass / model.volume

        #growth
        model.growth = np.nan

    @staticmethod
    def calc_mass(model, now):
        '''Calculate the mass of a model, and store it in dry_weight.

        Sum over all species in the model's cytoplasm.

        Args:
            now (float): the current simulation time
        '''
        model.mass = 2
        model.dry_weight = 1
        # ..todo: Need to activate
        print('Need to activate.')
        pass
        
        '''
        mass = 0.
        for submodel in model.submodels:
            mass += submodel.private_species_state.mass(compartments='c')
        mass += model.shared_species.mass()
        mass /= Avogadro
        model.mass = mass
        model.dry_weight = model.fraction_dry_weight * mass
        '''

    @staticmethod
    def get_species_count_array(model, now):
        '''Map current species counts into an np array.

        Args:
            model (:obj:`wc_lang.core.Model`): a `Model` instance
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
                    model.local_species_population.read( now, [specie_name] )[specie_name]
        return species_counts

    @staticmethod
    def find_private_species(model):
        '''Identify the model's species that are private to a submodel.

        Find the species in a model that are modeled by only one submodel.

        Args:
            model (:obj:`wc_lang.core.Model`): a `Model` instance

        Returns:
            dict: a dict that maps each submodel id to a set containing the ids of species that are
                modeled by only the submodel.
        '''
        species_to_submodels = defaultdict(set)
        for submodel in model.submodels:
            for specie in submodel.species:
                species_to_submodels[specie.id].add(submodel.id)

        private_species = dict()
        for submodel in model.submodels:
            private_species[submodel.id] = set()
        for specie_id,submodels in species_to_submodels.items():
            if 1==len(species_to_submodels[specie_id]):
                submodel_id = species_to_submodels[specie_id].pop()
                private_species[submodel_id].add(specie_id)
        return private_species

    @staticmethod
    def find_shared_species(model):
        '''Identify the model's species that are shared by multiple submodels.

        Find the species in a model that are modeled by multiple submodels.

        Args:
            model (:obj:`wc_lang.core.Model`): a `Model` instance

        Returns:
            set: a set containing the ids of shared species.
        '''
        all_species = set()
        for submodel in model.submodels:
            all_species |= set([s.id for s in submodel.species])

        private_species = ExecutableModel.find_private_species(model)
        shared_species = set()
        for p in private_species.values():
            shared_species |= p
        shared_species_ids = all_species - shared_species
        return(shared_species_ids)


        for species in model.species:
            for conc in species.concentrations:
                    species_compartment_name( species, conc.compartment ),

    @staticmethod
    def transcode(rate_law, species, compartments):
        '''Translate a rate law into a python expression that can be evaluated during a simulation.

        The python expression is stored in `rate_law.transcoded`, which is used by
        `Submodel.calc_reaction_rates()`.

        Args:
            rate_law (:obj:`RateLaw`): a `RateLaw` instance
            species (:obj:list): the species in the model containing rate_law
            compartments (:obj:list): the compartments in the model containing rate_law

        Raises:
            ValueError: If `rate_law` contains `__`, increasing its security risk.
        '''
        if getattr(rate_law, 'law', None) is None:
            return
        if '__' in rate_law.law:
            raise ValueError("Security risk: rate law '{}' contains '__'.".format(rate_law.law))

        rate_law.transcoded = rate_law.law
        for spec in species:
            for comp in compartments:
                id = '{0}[{1}]'.format(spec.id, comp.id)
                rate_law.transcoded = rate_law.transcoded.replace(id,
                    "species_concentrations['{}']".format(id))

    @staticmethod
    def get_initial_specie_counts(model):
        '''Get a dictionary of initial species counts for a model.

        Args:
            model (:obj:`wc_lang.core.Model`): The `Model` instance.

        Returns:
            Current species count, in a dict: species_id -> count
        '''
        ids = [s for s in ExecutableModel.all_species(model)]
        return model.cell_state.read(0, ids)

    @staticmethod
    def get_initial_specie_concentrations(model):
        '''Get a dictionary of current species concentrations for this submodel.

        Returns:
            Current species concentrations, in a dict: species_id -> concentration
        '''
        counts = ExecutableModel.get_initial_specie_counts(model)
        ids = [s for s in ExecutableModel.all_species(model)]
        return { specie_id:(counts[specie_id] / model.volume) / Avogadro for specie_id in ids }

    @staticmethod
    def all_species(model):
        '''Generator function that provides all species names.'''
        for species in model.species:
            for conc in species.concentrations:
                yield species_compartment_name(species, conc.compartment)

    @staticmethod
    def transcode_rate_laws(model):
        '''Transcode and test all rate laws in a model.

        Args:
            model (:obj:`wc_lang.core.Model`): The `Model` instance.

        Raises:
            Exception: If any reaction has an error.
        '''
        errors = []
        species_concentrations = ExecutableModel.get_initial_specie_concentrations(model)
        for reaction in model.reactions:
            if getattr(reaction.rate_law, 'law', None) is None:
                continue
            ExecutableModel.transcode(reaction.rate_law, model.species, model.compartments)
            try:
                Submodel.eval_rate_law(reaction, species_concentrations)
            except Exception as error:
                errors.append(str(error))
        if len(errors):
            raise Exception(errors)

class ExchangedSpecies(object):
    ''' Represents an exchanged species and its exchange reaction

    Attributes:
        id (:obj:`str`): id
        species_index (:obj:`int`): index of exchanged species within list of species
        fba_reaction_index (:obj:`int`): index of species' exchange reaction within list of cobra model reactions
        is_carbon_containing(:obj:`bool`): indicates if exchanged species contains carbon
    '''

    def __init__(self, id, species_index, fba_reaction_index, is_carbon_containing):
        ''' Construct an object to represent an exchanged species and its exchange reaction

        Args:
            id (:obj:`str`): id
            species_index (:obj:`int`): index of exchanged species within list of species
            fba_reaction_index (:obj:`int`): index of species' exchange reaction within list of cobra model reactions
            is_carbon_containing(:obj:`bool`): indicates if exchanged species contains carbon
        '''
        self.id = id
        self.species_index = species_index
        self.fba_reaction_index = fba_reaction_index
        self.is_carbon_containing = is_carbon_containing
