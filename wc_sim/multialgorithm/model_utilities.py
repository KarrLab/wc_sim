'''A set of static methods that help prepare Models for simulation.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-10
:Copyright: 2016-2017, Karr Lab
:License: MIT
'''
import tokenize, token, re
from collections import defaultdict
from six.moves import cStringIO
from math import isnan

from wc_lang.core import RateLawEquation
from wc_utils.util.list import difference


class ModelUtilities(object):
    '''A set of static methods that help prepare Models for simulation.'''

    @staticmethod
    def find_private_species(model, return_ids=False):
        '''Identify a model's species that are private to a submodel.

        Find the species in a model that are modeled privately by a single submodel. This analysis
        relies on the observation that a submodel can only access species that participate in
        reactions that occurs in the submodel. (It might not access some of these species too,
        if they're initialized with concentration=0 and the reactions in which they participate
        never fire. However, that cannot be determined statically.)

        Args:
            model (:obj:`Model`): a `Model` instance
            return_ids (:obj:`boolean`, optional): if set, return object ids rather than references

        Returns:
            dict: a dict that maps each submodel to a set containing the species
                modeled by only the submodel.
        '''
        species_to_submodels = defaultdict(list)
        for submodel in model.get_submodels():
            for specie in submodel.get_species():
                species_to_submodels[specie].append(submodel)

        private_species = dict()
        for submodel in model.get_submodels():
            private_species[submodel] = list()
        for specie,submodels in species_to_submodels.items():
            if 1==len(species_to_submodels[specie]):
                submodel = species_to_submodels[specie].pop()
                private_species[submodel].append(specie)

        if return_ids:
            tmp_dict = {}
            for submodel,species in private_species.items():
                tmp_dict[submodel.get_primary_attribute()] = list([specie.serialize() for specie in species])
            return tmp_dict
        return private_species

    @staticmethod
    def find_shared_species(model, return_ids=False):
        '''Identify the model's species that are shared by multiple submodels.

        Find the species in a model that are modeled by multiple submodels.

        Args:
            model (:obj:`Model`): a `Model` instance
            return_ids (:obj:`boolean`, optional): if set, return object ids rather than references

        Returns:
            set: a set containing the shared species.
        '''
        all_species = model.get_species()

        private_species_dict = ModelUtilities.find_private_species(model)
        private_species = []
        for p in private_species_dict.values():
            private_species.extend(p)
        shared_species = difference(all_species, private_species)
        if return_ids:
            return set([shared_specie.serialize() for shared_specie in shared_species])
        return(shared_species)

    @staticmethod
    def initial_specie_concentrations(model):
        '''Get a dictionary of the initial species concentrations in a model

        Args:
            model (:obj:`Model`): a `Model` instance

        Returns:
            `dict`: specie_id -> concentration, for each specie
        '''
        concentrations = {specie.serialize():0.0 for specie in model.get_species()}
        for concentration in model.get_concentrations():
            concentrations[concentration.species.serialize()] = concentration.value
        return concentrations

    @staticmethod
    def parse_specie_id(specie_id):
        '''Get the specie type and compartment from a specie id

        Args:
            specie_id (:obj:`string`): a specie id, in the form "specie_type_id[compartment]",
            where specie_type_id and compartment must be legal python identifiers

        Returns:
            `tuple`: (specie_type_id, compartment)

        Raises:
            ValueError: if specie_id is not in the right form
        '''
        match = re.match('^([a-z][a-z0-9_]*)\[([a-z][a-z0-9_]*)\]$', specie_id, flags=re.I)
        if match:
            return (match.group(1), match.group(2))
        raise ValueError("Cannot parse specie_id, '{}' not in the form "
            "'python_identifier[python_identifier]'".format(specie_id))

    @staticmethod
    def get_species_types(specie_ids):
        '''Get the specie types from an iterator that provides specie ids

        Deterministic -- that is, given a sequence of specie ids provided by `specie_ids`
        will always return the same list of specie type ids

        Args:
            specie_ids (:obj:`iterator`): an iterator that provides specie ids

        Returns:
            `list`: an iterator over the specie type ids in `specie_ids`
        '''
        specie_types = set()
        specie_types_list = []
        for specie_id in specie_ids:
            specie_type_id, _ = ModelUtilities.parse_specie_id(specie_id)
            if not specie_type_id in specie_types:
                specie_types.add(specie_type_id)
                specie_types_list.append(specie_type_id)
        return specie_types_list
