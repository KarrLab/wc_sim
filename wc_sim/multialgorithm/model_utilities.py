'''A set of static methods that help prepare Models for simulation.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-04-10
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import collections
import numpy
import pint
import re
from enum import Enum
from numpy.random import RandomState
from scipy.constants import Avogadro
from wc_lang import Species
from wc_onto import onto
from wc_utils.util.list import difference
from wc_utils.util.ontology import are_terms_equivalent
from wc_utils.util.units import unit_registry


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
        species_to_submodels = collections.defaultdict(list)
        for submodel in model.get_submodels():
            for species in submodel.get_children(kind='submodel', __type=Species):
                species_to_submodels[species].append(submodel)

        private_species = dict()
        for submodel in model.get_submodels():
            private_species[submodel] = list()
        for species, submodels in species_to_submodels.items():
            if 1 == len(species_to_submodels[species]):
                submodel = species_to_submodels[species].pop()
                private_species[submodel].append(species)

        # TODO(Arthur): globally s/serialize()/id()/
        if return_ids:
            tmp_dict = {}
            for submodel, species in private_species.items():
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
    def concentration_to_molecules(species, volume, random_state):
        '''Provide the initial copy number of `species` from its concentration

        Copy number is be rounded to the closest integer to avoid truncating small populations.

        Args:
            species (:obj:`Species`): a `Species` instance; the `species.concentration.units` must
                be an instance of `unit_registry.Unit` and in `species.concentration.units.choices`
            volume (:obj:`float`): volume for calculating copy numbers
            random_state (:obj:`RandomState`): random state for sampling from distribution of initial
                concentrations

        Returns:
            `int`: the `species'` copy number

        Raises:
            :obj:`ValueError`: if the concentration uses illegal or unsupported units
        '''
        dist_conc = species.distribution_init_concentration
        if dist_conc is None:
            return 0
        else:
            if not are_terms_equivalent(dist_conc.distribution, onto['WC:normal_distribution']): # normal
                raise ValueError('Unsupported random distribution `{}`'.format(dist_conc.distribution.name))
            mean = dist_conc.mean
            std = dist_conc.std
            if numpy.isnan(std):
                std = mean / 10.
            conc = max(0., random_state.normal(mean, std))

            if not isinstance(dist_conc.units, unit_registry.Unit):
                raise ValueError('Unsupported units "{}"'.format(dist_conc.units))
            units = unit_registry.parse_expression(str(dist_conc.units))

            try:
                scale = units.to(unit_registry.parse_units('molecule'))
                return round(scale.magnitude * conc)
            except pint.DimensionalityError:
                pass
            
            try:
                scale = units.to(unit_registry.parse_units('M'))
                # population must be rounded to the closest integer to avoid truncating small populations
                return int(round(scale.magnitude * conc * volume * Avogadro))
            except pint.DimensionalityError as error:
                pass

            raise ValueError("Unsupported unit '{}'".format(dist_conc.units))

    @staticmethod
    def parse_species_id(species_id):
        '''Get the specie type and compartment from a specie id

        Args:
            species_id (:obj:`string`): a specie id, in the form "species_type_id[compartment]",
            where species_type_id and compartment must be legal python identifiers

        Returns:
            `tuple`: (species_type_id, compartment)

        Raises:
            ValueError: if species_id is not in the right form
        '''
        match = re.match(r'^([a-z][a-z0-9_]*)\[([a-z][a-z0-9_]*)\]$', species_id, flags=re.I)
        if match:
            return (match.group(1), match.group(2))
        raise ValueError("Cannot parse species_id, '{}' not in the form "
                         "'python_identifier[python_identifier]'".format(species_id))

    @staticmethod
    def get_species_types(species_ids):
        '''Get the specie types from an iterator that provides specie ids

        Deterministic -- that is, given a sequence of specie ids provided by `species_ids`
        will always return the same list of specie type ids

        Args:
            species_ids (:obj:`iterator`): an iterator that provides specie ids

        Returns:
            `list`: an iterator over the specie type ids in `species_ids`
        '''
        species_types = set()
        species_types_list = []
        for species_id in species_ids:
            species_type_id, _ = ModelUtilities.parse_species_id(species_id)
            if not species_type_id in species_types:
                species_types.add(species_type_id)
                species_types_list.append(species_type_id)
        return species_types_list
