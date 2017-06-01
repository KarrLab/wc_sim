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
    def transcode(rate_law, species):
        '''Translate a `wc_lang.core.RateLaw` into a python expression that can be evaluated
        during a simulation.

        Args:
            rate_law (:obj:`RateLaw`): a rate law
            species (:obj:`set`): the species in a submodel that uses the rate law

        Returns:
            The python expression, or None if the rate law doesn't have an equation

        Raises:
            ValueError: If `rate_law` contains `__`, which increases its security risk, or
            if `rate_law` refers to species not in `species`
        '''
        def possible_specie_id(tokens):
            '''Determine whether `tokens` contains a specie id of the form 'string[string]'

            Args:
                tokens (:obj:`list` of (token_num, token_val)): a list of Python tokens

            Returns:
                True if `tokens` might contain a specie id
            '''
            if len(tokens) < 4:
                return False
            toknums = [token[0] for token in tokens]
            specie_id_pattern = [token.NAME, token.OP, token.NAME, token.OP]
            if toknums == specie_id_pattern:
                tokvals = [token[1] for token in tokens]
                return tokvals[1] == '[' and tokvals[3] == ']'

        def convert_specie_name(tokens, species_ids, rate_law_expression):
            '''Translate a `tokens` into a python expression that can be evaluated
            during a simulation

            Args:
                tokens (:obj:`list` of (token_num, token_val)): a list of 4 Python tokens that
                    should comprise a specie id
                species_ids (:obj:`set`): ids of the species used by the rate law expression
                rate_law_expression (:obj:`string`): the rate law expression being transcoded

            Returns:
                (:obj:`string`): a Python expression, transcoded to look up the specie concentration
                    in `concentrations[]`

            Raises:
                ValueError: if `tokens` does not represent a specie in `species_ids`
            '''
            tokvals = [token[1] for token in tokens]
            parsed_id = "{}[{}]".format(tokvals[0], tokvals[2])
            if parsed_id in species_ids:
                return " concentrations['{}']".format(parsed_id)
            else:
                raise ValueError("'{}' not a known specie in rate law".format(
                    parsed_id, rate_law_expression))

        if getattr(rate_law, 'equation', None) is None:
            return
        rate_law_equation = rate_law.equation
        if '__' in rate_law_equation.expression:
            raise ValueError("Security risk: rate law expression '{}' contains '__'.".format(
                rate_law_equation.expression))

        rate_law_expression = rate_law_equation.expression
        species_ids = set([specie.serialize() for specie in species])
        
        # rate laws must be tokenized to properly construct a python expression
        # otherwise, if a specie name matches the suffix of another this string replace will fail:
        #   py_expression = py_expression.replace(id, "concentrations['{}']".format(id))
        #   e.g., suppose these are two species: AB[c], CAB[c]
        #   then the replace operation could produce Cconcentrations['AB[c]']
        g = tokenize.generate_tokens(cStringIO(rate_law_equation.expression).readline)
        tokens = [(toknum, tokval) for toknum, tokval, _, _, _ in g]
        result = []
        idx = 0
        while idx < len(tokens):
            if possible_specie_id(tokens[idx:idx+4]):
                result.append(
                    (token.NAME, convert_specie_name(tokens[idx:], species_ids, rate_law_expression)))
                idx += 4
            else:
                result.append((tokens[idx]))
                idx += 1
        py_expression = tokenize.untokenize(result)
        return py_expression

    @staticmethod
    def eval_rate_laws(reaction, concentrations):
        '''Evaluate a reaction's rate laws at the given species concentrations

        Args:
            reaction (:obj:`Reaction`): a Reaction instance
            concentrations (:obj:`dict` of :obj:`species_id` -> :obj:`Species`):
                a dictionary of species concentrations

        Returns:
            (:obj:`dict` of `float`): the forward and, perhaps, backward rates

        Raises:
            ValueError: if the reaction's rate law has a syntax error
            NameError: if the rate law references a specie whose concentration is not provided in
                `concentrations`
            Error: if the rate law has other errors, such as a reference to an unknown function
        '''
        rates = []
        for rate_law in reaction.rate_laws:
            transcoded_reaction = rate_law.equation.transcoded

            local_ns = {func.__name__: func for func in RateLawEquation.Meta.valid_functions}
            if hasattr(rate_law, 'k_cat') and not isnan(rate_law.k_cat):
                local_ns['k_cat'] = rate_law.k_cat
            if hasattr(rate_law, 'k_m') and not isnan(rate_law.k_m):
                local_ns['k_m'] = rate_law.k_m
            local_ns['concentrations'] = concentrations

            try:
                # the empty '__builtins__' reduces security risks; see "Eval really is dangerous"
                # return eval(transcoded_reaction, {'__builtins__': {}},
                rates.append(eval(transcoded_reaction, {}, local_ns))
            except SyntaxError as error:
                raise ValueError("Error: reaction '{}' has syntax error in transcoded rate law '{}'.".format(
                    reaction.id, transcoded_reaction))
            except NameError as error:
                raise NameError("Error: NameError in transcoded rate law '{}' of reaction '{}': '{}'".format(
                    transcoded_reaction, reaction.id, error))
            except Exception as error:
                raise Exception("Error: error in transcoded rate law '{}' of reaction '{}': '{}'".format(
                    transcoded_reaction, reaction.id, error))
        return rates

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
