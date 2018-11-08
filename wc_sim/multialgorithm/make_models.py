""" Make simple models for testing

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-04-27
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
from argparse import Namespace
import re
import numpy as np
from scipy.constants import Avogadro

from obj_model import utils
from wc_utils.util.enumerate import CaseInsensitiveEnum
from wc_lang.io import Reader, Writer
from wc_lang.core import (Model, Submodel,  SpeciesType, SpeciesTypeType, Species,
                          Reaction, Observable, Compartment,
                          SpeciesCoefficient, Parameter,
                          RateLaw, RateLawDirection, RateLawEquation, SubmodelAlgorithm, Concentration,
                          BiomassComponent, BiomassReaction, StopCondition, ConcentrationUnit,
                          ExpressionMethods)
from wc_lang.prepare import PrepareModel, CheckModel
from wc_lang.transform import SplitReversibleReactionsTransform


class RateLawType(int, CaseInsensitiveEnum):
    """ Rate law type """
    constant = 1
    reactant_pop = 2
    product_pop = 3


class MakeModels(object):
    """ Make a variety of simple models for testing
    """

    @staticmethod
    def get_model_type_params(model_type):
        """ Given a model type, generate params for creating the model
        """
        num_species = 0
        result = re.search(r'(\d) species', model_type)
        if result:
            num_species = int(result.group(1))

        num_reactions = 0
        result = re.search(r'(\d) reaction', model_type)
        if result:
            num_reactions = int(result.group(1))

        reversible = False
        if 'pair of symmetrical reactions' in model_type:
            reversible = True
            num_reactions = 1

        rate_law_type = RateLawType.constant
        if 'rates given by reactant population' in model_type:
            rate_law_type = RateLawType.reactant_pop
        if 'rates given by product population' in model_type:
            rate_law_type = RateLawType.product_pop

        return (num_species, num_reactions, reversible, rate_law_type)

    @staticmethod
    def convert_pop_conc(specie_copy_number, vol):
        return specie_copy_number/(vol*Avogadro)

    @staticmethod
    def add_test_submodel(model, model_type, submodel_num, init_vol, species_types,
        default_specie_copy_number, specie_copy_numbers, equations):
        """ Create a test submodel

        * 1 compartment
        * 1 submodel
        * 1 or 2 species
        * 1 or 2 reactions

        Args:
            model (:obj:`Model`): model
            model_type (:obj:`str`): model type description
            submodel_num (:obj:`int`): submodel index within list of submodels
            init_vol (:obj:`float`): initial volume of the compartment
            species_types (:obj:`list` of :obj:`SpeciesType`): species types
            default_specie_copy_number (:obj:`int`): default population of all species in their compartments
            specie_copy_numbers (:obj:`dict`): populations for particular species, which overrides `default_specie_copy_number`            
            equations (:obj:`dict`):
        """
        # local: compartment, species, conc, reaction, rate law, init_vol,
        default_concentration = MakeModels.convert_pop_conc(default_specie_copy_number, init_vol)

        num_species, num_reactions, reversible, rate_law_type = MakeModels.get_model_type_params(model_type)

        # Compartment
        comp = model.compartments.create(id='compt_{}'.format(submodel_num),
            name='compartment num {}'.format(submodel_num), initial_volume=init_vol)

        # Species, Concentrations and Observables
        species = []
        objects = {
            Species:{},
            Observable:{}
        }
        for i in range(num_species):
            specie = comp.species.create(species_type=species_types[i])
            species.append(specie)
            objects[Species][specie.get_id()] = specie
            if specie_copy_numbers is not None and specie.id() in specie_copy_numbers:
                concentration = MakeModels.convert_pop_conc(specie_copy_numbers[specie.id()], init_vol)
                Concentration(species=specie, value=concentration, units=ConcentrationUnit.M.value)
            else:
                Concentration(species=specie, value=default_concentration, units=ConcentrationUnit.M.value)
            obs_id = 'obs_{}_{}'.format(submodel_num, i)
            expr = "1.5 * {}".format(specie.get_id())
            objects[Observable][obs_id] = obs_plain = \
                ExpressionMethods.make_obj(model, Observable, obs_id, expr, objects)

            obs_id = 'obs_dep_{}_{}'.format(submodel_num, i)
            expr = "2 * {}".format(obs_plain)
            objects[Observable][obs_id] = ExpressionMethods.make_obj(model, Observable, obs_id, expr, objects)

        # Submodel
        id = 'submodel_{}'.format(submodel_num)
        submodel = model.submodels.create(id=id, name=id, algorithm=SubmodelAlgorithm.ssa)

        # Reactions and RateLaws
        if num_species:
            backward_product = forward_reactant = species[0]
            if 1 < num_species:
                backward_reactant = forward_product = species[1]

        if num_reactions:
            id = 'test_reaction_{}_1'.format(submodel_num)
            reaction = submodel.reactions.create(id=id, name=id, reversible=reversible)
            reaction.participants.create(species=forward_reactant, coefficient=-1)
            if rate_law_type.name == 'constant':
                expression = '1'
                modifiers = []
            if rate_law_type.name == 'reactant_pop':
                expression = forward_reactant.id()
                modifiers = [forward_reactant]
            if rate_law_type.name == 'product_pop':
                expression = forward_product.id()
                modifiers = [forward_product]
            equation = equations.get(expression, None)
            if not equation:
                equation = RateLawEquation(expression=expression, modifiers=modifiers)
                equations[expression] = equation
            reaction.rate_laws.create(direction=RateLawDirection.forward, equation=equation)

            if num_species == 2:
                reaction.participants.create(species=forward_product, coefficient=1)

            if reversible:
                # make backward rate law
                # RateLawEquations identical to the above must be recreated so backreferences work
                if rate_law_type.name == 'constant':
                    expression = '1'
                    modifiers = []
                if rate_law_type.name == 'reactant_pop':
                    expression = backward_reactant.id()
                    modifiers = [backward_reactant]
                if rate_law_type.name == 'product_pop':
                    expression = backward_product.id()
                    modifiers = [backward_product]
                equation = equations.get(expression, None)
                if not equation:
                    equation = RateLawEquation(expression=expression, modifiers=modifiers)
                    equations[expression] = equation
                reaction.rate_laws.create(direction=RateLawDirection.backward, equation=equation)

    @staticmethod
    def make_test_model(model_type,
                        init_vols=None,
                        molecular_weight=10,
                        num_submodels=1,
                        default_specie_copy_number=1000000,
                        specie_copy_numbers=None,
                        transfer_reactions=False,
                        transform_prep_and_check=True):
        """ Create a test model with multiple SSA submodels

        Args:
            model_type (:obj:`str`): model type description
            init_vols (:obj:`list` of `float`, optional): initial volume of each compartment; default=1E-16
            molecular_weight (:obj:`float`): the MW of each species type; default=10
            num_submodels (:obj:`int`): number of submodels
            default_specie_copy_number (:obj:`int`): default population of all species in their compartments
            specie_copy_numbers (:obj:`dict`): populations for particular species, which overrides `default_specie_copy_number`
            transfer_reactions (:obj:`bool`, optional): whether the model contains transfer reactions between
                compartments; to be implemented
            transform_prep_and_check (:obj:`bool`, optional): whether to transform, prepare and check the model

        Returns:
            :obj:`Model`: a `wc_lang` model
        """
        # TODO(Arthur): implement transfer reactions
        # global: model_type, Model, SpeciesTypes, specie_copy_numbers, Parameters, ...
        default_vol = 1E-16
        init_vols = [default_vol]*num_submodels if init_vols is None else init_vols
        if len(init_vols) != num_submodels:
            raise ValueError("len(init_vols) ({}) != num_submodels ({})".format(len(init_vols), num_submodels))

        num_species, num_reactions, reversible, rate_law_type = MakeModels.get_model_type_params(model_type)
        if (2 < num_species or 1 < num_reactions or
            (0 < num_reactions and num_species == 0) or
                (rate_law_type == RateLawType.product_pop and num_species != 2)):
            raise ValueError("invalid combination of num_species ({}), num_reactions ({}), rate_law_type ({})".format(
                num_species, num_reactions, rate_law_type.name))

        if num_submodels<1:
            raise ValueError("invalid num_submodels ({})".format(num_submodels))

        # Model
        model = Model(id='test_model', name='{} with {} submodels'.format(model_type, num_submodels),
            version='0.0.0', wc_lang_version='0.0.1')

        # SpeciesTypes
        species_types = []
        for i in range(num_species):
            spec_type = model.species_types.create(
                id='spec_type_{}'.format(i),
                type=SpeciesTypeType.protein,
                molecular_weight=molecular_weight)
            species_types.append(spec_type)

        # make submodels
        equations = {}
        for i in range(num_submodels):
            submodel_num = i+1
            MakeModels.add_test_submodel(model, model_type, submodel_num, init_vols[i],
                species_types, default_specie_copy_number=default_specie_copy_number,
                specie_copy_numbers=specie_copy_numbers, equations=equations)

        # Parameters
        model.parameters.create(id='fractionDryWeight', value=0.3)
        model.parameters.create(id='carbonExchangeRate', value=12, units='mmol/gDCW/h')
        model.parameters.create(id='nonCarbonExchangeRate', value=20, units='mmol/gDCW/h')

        if transform_prep_and_check:
            # prepare & check the model
            SplitReversibleReactionsTransform().run(model)
            PrepareModel(model).run()
            # check model transcodes the rate law expressions
            CheckModel(model).run()

        # create Manager indices
        # TODO(Arthur): should be automated in a finalize() method
        for base_model in [Submodel,  SpeciesType, Reaction, Observable, Compartment, Parameter]:
            base_model.get_manager().insert_all_new()

        return model
