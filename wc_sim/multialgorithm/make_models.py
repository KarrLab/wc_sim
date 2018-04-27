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
                          SpeciesCoefficient, ObservableCoefficient, Parameter,
                          RateLaw, RateLawDirection, RateLawEquation, SubmodelAlgorithm, Concentration,
                          BiomassComponent, BiomassReaction, StopCondition)
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

    def make_test_model(self, model_type, specie_copy_number=1000000, init_vol=1E-16):
        """ Create a test model

        Args:
            model_type (:obj:`str`): model type description
            specie_copy_number (:obj:`int`): population of each species in its compartment

        Returns:
            :obj:`Model`: a `wc_lang` model
        """
        concentration = self.convert_pop_conc(specie_copy_number, init_vol)

        num_species, num_reactions, reversible, rate_law_type = self.get_model_type_params(model_type)
        if (2<num_species or 1<num_reactions or
            (0<num_reactions and num_species==0) or
            (rate_law_type == RateLawType.product_pop and num_species != 2)):
            raise ValueError("invalid combination of num_species ({}), num_reactions ({}), rate_law_type ({})".format(
                num_species, num_reactions, rate_law_type.name))

        # Model
        model = Model(id='test_model', version='0.0.0', wc_lang_version='0.0.1')
        # Compartment
        comp = model.compartments.create(id='c', initial_volume=init_vol)

        # SpeciesTypes, Species and Concentrations
        species = []
        for i in range(num_species):
            spec_type = model.species_types.create(
                id='spec_type_{}'.format(i),
                type=SpeciesTypeType.protein,
                molecular_weight=10)
            spec = comp.species.create(species_type=spec_type)
            species.append(spec)
            Concentration(species=spec, value=concentration)
        # Submodel
        submodel = model.submodels.create(id='test_submodel', algorithm=SubmodelAlgorithm.ssa,
            compartment=comp)

        # Reactions and RateLaws
        if num_species:
            backward_product = forward_reactant = species[0]
            if 1<num_species:
                backward_reactant = forward_product = species[1]

        # ignore modifiers, which aren't used by the simulator
        if num_reactions:
            reaction = submodel.reactions.create(id='test_reaction_1', reversible=reversible)
            reaction.participants.create(species=forward_reactant, coefficient=-1)
            if rate_law_type.name =='constant':
                equation=RateLawEquation(expression='1')
            if rate_law_type.name == 'reactant_pop':
                equation=RateLawEquation(expression=forward_reactant.id())
            if rate_law_type.name == 'product_pop':
                equation=RateLawEquation(expression=forward_product.id())
            reaction.rate_laws.create(direction=RateLawDirection.forward, equation=equation)

            if num_species == 2:
                reaction.participants.create(species=forward_product, coefficient=1)

            if reversible:
                # make backward rate law
                # RateLawEquations identical to the above must be recreated so backreferences work
                if rate_law_type.name == 'constant':
                    equation=RateLawEquation(expression='1')
                if rate_law_type.name == 'reactant_pop':
                    equation=RateLawEquation(expression=backward_reactant.id())
                if rate_law_type.name == 'product_pop':
                    equation=RateLawEquation(expression=backward_product.id())

                rate_law = RateLaw(direction=RateLawDirection.backward, equation=equation)
                reaction.rate_laws.add(rate_law)

        # Parameters
        model.parameters.create(id='fractionDryWeight', value=0.3)
        model.parameters.create(id='carbonExchangeRate', value=12, units='mmol/gDCW/h')
        model.parameters.create(id='nonCarbonExchangeRate', value=20, units='mmol/gDCW/h')

        # prepare & check the model
        SplitReversibleReactionsTransform().run(model)
        PrepareModel(model).run()
        # check model transcodes the rate law expressions
        CheckModel(model).run()

        # create Manager indices
        # TODO(Arthur): should be automated in a finalize() method
        for base_model in [Submodel,  SpeciesType, Reaction, Observable, Compartment, Parameter]:
            base_model.get_manager().insert_all_new()

        self.model = model
        return model
