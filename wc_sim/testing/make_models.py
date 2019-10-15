""" Make simple models for testing

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-04-27
:Copyright: 2018, Karr Lab
:License: MIT
"""

from obj_tables.expression import Expression
from scipy.constants import Avogadro
from wc_lang import (Model, Submodel, Compartment, SpeciesType, Species,
                     Observable, Function, FunctionExpression,
                     Reaction, RateLawDirection, RateLawExpression, Parameter,
                     DistributionInitConcentration, Validator, InitVolume, ChemicalStructure)
from wc_lang.transform import PrepForWcSimTransform
from wc_utils.util.enumerate import CaseInsensitiveEnum
from wc_onto import onto
from wc_utils.util.string import indent_forest
from wc_utils.util.units import unit_registry
import re
import numpy as np

class RateLawType(int, CaseInsensitiveEnum):
    """ Rate law type """
    constant = 1
    reactant_pop = 2
    product_pop = 3


class MakeModel(object):
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
    def convert_pop_conc(species_copy_number, vol):
        return species_copy_number / (vol * Avogadro)

    @classmethod
    def add_test_submodel(cls, model, model_type, submodel_num, comp, species_types,
                          default_species_copy_number, default_species_std,
                          species_copy_numbers, species_stds, expressions):
        """ Create a test submodel

        Copy number arguments are converted into concentrations at the mean specified compartment
        volume.

        * 1 compartment
        * 1 submodel
        * 1 or 2 species
        * 1 or 2 reactions

        Args:
            model (:obj:`Model`): model
            model_type (:obj:`str`): model type description
            submodel_num (:obj:`int`): submodel index within list of submodels
            comp (:obj:`Compartment`): the submodel's compartment
            species_types (:obj:`list` of :obj:`SpeciesType`): species types
            default_species_copy_number (:obj:`int`): default population of all species in their compartments
            default_species_std (:obj:`float`): default standard deviation of species populations
            species_copy_numbers (:obj:`dict`): populations for particular species, which overrides
                `default_species_copy_number`
            species_stds (:obj:`dict`): standard deviations of species population, which overrides
                `default_species_std`
            expressions (:obj:`dict`):
        """
        num_species, num_reactions, reversible, rate_law_type = cls.get_model_type_params(model_type)

        objects = {
            Compartment: {},
            Species: {},
            Observable: {},
            Function: {},
            Parameter: {},
        }

        # parameters
        avogadro_param = model.parameters.get_or_create(id='Avogadro')
        avogadro_param.value = Avogadro
        avogadro_param.units = unit_registry.parse_units('molecule mol^-1')

        objects[Parameter][avogadro_param.id] = avogadro_param

        # Compartment
        objects[Compartment][comp.id] = comp
        density = comp.init_density
        objects[Parameter][density.id] = density

        volume = model.functions.create(id='volume_compt_{}'.format(submodel_num),
                                        units=unit_registry.parse_units('l'))
        volume.expression, error = FunctionExpression.deserialize('{} / {}'.format(comp.id, density.id), objects={
            Compartment: {comp.id: comp},
            Parameter: {density.id: density},
        })
        assert error is None, str(error)
        objects[Function][volume.id] = volume

        # Species, Concentrations and Observables
        default_concentration = cls.convert_pop_conc(default_species_copy_number, comp.init_volume.mean)
        default_std = cls.convert_pop_conc(default_species_std, comp.init_volume.mean)
        species = []
        for i in range(num_species):
            specie = comp.species.create(species_type=species_types[i], model=model)
            specie.id = specie.gen_id()
            species.append(specie)
            objects[Species][specie.id] = specie
            if species_copy_numbers is not None and specie.id in species_copy_numbers:
                mean = cls.convert_pop_conc(species_copy_numbers[specie.id], comp.init_volume.mean)
                if species_stds:
                    std = cls.convert_pop_conc(species_stds[specie.id], comp.init_volume.mean)
                else:
                    std = cls.convert_pop_conc(default_species_std, comp.init_volume.mean)
                conc = DistributionInitConcentration(
                    species=specie, mean=mean, std=std,
                    units=unit_registry.parse_units('M'),
                    model=model)
            else:
                conc = DistributionInitConcentration(
                    species=specie, mean=default_concentration, std=default_std,
                    units=unit_registry.parse_units('M'),
                    model=model)
            conc.id = conc.gen_id()
            obs_id = 'obs_{}_{}'.format(submodel_num, i)
            expr = "1.5 * {}".format(specie.id)
            objects[Observable][obs_id] = obs_plain = \
                Expression.make_obj(model, Observable, obs_id, expr, objects)

            obs_id = 'obs_dep_{}_{}'.format(submodel_num, i)
            expr = "2 * {}".format(obs_plain)
            objects[Observable][obs_id] = Expression.make_obj(model, Observable, obs_id, expr, objects)

        # Submodel
        id = 'submodel_{}'.format(submodel_num)
        submodel = model.submodels.create(id=id, name=id, framework=onto['WC:stochastic_simulation_algorithm'])

        # Reactions and RateLaws
        if num_species:
            backward_product = forward_reactant = species[0]
            if 1 < num_species:
                backward_reactant = forward_product = species[1]

        if num_reactions:
            id = 'test_reaction_{}_1'.format(submodel_num)
            reaction = submodel.reactions.create(id=id, name=id, reversible=reversible, model=model)
            reaction.participants.create(species=forward_reactant, coefficient=-1)
            # TODO(Arthur): test all branches here
            if rate_law_type.name == 'constant':
                param = model.parameters.create(id='k_cat_{}_1_for'.format(submodel_num),
                                                value=1., units=unit_registry.parse_units('s^-1'))
                expression_str = param.id
            elif rate_law_type.name == 'reactant_pop':
                param = model.parameters.create(id='k_cat_{}_1_for'.format(submodel_num),
                                                value=1., units=unit_registry.parse_units('M^-1 s^-1'))
                expression_str = '{} * {} / {} / {}'.format(
                    param.id, forward_reactant.id,
                    avogadro_param.id,
                    forward_reactant.compartment.init_density.function_expressions[0].function.id)
            elif rate_law_type.name == 'product_pop':
                param = model.parameters.create(id='k_cat_{}_1_for'.format(submodel_num),
                                                value=1., units=unit_registry.parse_units('M^-1 s^-1'))
                expression_str = '{} * {} / {} / {}'.format(
                    param.id, forward_product.id,
                    avogadro_param.id,
                    forward_product.compartment.init_density.function_expressions[0].function.id)
            objects[Parameter][param.id] = param
            expression_obj = expressions.get(expression_str, None)
            if not expression_obj:
                expression_obj, errors = RateLawExpression.deserialize(expression_str, objects)
                assert errors is None, str(errors)
                expressions[expression_str] = expression_obj
            rl = reaction.rate_laws.create(
                direction=RateLawDirection.forward, expression=expression_obj,
                model=model)
            rl.id = rl.gen_id()

            if num_species == 2:
                reaction.participants.create(species=forward_product, coefficient=1)

            if reversible:
                # make backward rate law
                # RateLawEquations identical to the above must be recreated so back references work
                if rate_law_type.name == 'constant':
                    param = model.parameters.create(id='k_cat_{}_1_bck'.format(submodel_num),
                                                    value=1., units=unit_registry.parse_units('s^-1'))
                    expression_str = param.id
                elif rate_law_type.name == 'reactant_pop':
                    param = model.parameters.create(id='k_cat_{}_1_bck'.format(submodel_num),
                                                    value=1., units=unit_registry.parse_units('M^-1 s^-1'))
                    expression_str = '{} * {} / {} / {}'.format(
                        param.id, backward_reactant.id,
                        avogadro_param.id,
                        backward_reactant.compartment.init_density.function_expressions[0].function.id)
                elif rate_law_type.name == 'product_pop':
                    param = model.parameters.create(id='k_cat_{}_1_bck'.format(submodel_num),
                                                    value=1., units=unit_registry.parse_units('M^-1 s^-1'))
                    expression_str = '{} * {} / {} / {}'.format(
                        param.id, backward_product.id,
                        avogadro_param.id,
                        backward_product.compartment.init_density.function_expressions[0].function.id)
                objects[Parameter][param.id] = param
                expression_obj = expressions.get(expression_str, None)
                if not expression_obj:
                    expression_obj, errors = RateLawExpression.deserialize(expression_str, objects)
                    assert errors is None
                    expressions[expression_str] = expression_obj
                rl = reaction.rate_laws.create(direction=RateLawDirection.backward, expression=expression_obj,
                                               model=model)
                rl.id = rl.gen_id()

    @classmethod
    def make_test_model(cls, model_type,
                        init_vols=None,
                        init_vol_stds=None,
                        density=1100,
                        molecular_weight=10.,
                        charge=0,
                        num_submodels=1,
                        default_species_copy_number=1000000,
                        default_species_std=100000,
                        species_copy_numbers=None,
                        species_stds=None,
                        transfer_reactions=False,
                        transform_prep_and_check=True):
        """ Create a test model with multiple SSA submodels

        Properties of the model:

        * Each submodel runs SSA
        * Each submodel has one compartment

        Args:
            model_type (:obj:`str`): model type description
            init_vols (:obj:`list` of :obj:`float`, optional): initial volume of each compartment; default=1E-16
            init_vol_stds (:obj:`list` of :obj:`float`, optional): initial std. dev. of volume of each
                compartment; default=`init_vols/10.`
            density (:obj:`float`, optional): the density of each compartment; default=1100 g/l
            molecular_weight (:obj:`float`, optional): the molecular weight of each species type; default=10
            charge (:obj:`int`, optional): charge of each species type; default=0
            num_submodels (:obj:`int`, optional): number of submodels
            default_species_copy_number (:obj:`int`, optional): default population of all species in
                their compartments
            default_species_std (:obj:`int`, optional): default standard deviation of population of
                all species in their compartments
            species_copy_numbers (:obj:`dict`, optional): populations for particular species, which
                overrides `default_species_copy_number`
            species_stds (:obj:`dict`, optional): standard deviations for particular species, which
                overrides `default_species_std`
            transfer_reactions (:obj:`bool`, optional): whether the model contains transfer reactions
                between compartments; to be implemented
            transform_prep_and_check (:obj:`bool`, optional): whether to transform, prepare and check
                the model

        Returns:
            :obj:`Model`: a `wc_lang` model

        Raises:
            :obj:`ValueError`: if arguments are inconsistent
        """
        # TODO(Arthur): implement transfer reactions
        num_species, num_reactions, reversible, rate_law_type = cls.get_model_type_params(model_type)
        if (2 < num_species or 1 < num_reactions or
            (0 < num_reactions and num_species == 0) or
                (rate_law_type == RateLawType.product_pop and num_species != 2)):
            raise ValueError("invalid combination of num_species ({}), num_reactions ({}), rate_law_type ({})".format(
                num_species, num_reactions, rate_law_type.name))

        if num_submodels < 1:
            raise ValueError("invalid num_submodels ({})".format(num_submodels))

        # Model
        model = Model(id='test_model', name='{} with {} submodels'.format(model_type, num_submodels),
                      version='0.0.0', wc_lang_version='0.0.1')

        structure = ChemicalStructure(molecular_weight=molecular_weight, charge=charge)

        # make compartments
        default_vol = 1E-16
        init_vols = [default_vol] * num_submodels if init_vols is None else init_vols
        init_vols = np.asarray(init_vols)
        init_vol_stds = init_vols / 10. if init_vol_stds is None else np.asarray(init_vol_stds)
        if len(init_vols) != num_submodels or len(init_vol_stds) != num_submodels:
            raise ValueError("len(init_vols) ({}) or len(init_vol_stds) ({}) != num_submodels ({})".format(
                             len(init_vols), len(init_vol_stds), num_submodels))

        # make InitVolumes, which must have unique attributes for round-trip model file equality
        initial_volumes = {}
        for i in range(num_submodels):
            attributes = (init_vols[i], init_vol_stds[i])
            if attributes not in initial_volumes:
                initial_volumes[attributes] = InitVolume(mean=init_vols[i], std=init_vol_stds[i],
                                                         units=unit_registry.parse_units('l'))
        compartments = []
        for i in range(num_submodels):
            comp_num = i + 1
            init_volume = initial_volumes[(init_vols[i], init_vol_stds[i])]
            comp = model.compartments.create(id='compt_{}'.format(comp_num),
                                             name='compartment num {}'.format(comp_num),
                                             biological_type=onto['WC:cellular_compartment'],
                                             init_volume=init_volume)
            comp.init_density = model.parameters.create(id='density_compt_{}'.format(comp_num),
                                                        value=density, units=unit_registry.parse_units('g l^-1'))
            compartments.append(comp)

        # make SpeciesTypes
        species_types = []
        for i in range(num_species):
            spec_type = model.species_types.create(
                id='spec_type_{}'.format(i),
                type=onto['WC:protein'],  # protein
                structure=structure)
            species_types.append(spec_type)

        # make submodels
        expressions = {}
        for i in range(num_submodels):
            submodel_num = i + 1
            cls.add_test_submodel(model, model_type, submodel_num, compartments[i],
                                  species_types, default_species_copy_number=default_species_copy_number,
                                  default_species_std=default_species_std,
                                  species_copy_numbers=species_copy_numbers, species_stds=species_stds,
                                  expressions=expressions)

        if transform_prep_and_check:
            # prepare & check the model
            PrepForWcSimTransform().run(model)
            errors = Validator().run(model)
            if errors:
                raise ValueError(indent_forest(['The model is invalid:', [errors]]))

        # create Manager indices
        # TODO(Arthur): should be automated in a finalize() method
        for base_model in [Submodel,  SpeciesType, Reaction, Observable, Compartment, Parameter]:
            base_model.get_manager().insert_all_new()

        return model
