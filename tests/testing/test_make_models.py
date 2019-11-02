"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-02-08
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

from numpy.random import RandomState
from scipy.constants import Avogadro
import numpy as np
import os
import shutil
import tempfile
import unittest

from wc_lang import Model, RateLawDirection, InitVolume
from wc_lang.io import Reader, Writer
from wc_onto import onto
from wc_sim.testing.make_models import MakeModel, RateLawType
from wc_sim.model_utilities import ModelUtilities
from wc_utils.util.units import unit_registry


class TestMakeModels(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_types = ['no reactions',
                            '1 species, 1 reaction',
                            '2 species, 1 reaction',
                            '2 species, a pair of symmetrical reactions with constant rates',
                            '2 species, a pair of symmetrical reactions rates given by reactant population',
                            '2 species, a pair of symmetrical reactions rates given by product population',
                            ]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_make_test_model(self):
        '''
        Simple SSA model tests:
            no reactions: simulation terminates immediately
            1 species:
                one reaction consume specie, at constant rate: consume all reactant, in time given by rate
            2 species:
                one reaction: convert all reactant into product, in time given by rate
                a pair of symmetrical reactions with constant rates: maintain steady state, on average
                a pair of symmetrical reactions rates given by reactant population: maintain steady state, on average
                a pair of symmetrical reactions rates given by product population: exhaust on species,
                    with equal chance for each species
            ** ring of futile reactions with balanced rates: maintain steady state, on average
        '''

        # test get_model_type_params
        expected_params_list = [
            (0, 0, False, RateLawType.constant),
            (1, 1, False, RateLawType.constant),
            (2, 1, False, RateLawType.constant),
            (2, 1, True, RateLawType.constant),
            (2, 1, True, RateLawType.reactant_pop),
            (2, 1, True, RateLawType.product_pop)
        ]
        for model_type, expected_params in zip(self.model_types, expected_params_list):
            params = MakeModel.get_model_type_params(model_type)
            self.assertEqual(params, expected_params)

        # test make_test_model
        for model_type in self.model_types:
            for num_submodels in [1, 10]:
                model = MakeModel.make_test_model(model_type, num_submodels=num_submodels,
                                                  transform_prep_and_check=False)
                self.assertEqual(model.validate(), None)

                # test round-tripping
                file = model_type.replace(' ', '_').replace(',', '')
                file = "{}_{}_submodels.xlsx".format(file, num_submodels)
                filename = os.path.join(self.test_dir, file)
                Writer().run(filename, model, data_repo_metadata=False)
                round_trip_model = Reader().run(filename)[Model][0]
                self.assertEqual(round_trip_model.validate(), None)
                self.assertTrue(round_trip_model.is_equal(model, tol=1e-8))
                self.assertEqual(model.difference(round_trip_model, tol=1e-8), '')

        # unittest one of the models made
        # TODO (ARTHUR): test with multiple submodels
        # TODO (ARTHUR): test observables
        # TODO (ARTHUR): test functions
        model = MakeModel.make_test_model(self.model_types[4])
        self.assertEqual(model.id, 'test_model')
        comp = model.compartments[0]
        self.assertEqual(comp.id, 'compt_1')
        species_type_ids = set([st.id for st in model.species_types])
        self.assertEqual(species_type_ids, set(['spec_type_0', 'spec_type_1']))
        species_ids = set([s.id for s in comp.species])
        self.assertEqual(species_ids, set(['spec_type_0[compt_1]', 'spec_type_1[compt_1]']))
        submodel = model.submodels[0]

        # reaction was split by SplitReversibleReactionsTransform
        ratelaw_elements = set()
        for r in submodel.reactions:
            self.assertFalse(r.reversible)
            rl = r.rate_laws[0]
            ratelaw_elements.add((rl.direction, rl.expression.expression))
        expected_rate_laws = set([
            # direction, expression.expression
            (RateLawDirection.forward, 'k_cat_1_1_for * spec_type_0[compt_1] / Avogadro / volume_compt_1'),   # forward
            (RateLawDirection.forward, 'k_cat_1_1_bck * spec_type_1[compt_1] / Avogadro / volume_compt_1'),   # backward, but reversed
        ])
        self.assertEqual(ratelaw_elements, expected_rate_laws)

        participant_elements = set()
        for r in submodel.reactions:
            r_list = []
            for part in r.participants:
                r_list.append((part.species.id, part.coefficient))
            participant_elements.add(tuple(sorted(r_list)))
        expected_participants = set([
            # id, coefficient
            tuple(sorted((('spec_type_0[compt_1]', -1), ('spec_type_1[compt_1]',  1)))),    # forward
            tuple(sorted((('spec_type_1[compt_1]', -1), ('spec_type_0[compt_1]',  1)))),    # reversed
        ])
        self.assertEqual(participant_elements, expected_participants)

        # test exceptions
        with self.assertRaises(ValueError):
            MakeModel.make_test_model('3 reactions')

        with self.assertRaises(ValueError):
            MakeModel.make_test_model(self.model_types[0], num_submodels=0)

    def get_volume(self, compartment):
        # sample the volume in compartment
        mean = compartment.init_volume.mean
        std = compartment.init_volume.std
        volume = max(0., RandomState().normal(mean, std))
        return volume

    def get_cn(self, model, species):
        volume = self.get_volume(model.compartments[0])
        dist_conc = species.distribution_init_concentration
        mean = dist_conc.mean
        std = dist_conc.std
        return ModelUtilities.sample_copy_num_from_concentration(species, volume, RandomState())

    def test_make_test_model_init_vols(self):
        # test the volume settings in make_test_model
        # no vol arguments
        default_vol = 1E-16
        volumes = []
        for _ in range(10):
            model = MakeModel.make_test_model('no reactions')
            volumes.append(self.get_volume(model.compartments[0]))
        self.assertTrue(np.abs((np.mean(volumes) - default_vol) / default_vol) < 0.1)

        # just init_vols
        specified_vol = 5E-16
        volumes = []
        for _ in range(10):
            model = MakeModel.make_test_model('no reactions', init_vols=[specified_vol])
            volumes.append(self.get_volume(model.compartments[0]))
        self.assertTrue(np.abs((np.mean(volumes) - specified_vol) / specified_vol) < 0.1)

        # just init_vol_stds
        volumes = []
        for _ in range(10):
            model = MakeModel.make_test_model('no reactions', init_vol_stds=[default_vol/10.])
            volumes.append(self.get_volume(model.compartments[0]))
        self.assertTrue(np.abs((np.mean(volumes) - default_vol) / default_vol) < 0.1)

        # no variance
        model = MakeModel.make_test_model('no reactions', init_vol_stds=[0])
        self.assertEqual(self.get_volume(model.compartments[0]), default_vol)

        # both init_vols and init_vol_stds
        volumes = []
        for _ in range(10):
            model = MakeModel.make_test_model('no reactions',
                                              init_vols=[default_vol],
                                              init_vol_stds=[default_vol/10.])
            volumes.append(self.get_volume(model.compartments[0]))
        self.assertTrue(np.abs((np.mean(volumes) - default_vol) / default_vol) < 0.1)

        # raise init_vols or init_vol_stds exception
        with self.assertRaises(ValueError):
            MakeModel.make_test_model('no reactions', init_vols=[default_vol, default_vol])

        with self.assertRaises(ValueError):
            MakeModel.make_test_model('no reactions', init_vol_stds=[])

    def setup_submodel_params(self, model_type, num_species, vol_mean, vol_std):
        # make Model
        model = Model(id='test_model', name=model_type,
                      version='0.0.0', wc_lang_version='0.0.1')
        # make SpeciesTypes
        species_types = []
        for i in range(num_species):
            spec_type = model.species_types.create(id='spec_type_{}'.format(i))
            species_types.append(spec_type)
        initial_volume = InitVolume(mean=vol_mean, std=vol_std, units=unit_registry.parse_units('l'))
        comp = model.compartments.create(id='compt_1',
                                         biological_type=onto['WC:cellular_compartment'],
                                         init_volume=initial_volume)
        comp.init_density = model.parameters.create(id='density_compt_1',
                                                    value=1100, units=unit_registry.parse_units('g l^-1'))
        return (model, species_types, comp)

    def test_add_test_submodel(self):
        # test the concentration initialization in add_test_submodel
        model_type = '1 species, 1 reaction'
        num_species = 1
        vol_mean, vol_std = 1E-16, 1E-17
        specie_id = 'spec_type_0[compt_1]'
        model, species_types, comp = self.setup_submodel_params(model_type, num_species, vol_mean, vol_std)
        default_species_copy_number, default_species_std = 1_000_000, 100_000
        species_copy_numbers, species_stds = {}, {}

        # specie not in species_copy_numbers or species_stds
        MakeModel.add_test_submodel(model, model_type, 0, comp, species_types,
                                    default_species_copy_number,
                                    default_species_std,
                                    species_copy_numbers,
                                    species_stds, {})
        conc = model.get_distribution_init_concentrations()[0]
        expected_mean_conc = MakeModel.convert_pop_conc(default_species_copy_number, vol_mean)
        expected_std_conc = MakeModel.convert_pop_conc(default_species_std, vol_mean)
        self.assertEqual(conc.mean, expected_mean_conc)
        self.assertEqual(conc.std, expected_std_conc)

        # specie only in species_copy_numbers
        model, species_types, comp = self.setup_submodel_params(model_type, num_species, vol_mean, vol_std)
        species_copy_number = 1_000_000
        species_copy_numbers, species_stds = {specie_id: species_copy_number}, {}
        MakeModel.add_test_submodel(model, model_type, 0, comp, species_types,
                                    default_species_copy_number,
                                    default_species_std,
                                    species_copy_numbers,
                                    species_stds, {})
        conc = model.get_distribution_init_concentrations()[0]
        expected_mean_conc = MakeModel.convert_pop_conc(species_copy_number, vol_mean)
        expected_std_conc = MakeModel.convert_pop_conc(default_species_std, vol_mean)
        self.assertEqual(conc.mean, expected_mean_conc)
        self.assertEqual(conc.std, expected_std_conc)

        # specie only in species_stds
        model, species_types, comp = self.setup_submodel_params(model_type, num_species, vol_mean, vol_std)
        species_std = 100_000
        species_copy_numbers, species_stds = {}, {specie_id: species_std}
        MakeModel.add_test_submodel(model, model_type, 0, comp, species_types,
                                    default_species_copy_number,
                                    default_species_std,
                                    species_copy_numbers,
                                    species_stds, {})
        conc = model.get_distribution_init_concentrations()[0]
        expected_mean_conc = MakeModel.convert_pop_conc(default_species_copy_number, vol_mean)
        expected_std_conc = MakeModel.convert_pop_conc(default_species_std, vol_mean)
        self.assertEqual(conc.mean, expected_mean_conc)
        self.assertEqual(conc.std, expected_std_conc)

        # specie in species_copy_numbers and species_stds
        model, species_types, comp = self.setup_submodel_params(model_type, num_species, vol_mean, vol_std)
        species_copy_numbers, species_stds = {specie_id: species_copy_number}, {specie_id: species_std}
        MakeModel.add_test_submodel(model, model_type, 0, comp, species_types,
                                    default_species_copy_number,
                                    default_species_std,
                                    species_copy_numbers,
                                    species_stds, {})
        conc = model.get_distribution_init_concentrations()[0]
        expected_mean_conc = MakeModel.convert_pop_conc(species_copy_number, vol_mean)
        expected_std_conc = MakeModel.convert_pop_conc(species_std, vol_mean)
        self.assertEqual(conc.mean, expected_mean_conc)
        self.assertEqual(conc.std, expected_std_conc)
