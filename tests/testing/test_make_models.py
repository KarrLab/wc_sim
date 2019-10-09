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

from wc_lang import Model, RateLawDirection
from wc_lang.io import Reader, Writer
from wc_sim.testing.make_models import MakeModel, RateLawType
from wc_sim.model_utilities import ModelUtilities


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

    # TODO(Arthur): fully test wc_sim/testing/make_models.py
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

        # test default_species_copy_number, species_copy_numbers, and init_vols
        default_cn = 2000000
        spec_type_0_cn = 100000
        init_vols = [1E-13]
        model = MakeModel.make_test_model(self.model_types[4], default_species_copy_number=default_cn,
                                          species_copy_numbers={'spec_type_0[compt_1]': spec_type_0_cn},
                                          init_vols=init_vols)

        def get_concentrations(model):
            concentrations = []
            for concentration in model.get_distribution_init_concentrations():
                concentrations.append((concentration.species.species_type.id, concentration.mean))
            return tuple(concentrations)

        expected_concentrations = (
            ('spec_type_0', MakeModel.convert_pop_conc(spec_type_0_cn, init_vols[0])),
            ('spec_type_1', MakeModel.convert_pop_conc(default_cn, init_vols[0])),
        )
        self.assertEqual(get_concentrations(model), expected_concentrations)

        init_vol_stds = [0]
        model = MakeModel.make_test_model(self.model_types[4], default_species_copy_number=default_cn,
                                          species_copy_numbers={'spec_type_0[compt_1]': spec_type_0_cn},
                                          init_vols=init_vols, init_vol_stds=init_vol_stds)
        self.assertEqual(get_concentrations(model), expected_concentrations)

        # test exception
        with self.assertRaises(ValueError):
            MakeModel.make_test_model('3 reactions')

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
        print('mean, stddev', mean, std)
        return ModelUtilities.concentration_to_molecules(species, volume, RandomState())

    def test_make_test_model_init_vols(self):
        print()
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

    def test_make_test_model_species_pop(self):
        print()
        one_specie = 'spec_type_0[compt_1]'
        initial_cn = 1000000
        for i in range(5):
            model = MakeModel.make_test_model('1 species, 1 reaction',
                                              init_vol_stds=[0],
                                              species_copy_numbers={one_specie: initial_cn},
                                              species_stds={one_specie: 0})
            # get species population
            cn_one_specie = self.get_cn(model, model.species[0])
            print('cn_one_specie', cn_one_specie)
