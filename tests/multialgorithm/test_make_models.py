"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

import unittest
import os
import shutil
import tempfile

from wc_lang.io import Writer, Reader
from wc_lang.core import RateLawDirection
from wc_sim.multialgorithm.make_models import MakeModels, RateLawType


class TestMakeModels(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

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
                a pair of symmetrical reactions rates given by product population: exhaust on species, with equal chance for each species
            ** ring of futile reactions with balanced rates: maintain steady state, on average
        '''
        model_types = ['no reactions',
            '1 species, 1 reaction',
            '2 species, 1 reaction',
            '2 species, a pair of symmetrical reactions with constant rates',
            '2 species, a pair of symmetrical reactions rates given by reactant population',
            '2 species, a pair of symmetrical reactions rates given by product population',
        ]

        # test get_model_type_params
        expected_params_list = [
            (0, 0, False, RateLawType.constant),
            (1, 1, False, RateLawType.constant),
            (2, 1, False, RateLawType.constant),
            (2, 1, True, RateLawType.constant),
            (2, 1, True, RateLawType.reactant_pop),
            (2, 1, True, RateLawType.product_pop)
        ]
        for model_type,expected_params in zip(model_types, expected_params_list):
            params = MakeModels.get_model_type_params(model_type)
            self.assertEqual(params, expected_params)

        # test make_test_model
        for model_type in model_types:
            model = MakeModels.make_test_model(model_type, transform_prep_and_check=False)
            self.assertEqual(model.validate(), None)

            # test round tripping
            file = model_type.replace(' ', '_').replace(',', '')
            filename = os.path.join(self.test_dir, file+'.xlsx')
            Writer().run(model, filename, set_repo_metadata_from_path=False)
            round_trip_model = Reader().run(filename)
            self.assertEqual(round_trip_model.validate(), None)
            self.assertTrue(round_trip_model.is_equal(model))
            self.assertEqual(model.difference(round_trip_model), '')

        # unittest one of the models made
        model = MakeModels.make_test_model(model_types[4])
        self.assertEqual(model.id, 'test_model')
        comp = model.compartments[0]
        self.assertEqual(comp.id, 'c')
        species_type_ids = set([st.id for st in model.species_types])
        self.assertEqual(species_type_ids, set(['spec_type_0', 'spec_type_1']))
        species_ids = set([s.id() for s in comp.species])
        self.assertEqual(species_ids, set(['spec_type_0[c]', 'spec_type_1[c]']))
        submodel = model.submodels[0]
        self.assertEqual(model.submodels[0].compartment, comp)

        # reaction was split by SplitReversibleReactionsTransform
        ratelaw_elements = set()
        for r in submodel.reactions:
            self.assertFalse(r.reversible)
            rl = r.rate_laws[0]
            ratelaw_elements.add((rl.direction, rl.equation.expression))
        expected_rate_laws = set([
            # direction, equation expression
            (RateLawDirection.forward, 'spec_type_0[c]'),   # forward
            (RateLawDirection.forward, 'spec_type_1[c]'),   # backward, but reversed
        ])
        self.assertEqual(ratelaw_elements, expected_rate_laws)

        participant_elements = set()
        for r in submodel.reactions:
            r_list = []
            for part in r.participants:
                r_list.append((part.species.id(), part.coefficient))
            participant_elements.add(tuple(sorted(r_list)))
        expected_participants = set([
            # id, coefficient
            tuple(sorted((('spec_type_0[c]', -1), ('spec_type_1[c]',  1)))),    # forward
            tuple(sorted((('spec_type_1[c]', -1), ('spec_type_0[c]',  1)))),    # reversed
        ])
        self.assertEqual(participant_elements, expected_participants)
        self.assertIn('fractionDryWeight', [p.id for p in model.get_parameters()])

        # test default_specie_copy_number, specie_copy_numbers, and init_vol
        default_cn = 2000000
        spec_type_0_cn = 100000
        init_vol = 1E-13
        model = MakeModels.make_test_model(model_types[4], default_specie_copy_number=default_cn,
            specie_copy_numbers={'spec_type_0[c]':spec_type_0_cn}, init_vol=init_vol)
        concentrations = []
        for concentration in model.get_concentrations():
            concentrations.append((concentration.species.species_type.id, concentration.value))
        concentrations = tuple(concentrations)
        expected_concentrations = (
            ('spec_type_0', MakeModels.convert_pop_conc(spec_type_0_cn, init_vol)),
            ('spec_type_1', MakeModels.convert_pop_conc(default_cn, init_vol)),
        )
        self.assertEqual(concentrations, expected_concentrations)

        # test exception
        with self.assertRaises(ValueError):
            MakeModels.make_test_model('3 reactions')
