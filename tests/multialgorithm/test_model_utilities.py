'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-04-10
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import os
import unittest
from argparse import Namespace
from scipy.constants import Avogadro

import wc_lang
from obj_model import utils
from wc_lang.io import Reader
from wc_sim.multialgorithm.model_utilities import ModelUtilities


class TestModelUtilities(unittest.TestCase):

    def get_submodel(self, model, id_val):
        return model.submodels.get_one(id=id_val)

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # read a model
        self.model = Reader().run(self.MODEL_FILENAME, strict=False)

    def test_find_private_species(self):
        # since set() operations are being used, this test does not ensure that the methods being
        # tested are deterministic and repeatable; repeatability should be tested by running the
        # code under different circumstances and ensuring identical output
        private_species = ModelUtilities.find_private_species(self.model)

        mod1_expected_species_ids = ['specie_1[c]', 'specie_1[e]', 'specie_2[e]']
        mod1_expected_species = wc_lang.Species.get(mod1_expected_species_ids, self.model.get_species())
        self.assertEqual(set(private_species[self.get_submodel(self.model, 'submodel_1')]),
            set(mod1_expected_species))

        mod2_expected_species_ids = ['specie_4[c]', 'specie_5[c]', 'specie_6[c]']
        mod2_expected_species = wc_lang.Species.get(mod2_expected_species_ids, self.model.get_species())
        self.assertEqual(set(private_species[self.get_submodel(self.model, 'submodel_2')]),
            set(mod2_expected_species))

        private_species = ModelUtilities.find_private_species(self.model, return_ids=True)
        self.assertEqual(set(private_species['submodel_1']), set(mod1_expected_species_ids))
        self.assertEqual(set(private_species['submodel_2']), set(mod2_expected_species_ids))

    @unittest.skip("to be fixed")
    def test_find_shared_species(self):
        self.assertEqual(set(ModelUtilities.find_shared_species(self.model)),
            set(wc_lang.Species.get(['specie_2[c]', 'specie_3[c]'], self.model.get_species())))

        self.assertEqual(set(ModelUtilities.find_shared_species(self.model, return_ids=True)),
            set(['specie_2[c]', 'specie_3[c]']))

    def test_get_initial_specie_concentrations(self):
        initial_specie_concentrations = ModelUtilities.initial_specie_concentrations(self.model)
        some_specie_concentrations = {
            'specie_2[e]':2.0E-4,
            'specie_2[c]':5.0E-4 }
        for k in some_specie_concentrations.keys():
            self.assertEqual(initial_specie_concentrations[k], some_specie_concentrations[k])

    def test_parse_specie_id(self):
        self.assertEqual(ModelUtilities.parse_specie_id('good_id[good_compt]'), ('good_id', 'good_compt'))
        with self.assertRaises(ValueError):
            ModelUtilities.parse_specie_id('1_bad_good_id[good_compt]')
        with self.assertRaises(ValueError):
            ModelUtilities.parse_specie_id('good_id[_bad_compt]')

    def test_get_species_types(self):
        self.assertEqual(ModelUtilities.get_species_types([]), [])

        specie_type_ids = [specie_type.id for specie_type in self.model.get_species_types()]
        specie_ids = [specie.serialize() for specie in self.model.get_species()]
        self.assertEqual(sorted(ModelUtilities.get_species_types(specie_ids)), sorted(specie_type_ids))

    def test_concentration_to_molecules(self):
        model = wc_lang.Model()

        submodel = model.submodels.create(id='submodel', algorithm=wc_lang.SubmodelAlgorithm.ssa)

        compartment_c = model.compartments.create(id='c', initial_volume=1.)

        species_types = {}
        for cu in wc_lang.ConcentrationUnit:
            id = "species_type_{}".format(cu.name.replace(' ', '_'))
            species_types[cu.name] = model.species_types.create(id=id, molecular_weight=10)

        for other in ['no_units', 'no_concentration', 'no_such_concentration_unit']:
            species_types[other] = model.species_types.create(id=other, molecular_weight=10)

        species = {}
        for key,species_type in species_types.items():
            species[key] = wc_lang.Species(species_type=species_type, compartment=compartment_c)

        conc_value = 2.
        for key,specie in species.items():
            if key in wc_lang.ConcentrationUnit.__members__:
                wc_lang.Concentration(species=specie, value=conc_value, units=wc_lang.ConcentrationUnit.__members__[key].value)
            elif key == 'no_units':
                wc_lang.Concentration(species=specie, value=conc_value)
            elif key == 'no_concentration':
                continue

        copy_number = ModelUtilities.concentration_to_molecules(species['molecules'])
        self.assertEqual(copy_number, conc_value)
        copy_number = ModelUtilities.concentration_to_molecules(species['M'])
        self.assertEqual(copy_number, conc_value * Avogadro)
        copy_number = ModelUtilities.concentration_to_molecules(species['no_units'])
        self.assertEqual(copy_number, conc_value * Avogadro)
        copy_number = ModelUtilities.concentration_to_molecules(species['mM'])
        self.assertEqual(copy_number, 10**-3 * conc_value * Avogadro)
        copy_number = ModelUtilities.concentration_to_molecules(species['uM'])
        self.assertEqual(copy_number, 10**-6 * 2. * Avogadro)
        copy_number = ModelUtilities.concentration_to_molecules(species['nM'])
        self.assertEqual(copy_number, 10**-9 * 2. * Avogadro)
        copy_number = ModelUtilities.concentration_to_molecules(species['pM'])
        self.assertEqual(copy_number, 10**-12 * 2. * Avogadro)
        copy_number = ModelUtilities.concentration_to_molecules(species['fM'])
        self.assertAlmostEqual(copy_number, 10**-15 * 2. * Avogadro, delta=1)
        copy_number = ModelUtilities.concentration_to_molecules(species['aM'])
        self.assertAlmostEqual(copy_number, 10**-18 * 2. * Avogadro, delta=1)
        copy_number = ModelUtilities.concentration_to_molecules(species['no_concentration'])
        self.assertEqual(copy_number, 0)
        with self.assertRaises(ValueError):
            ModelUtilities.concentration_to_molecules(species['moles dm^-2'])

        species_tmp = wc_lang.Species(species_type=species_type, compartment=compartment_c)
        wc_lang.Concentration(species=species_tmp, value=conc_value, units='molecules')
        with self.assertRaises(ValueError):
            ModelUtilities.concentration_to_molecules(species_tmp)
        species_tmp2 = wc_lang.Species(species_type=species_type, compartment=compartment_c)
        wc_lang.Concentration(species=species_tmp2, value=conc_value, units=0)
        with self.assertRaises(ValueError):
            ModelUtilities.concentration_to_molecules(species_tmp2)
