'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-04-10
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import numpy
import os
import unittest
import wc_lang
from argparse import Namespace
from obj_tables import utils
from scipy.constants import Avogadro
from wc_lang.io import Reader
from wc_sim.model_utilities import ModelUtilities
from wc_onto import onto
from wc_utils.util.units import unit_registry


class TestModelUtilities(unittest.TestCase):

    def get_submodel(self, model, id_val):
        return model.submodels.get_one(id=id_val)

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # read a model
        self.model = Reader().run(self.MODEL_FILENAME, ignore_extra_models=True)[wc_lang.Model][0]

    def test_find_private_species(self):
        # since set() operations are being used, this test does not ensure that the methods being
        # tested are deterministic and repeatable; repeatability should be tested by running the
        # code under different circumstances and ensuring identical output
        private_species = ModelUtilities.find_private_species(self.model)

        submodel_1_exp_species_ids = ['species_1[c]', 'species_1[e]', 'species_2[e]']
        mod1_expected_species = wc_lang.Species.get(submodel_1_exp_species_ids, self.model.get_species())
        self.assertEqual(set(private_species[self.get_submodel(self.model, 'submodel_1')]),
                         set(mod1_expected_species))

        submodel_2_exp_species_ids = ['species_5[c]', 'species_4[c]', 'species_6[c]']
        mod2_expected_species = wc_lang.Species.get(submodel_2_exp_species_ids, self.model.get_species())
        self.assertEqual(set(private_species[self.get_submodel(self.model, 'submodel_2')]),
                         set(mod2_expected_species))

        private_species = ModelUtilities.find_private_species(self.model, return_ids=True)
        self.assertEqual(set(private_species['submodel_1']), set(submodel_1_exp_species_ids))
        self.assertEqual(set(private_species['submodel_2']), set(submodel_2_exp_species_ids))

    def test_find_shared_species(self):
        self.assertEqual(set(ModelUtilities.find_shared_species(self.model)),
                         set(wc_lang.Species.get(['species_2[c]', 'species_3[c]', 'H2O[e]', 'H2O[c]'],
                                                 self.model.get_species())))

        self.assertEqual(set(ModelUtilities.find_shared_species(self.model, return_ids=True)),
                         set(['species_2[c]', 'species_3[c]', 'H2O[e]', 'H2O[c]']))

    def test_sample_copy_num_from_concentration(self):
        model = wc_lang.Model()

        submodel = model.submodels.create(id='submodel', framework=onto['WC:stochastic_simulation_algorithm'])

        compartment_c = model.compartments.create(id='c', init_volume=wc_lang.InitVolume(mean=1.))

        structure = wc_lang.ChemicalStructure(molecular_weight=10.)

        species_types = {}
        cus_species_types = {}
        for cu in wc_lang.DistributionInitConcentration.units.choices:
            id = str(cu).replace(' ', '_')
            species_types[id] = model.species_types.create(id=id, structure=structure)
            cus_species_types[id] = cu

        for other in ['no_units', 'no_concentration', 'no_std']:
            species_types[other] = model.species_types.create(id=other, structure=structure)

        species = {}
        for key, species_type in species_types.items():
            species[key] = wc_lang.Species(species_type=species_type,
                                           compartment=compartment_c)
            species[key].id = species[key].gen_id()

        conc_value = 2_000.
        std_value = 0.
        for key, sp in species.items():
            if key in cus_species_types:
                wc_lang.DistributionInitConcentration(species=sp, mean=conc_value, std=std_value,
                                                      units=cus_species_types[key])
            elif key == 'no_units':
                wc_lang.DistributionInitConcentration(species=sp, mean=conc_value, std=std_value)
            elif key == 'no_std':
                wc_lang.DistributionInitConcentration(species=sp, mean=conc_value, std=float('NaN'),
                                                      units=cus_species_types['molecule'])
            elif key == 'no_concentration':
                continue

        conc_to_molecules = ModelUtilities.sample_copy_num_from_concentration
        random_state = numpy.random.RandomState()
        copy_number = conc_to_molecules(species['molecule'], species['molecule'].compartment.init_volume.mean, random_state)
        self.assertEqual(copy_number, conc_value)
        copy_number = conc_to_molecules(species['molar'], species['molar'].compartment.init_volume.mean, random_state)
        self.assertEqual(copy_number, conc_value * Avogadro)
        copy_number = conc_to_molecules(species['no_units'], species['no_units'].compartment.init_volume.mean, random_state)
        self.assertEqual(copy_number, conc_value * Avogadro)
        copy_number = conc_to_molecules(species['millimolar'], species['millimolar'].compartment.init_volume.mean, random_state)
        self.assertEqual(copy_number, 10**-3 * conc_value * Avogadro)
        copy_number = conc_to_molecules(species['micromolar'], species['micromolar'].compartment.init_volume.mean, random_state)
        self.assertEqual(copy_number, 10**-6 * conc_value * Avogadro)
        copy_number = conc_to_molecules(species['nanomolar'], species['nanomolar'].compartment.init_volume.mean, random_state)
        self.assertEqual(copy_number, 10**-9 * conc_value * Avogadro)
        copy_number = conc_to_molecules(species['picomolar'], species['picomolar'].compartment.init_volume.mean, random_state)
        self.assertEqual(copy_number, 10**-12 * conc_value * Avogadro)
        copy_number = conc_to_molecules(species['femtomolar'], species['femtomolar'].compartment.init_volume.mean, random_state)
        self.assertAlmostEqual(copy_number, 10**-15 * conc_value * Avogadro, delta=1)
        copy_number = conc_to_molecules(species['attomolar'], species['attomolar'].compartment.init_volume.mean, random_state)
        self.assertAlmostEqual(copy_number, 10**-18 * conc_value * Avogadro, delta=1)
        copy_number = conc_to_molecules(species['no_concentration'],
                                        species['no_concentration'].compartment.init_volume.mean, random_state)
        self.assertEqual(copy_number, 0)
        conc = species['no_std'].distribution_init_concentration
        copy_number = conc_to_molecules(species['no_std'], species['no_std'].compartment.init_volume.mean, random_state)
        self.assertNotEqual(copy_number, conc_value)

        with self.assertRaises(KeyError):
            conc_to_molecules(species['mol dm^-2'], species['no_concentration'].compartment.init_volume.mean,
                              random_state)

        species_tmp = wc_lang.Species(species_type=species_type,
                                      compartment=compartment_c)
        species_tmp.id = species_tmp.gen_id()
        wc_lang.DistributionInitConcentration(species=species_tmp, mean=conc_value, std=std_value,
                                              units='not type(unit_registry.Unit)')
        with self.assertRaisesRegex(ValueError, 'Unsupported unit type'):
            conc_to_molecules(species_tmp, species_tmp.compartment.init_volume.mean, random_state)

        species_tmp2 = wc_lang.Species(species_type=species_type,
                                       compartment=compartment_c)
        species_tmp2.id = species_tmp2.gen_id()
        wc_lang.DistributionInitConcentration(species=species_tmp2, mean=conc_value, std=std_value,
                                              units=wc_lang.InitVolume.units.choices[0])
        with self.assertRaisesRegex(ValueError, 'Unsupported unit'):
            conc_to_molecules(species_tmp2, species_tmp2.compartment.init_volume.mean, random_state)

    def test_get_species_types(self):
        self.assertEqual(ModelUtilities.get_species_types([]), [])

        species_type_ids = [species_type.id for species_type in self.model.get_species_types()]
        species_ids = [species.serialize() for species in self.model.get_species()]
        self.assertEqual(sorted(ModelUtilities.get_species_types(species_ids)), sorted(species_type_ids))

    def test_get_species_types(self):
        self.assertEqual(ModelUtilities.parse_species_id('species_type_id[compartment_id]'),
                         ('species_type_id', 'compartment_id'))
        with self.assertRaisesRegex(ValueError, 'Species id format should be'):
            ModelUtilities.parse_species_id('compartment_id]')
        with self.assertRaisesRegex(ValueError, 'Species id format should be'):
            ModelUtilities.parse_species_id('[compartment_id]')
        with self.assertRaisesRegex(ValueError, 'Species id format should be'):
            ModelUtilities.parse_species_id('species_type_id[]')
        with self.assertRaisesRegex(ValueError, 'Species id format should be'):
            ModelUtilities.parse_species_id('species_type_id[compartment_id]extra')
