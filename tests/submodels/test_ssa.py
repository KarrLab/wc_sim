"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2016-10-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from wc_lang import Species
from wc_sim.make_models import MakeModel
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.submodels.ssa import SsaSubmodel
import unittest


class TestSSaExceptions(unittest.TestCase):

    def setUp(self):
        self.model = \
            MakeModel.make_test_model('2 species, 1 reaction, with rates given by reactant population',
                                      species_copy_numbers={'spec_type_0[compt_1]': 10, 'spec_type_1[compt_1]': 10},
                                      species_stds={'spec_type_0[compt_1]': 0, 'spec_type_1[compt_1]': 0})

    def test_nan_propensities(self):
        st_0 = self.model.species_types.get_one(id='spec_type_0')
        st_0.structure.molecular_weight = float('NaN')
        multialgorithm_simulation = MultialgorithmSimulation(self.model, {})
        simulation_engine, _ = multialgorithm_simulation.build_simulation()
        with self.assertRaisesRegex(AssertionError, "total propensities is 'NaN'"):
            simulation_engine.initialize()


class TestSsaSubmodel(unittest.TestCase):

    def make_ssa_submodel(self, model, default_center_of_mass=None):
        multialgorithm_simulation = MultialgorithmSimulation(model, None)
        multialgorithm_simulation.build_simulation()
        wc_lang_ssa_submodel = model.submodels[0]

        ssa_submodel = SsaSubmodel(
            model.id,
            multialgorithm_simulation.dynamic_model,
            list(wc_lang_ssa_submodel.reactions),
            wc_lang_ssa_submodel.get_children(kind='submodel', __type=Species),
            multialgorithm_simulation.get_dynamic_compartments(wc_lang_ssa_submodel),
            multialgorithm_simulation.local_species_population,
            default_center_of_mass=default_center_of_mass)
        return ssa_submodel

    def test_SSA_submodel_init(self):
        model = MakeModel.make_test_model('1 species, 1 reaction')
        ssa_submodel = self.make_ssa_submodel(model, default_center_of_mass=20)
        self.assertTrue(isinstance(ssa_submodel, SsaSubmodel))

    def test_static_SSA_submodel_methods(self):
        # static tests of ssa methods
        model = MakeModel.make_test_model('1 species, 1 reaction')
        ssa_submodel = self.make_ssa_submodel(model)
        self.assertEqual(ssa_submodel.num_SsaWaits, 0)
        self.assertTrue(0 < ssa_submodel.ema_of_inter_event_time.get_ema())
        propensities, total_propensities = ssa_submodel.determine_reaction_propensities()
        # there's only one reaction
        self.assertEqual(propensities[0], total_propensities)

        spec_type_0_cn = 1000000
        species_copy_numbers = {
            'spec_type_0[compt_1]': spec_type_0_cn,
            'spec_type_1[compt_1]': 2 * spec_type_0_cn
        }
        species_stds = {
            'spec_type_0[compt_1]': 0,
            'spec_type_1[compt_1]': 0,
        }
        # with constant reaction rates, all propensities are equal
        model = MakeModel.make_test_model(
            '2 species, a pair of symmetrical reactions with constant rates',
            species_copy_numbers=species_copy_numbers, species_stds=species_stds)
        ssa_submodel = self.make_ssa_submodel(model)
        propensities, _ = ssa_submodel.determine_reaction_propensities()
        self.assertEqual(propensities[0], propensities[1])

        # with rates given by reactant population, propensities proportional to copy number
        model = MakeModel.make_test_model(
            '2 species, a pair of symmetrical reactions rates given by reactant population',
            species_copy_numbers=species_copy_numbers, species_stds=species_stds)
        ssa_submodel = self.make_ssa_submodel(model)
        propensities, _ = ssa_submodel.determine_reaction_propensities()
        self.assertEqual(2*propensities[0], propensities[1])
        ssa_submodel.execute_SSA_reaction(0)
        population = ssa_submodel.local_species_population.read(0,
                                                                set(['spec_type_0[compt_1]', 'spec_type_1[compt_1]']))
        expected_population = {
            'spec_type_0[compt_1]': spec_type_0_cn-1,
            'spec_type_1[compt_1]': 2*spec_type_0_cn+1
        }
        self.assertEqual(population, expected_population)
