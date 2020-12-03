"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2016-10-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import unittest

from de_sim.simulation_config import SimulationConfig
from wc_lang import Species
from wc_sim.multialgorithm_errors import DynamicFrozenSimulationError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.submodels.ssa import SsaSubmodel
from wc_sim.testing.make_models import MakeModel


class TestSsaSubmodel(unittest.TestCase):

    def make_ssa_submodel(self, model, default_center_of_mass=None):
        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        multialgorithm_simulation.build_simulation()
        # todo: don't call SsaSubmodel(); return dynamic_model.dynamic_submodels['submodel name here'] will work; see test_nrm.py
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
        species = ['spec_type_0[compt_1]', 'spec_type_1[compt_1]']
        species_copy_numbers = dict(zip(species, [spec_type_0_cn, 2 * spec_type_0_cn]))
        species_stds = dict(zip(species, [0]*2))
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
            init_vol_stds=[0],
            species_copy_numbers=species_copy_numbers, species_stds=species_stds)
        ssa_submodel = self.make_ssa_submodel(model)
        propensities, _ = ssa_submodel.determine_reaction_propensities()
        self.assertEqual(2*propensities[0], propensities[1])
        ssa_submodel.execute_SSA_reaction(0)
        expected_population = dict(zip(species, [spec_type_0_cn-1, 2*spec_type_0_cn+1]))
        population = ssa_submodel.local_species_population.read(0, set(species))
        self.assertEqual(population, expected_population)

        # test determine_reaction_propensities() exception
        model = MakeModel.make_test_model('2 species, 1 reaction', default_species_std=0)
        # set rxn rate constant to 0 so that reaction propensities are 0
        rl_constant = model.parameters.get_or_create(id='k_cat_1_1_for')
        rl_constant.value = 0
        ssa_submodel = self.make_ssa_submodel(model)
        with self.assertRaisesRegex(DynamicFrozenSimulationError,
                                    "propensities == 0 won't change species populations"):
            ssa_submodel.determine_reaction_propensities()
