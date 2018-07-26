"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-10-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import unittest
import os
import numpy as np

from wc_lang.core import (Model, Submodel,  SpeciesType, Species, Reaction, Compartment,
                          SpeciesCoefficient, Parameter, RateLaw, RateLawEquation,
                          SubmodelAlgorithm)
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm.submodels.ssa import SSASubmodel
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.message_types import GivePopulation, ExecuteSsaReaction
from wc_sim.multialgorithm.make_models import MakeModels, RateLawType


class TestSsaSubmodel(unittest.TestCase):

    def make_ssa_submodel(self, model, default_center_of_mass=None):
        multialgorithm_simulation = MultialgorithmSimulation(model, None)
        wc_lang_ssa_submodel = model.submodels[0]

        ssa_submodel = SSASubmodel(
            model.id,
            multialgorithm_simulation.dynamic_model,
            list(wc_lang_ssa_submodel.reactions),
            wc_lang_ssa_submodel.get_species(),
            wc_lang_ssa_submodel.parameters,
            multialgorithm_simulation.get_dynamic_compartments(wc_lang_ssa_submodel),
            multialgorithm_simulation.local_species_population,
            default_center_of_mass=default_center_of_mass)
        return ssa_submodel

    def test_SSA_submodel_init(self):
        model = MakeModels.make_test_model('1 species, 1 reaction')
        ssa_submodel = self.make_ssa_submodel(model, default_center_of_mass=20)
        self.assertTrue(isinstance(ssa_submodel, SSASubmodel))

    def test_static_SSA_submodel_methods(self):
        # static tests of ssa methods
        model = MakeModels.make_test_model('1 species, 1 reaction')
        ssa_submodel = self.make_ssa_submodel(model)
        self.assertEqual(ssa_submodel.num_SsaWaits, 0)
        self.assertTrue(0<ssa_submodel.ema_of_inter_event_time.get_ema())
        propensities, total_propensities = ssa_submodel.determine_reaction_propensities()
        # there's only one reaction
        self.assertEqual(propensities[0], total_propensities)

        spec_type_0_cn = 1000000
        specie_copy_numbers={
            'spec_type_0[compt_1]':spec_type_0_cn,
            'spec_type_1[compt_1]':2*spec_type_0_cn
        }
        # with constant reaction rates, all propensities are equal
        model = MakeModels.make_test_model(
            '2 species, a pair of symmetrical reactions with constant rates',
            specie_copy_numbers=specie_copy_numbers)
        ssa_submodel = self.make_ssa_submodel(model)
        propensities, _ = ssa_submodel.determine_reaction_propensities()
        self.assertEqual(propensities[0], propensities[1])

        # with rates given by reactant population, propensities proportional to copy number
        model = MakeModels.make_test_model(
            '2 species, a pair of symmetrical reactions rates given by reactant population',
            specie_copy_numbers=specie_copy_numbers)
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
