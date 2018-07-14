"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import unittest
import os, sys
import numpy as np
import warnings
from scipy.constants import Avogadro

from wc_lang.io import Reader
from wc_lang.core import Reaction, SpeciesType, Species, Model
from wc_lang.prepare import PrepareModel, CheckModel
from wc_lang.transform import SplitReversibleReactionsTransform
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm import message_types, distributed_properties
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name
from wc_sim.multialgorithm.make_models import MakeModels
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.submodels.skeleton_submodel import SkeletonSubmodel
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from obj_model.utils import get_component_by_id

def prepare_model(model):
    SplitReversibleReactionsTransform().run(model)
    # TODO:(Arthur): put these in a high-level prepare
    PrepareModel(model).run()
    CheckModel(model).run()

def make_dynamic_submodel_params(model, lang_submodel):
    multialgorithm_simulation = MultialgorithmSimulation(model, None)

    return (lang_submodel.id,
            lang_submodel.reactions,
            lang_submodel.get_species(),
            model.get_parameters(),
            multialgorithm_simulation.get_dynamic_compartments(lang_submodel),
            multialgorithm_simulation.local_species_population)


class TestDynamicSubmodel(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
            'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME, strict=False)
        prepare_model(self.model)
        self.dynamic_submodels = {}
        self.misconfigured_dynamic_submodels = {}
        for lang_submodel in self.model.get_submodels():
            self.dynamic_submodels[lang_submodel.id] = DynamicSubmodel(
                *make_dynamic_submodel_params(self.model, lang_submodel))

            # create dynamic submodels that lack a dynamic compartment
            (id, reactions, species, parameters, dynamic_compartments, local_species_pop) = \
                make_dynamic_submodel_params(self.model, lang_submodel)
            dynamic_compartments.popitem()
            self.misconfigured_dynamic_submodels[lang_submodel.id] = DynamicSubmodel(
                id, reactions, species, parameters, dynamic_compartments, local_species_pop)

    def test_get_state(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            self.assertEqual(dynamic_submodel.get_state(), DynamicSubmodel.GET_STATE_METHOD_MESSAGE)

    def expected_molar_conc(self, dynamic_submodel, specie_id):
        species = list(filter(lambda s: s.id() == specie_id, dynamic_submodel.species))[0]
        copy_num = ModelUtilities.concentration_to_molecules(species)
        return copy_num / (species.compartment.initial_volume * Avogadro)

    def test_get_specie_concentrations(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            for specie_id,value in dynamic_submodel.get_specie_concentrations().items():
                self.assertEqual(self.expected_molar_conc(dynamic_submodel, specie_id), value)

        for dynamic_submodel in self.misconfigured_dynamic_submodels.values():
            with self.assertRaises(MultialgorithmError) as context:
                dynamic_submodel.get_specie_concentrations()
            self.assertRegex(str(context.exception),
                "dynamic submodel .* lacks dynamic compartment .* for specie .*")

        # test volume=0 exception; must create model with 0<mass and then decrease counts
        model = MakeModels().make_test_model('1 species, 1 reaction',
            specie_copy_numbers={'spec_type_0[compt_1]': 1})
        dynamic_submodel = DynamicSubmodel(*make_dynamic_submodel_params(model, model.submodels[0]))
        dynamic_submodel.local_species_population.adjust_discretely(0, {'spec_type_0[compt_1]': -1})
        with self.assertRaises(MultialgorithmError) as context:
            dynamic_submodel.get_specie_concentrations()
        self.assertRegex(str(context.exception),
            "dynamic submodel .* cannot compute concentration in compartment .* with volume=0")

    def test_calc_reaction_rates(self):
        expected_rates = {'reaction_2': 0.0, 'reaction_4': 2.0}
        for dynamic_submodel in self.dynamic_submodels.values():
            rates = dynamic_submodel.calc_reaction_rates()
            for index,rxn in enumerate(dynamic_submodel.reactions):
                if rxn.id in expected_rates:
                    self.assertAlmostEqual(list(rates)[index], expected_rates[rxn.id])

    expected_enabled = {
        'submodel_1': set(['reaction_1', 'reaction_2', '__exchange_reaction__1', '__exchange_reaction__2']),
        'submodel_2': set(['reaction_4', 'reaction_3_forward', 'reaction_3_backward']),
    }
    def test_enabled_reaction(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            enabled = set()
            for rxn in dynamic_submodel.reactions:
                if dynamic_submodel.enabled_reaction(rxn):
                    enabled.add(rxn.id)
            self.assertEqual(TestDynamicSubmodel.expected_enabled[dynamic_submodel.id], enabled)

    def test_identify_enabled_reactions(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            expected_set = TestDynamicSubmodel.expected_enabled[dynamic_submodel.id]
            expected_array =\
                np.asarray([r.id in expected_set for r in dynamic_submodel.reactions]).astype(int)
            enabled = dynamic_submodel.identify_enabled_reactions()
            self.assertTrue(np.array_equal(enabled, expected_array))

    def test_execute_disabled_reactions(self):
        # test exception by executing reactions that aren't enabled
        enabled_rxn_ids = []
        for set_rxn_ids in TestDynamicSubmodel.expected_enabled.values():
            enabled_rxn_ids.extend(list(set_rxn_ids))
        enabled_reactions = [get_component_by_id(self.model.get_reactions(), rxn_id)
            for rxn_id in enabled_rxn_ids]
        for dynamic_submodel in self.dynamic_submodels.values():
            for reaction in dynamic_submodel.reactions:
                if reaction not in enabled_reactions:
                    with self.assertRaises(MultialgorithmError) as context:
                        dynamic_submodel.execute_reaction(reaction)
                    self.assertRegex(str(context.exception),
                        "dynamic submodel .* cannot execute reaction")

    def do_test_execute_reaction(self, reaction_id, expected_adjustments):
        rxn = self.model.get_component('reaction', reaction_id)
        dynamic_submodel = self.dynamic_submodels[rxn.submodel.id]
        before = dynamic_submodel.get_specie_counts()
        dynamic_submodel.execute_reaction(rxn)
        after = dynamic_submodel.get_specie_counts()
        for specie_id,change in expected_adjustments.items():
            self.assertEqual(change, after[specie_id]-before[specie_id])

    def test_execute_reaction(self):
        # test reactions 'by hand'
        # reversible reactions have been split in two
        self.do_test_execute_reaction('reaction_3_forward',
            {'specie_2[c]':-1, 'specie_4[c]':-2, 'specie_5[c]':1})
        # test reaction in which a species appears multiple times
        self.do_test_execute_reaction('reaction_2', {'specie_1[c]':1, 'specie_3[c]':1})


class TestSkeletonSubmodel(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
            'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME, strict=False)
        prepare_model(self.model)

    def make_sim_w_skeleton_submodel(self, lang_submodel, behavior):
        self.simulator = SimulationEngine()
        # concatenate tuples in fn call for Py 2.7: see https://stackoverflow.com/a/12830036
        skeleton_submodel = SkeletonSubmodel(
            *(make_dynamic_submodel_params(self.model, lang_submodel) + (behavior,)))
        self.simulator.add_object(skeleton_submodel)
        self.simulator.initialize()
        return skeleton_submodel

    def test_skeleton_submodel(self):
        behavior = {SkeletonSubmodel.INTER_REACTION_TIME: 2}
        lang_submodel = self.model.get_submodels()[0]
        end_time = 100
        skeleton_submodel = self.make_sim_w_skeleton_submodel(lang_submodel, behavior)
        self.assertEqual(self.simulator.simulate(end_time),
            end_time/behavior[SkeletonSubmodel.INTER_REACTION_TIME])

        behavior = {SkeletonSubmodel.INTER_REACTION_TIME: 2,
            SkeletonSubmodel.REACTION_TO_EXECUTE: 0}    # reaction #0 makes one more 'specie_1[c]'
        skeleton_submodel = self.make_sim_w_skeleton_submodel(lang_submodel, behavior)
        interval = 10
        pop_before = skeleton_submodel.local_species_population.read_one(0, 'specie_1[c]')
        for end_time in range(interval, 5*interval, interval):
            self.simulator.simulate(end_time)
            pop_after = skeleton_submodel.local_species_population.read_one(end_time, 'specie_1[c]')
            delta = pop_after - pop_before
            self.assertEqual(delta, interval/behavior[SkeletonSubmodel.INTER_REACTION_TIME])
            pop_before = pop_after
