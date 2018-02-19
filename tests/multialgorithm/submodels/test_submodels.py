"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import unittest
import os, sys
import six
import numpy as np
from builtins import super
import warnings

from scipy.constants import Avogadro

from wc_lang.io import Reader
from wc_lang.core import Reaction, SpeciesType, Species
from wc_lang.prepare import PrepareModel, CheckModel
from wc_lang.transform import SplitReversibleReactionsTransform
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.species_populations import LocalSpeciesPopulation
from wc_sim.multialgorithm import message_types, distributed_properties
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.submodels.skeleton_submodel import SkeletonSubmodel
from obj_model.utils import get_component_by_id

def prepare_model(model):
    SplitReversibleReactionsTransform().run(model)
    # TODO:(Arthur): put these in a high-level prepare
    PrepareModel(model).run()
    CheckModel(model).run()

def make_dynamic_submodel_params(model, lang_submodel):
    local_species_pop = MultialgorithmSimulation.make_local_species_pop(model)
    dynamic_compartments = MultialgorithmSimulation.create_dynamic_compartments_for_submodel(
        model,
        lang_submodel,
        local_species_pop)
    return (lang_submodel.id,
            lang_submodel.reactions,
            lang_submodel.get_species(),
            model.get_parameters(),
            dynamic_compartments,
            local_species_pop)


class TestDynamicSubmodel(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        SpeciesType.objects.reset()

        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
            'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME)
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

    def test_get_specie_concentrations(self):
        expected_conc = {}
        for conc in self.model.get_concentrations():
            expected_conc[
                Species.gen_id(conc.species.species_type, conc.species.compartment)] = conc.value
        for dynamic_submodel in self.dynamic_submodels.values():
            for specie_id,v in dynamic_submodel.get_specie_concentrations().items():
                if specie_id in expected_conc:
                    self.assertAlmostEqual(expected_conc[specie_id], v)
        for dynamic_submodel in self.misconfigured_dynamic_submodels.values():
            with self.assertRaises(MultialgorithmError) as context:
                dynamic_submodel.get_specie_concentrations()
            six.assertRegex(self, str(context.exception),
                "dynamic submodel .* lacks dynamic compartment .* for specie .*")

    def test_calc_reaction_rates(self):
        expected_rates = {'reaction_2': 0.0, 'reaction_4': 2.0}
        for dynamic_submodel in self.dynamic_submodels.values():
            rates = dynamic_submodel.calc_reaction_rates()
            for index,rxn in enumerate(dynamic_submodel.reactions):
                if rxn.id in expected_rates:
                    self.assertAlmostEqual(list(rates)[index], expected_rates[rxn.id])

    expected_enabled = {
        'submodel_1': set(['reaction_1', '__exchange_reaction__1', '__exchange_reaction__2']),
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

    def test_execute_reaction(self):
        # test one reaction 'by hand'
        # reversible reactions have been split in two
        adjustments = {'reaction_3_forward': {'specie_2[c]':-1, 'specie_4[c]':-2, 'specie_5[c]':1}}
        for dynamic_submodel in self.dynamic_submodels.values():
            for reaction in dynamic_submodel.reactions:
                if reaction.id in adjustments:
                    before = dynamic_submodel.get_specie_counts()
                    dynamic_submodel.execute_reaction(reaction)
                    after = dynamic_submodel.get_specie_counts()
                    for specie_id in adjustments[reaction.id].keys():
                        self.assertEqual(adjustments[reaction.id][specie_id],
                            after[specie_id]-before[specie_id])

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
                    six.assertRegex(self, str(context.exception),
                        "dynamic submodel .* cannot execute reaction")


class TestSkeletonSubmodel(unittest.TestCase):

    def setUp(self):
        SpeciesType.objects.reset()
        warnings.simplefilter("ignore")
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
            'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME)
        prepare_model(self.model)

    def make_sim_w_skeleton_submodel(self, lang_submodel, behavior):
        # concatenate tuples in fn call for Py 2.7: see https://stackoverflow.com/a/12830036
        self.simulator = SimulationEngine()
        self.simulator.register_object_types([SkeletonSubmodel])
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
