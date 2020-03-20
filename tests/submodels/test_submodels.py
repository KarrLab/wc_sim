"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
import numpy
import os
import unittest
import warnings

from de_sim.simulation_engine import SimulationEngine
from obj_tables.utils import get_component_by_id
from wc_lang import Model, Species, Validator
from wc_lang.io import Reader
from wc_lang.transform import PrepForWcSimTransform
from wc_onto import onto
from wc_sim.dynamic_components import DynamicModel
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.simulation import Simulation
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.submodels.testing.deterministic_simulation_algorithm import DsaSubmodel, ExecuteDsaReaction
from wc_sim.submodels.testing.skeleton_submodel import SkeletonSubmodel
from wc_sim.testing.make_models import MakeModel
from wc_utils.util.ontology import are_terms_equivalent
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.string import indent_forest


def prepare_model(model):
    PrepForWcSimTransform().run(model)
    errors = Validator().run(model)
    if errors:
        raise ValueError(indent_forest(['The model is invalid:', [errors]]))


# TODO(Arthur): make more reliable and comprehensive tests using approach in
# TestDsaSubmodel instead of make_dynamic_submodel_params
def make_dynamic_submodel_params(model, lang_submodel):
    multialgorithm_simulation = MultialgorithmSimulation(model, None)
    multialgorithm_simulation.initialize_components()
    multialgorithm_simulation.dynamic_model = \
        DynamicModel(multialgorithm_simulation.model,
                     multialgorithm_simulation.local_species_population,
                     multialgorithm_simulation.temp_dynamic_compartments)

    return (lang_submodel.id,
            multialgorithm_simulation.dynamic_model,
            lang_submodel.reactions,
            lang_submodel.get_children(kind='submodel', __type=Species),
            multialgorithm_simulation.get_dynamic_compartments(lang_submodel),
            multialgorithm_simulation.local_species_population)


class TestDynamicSubmodelStatically(unittest.TestCase):

    def setUp(self, std_init_concentrations=None):

        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME)[Model][0]
        prepare_model(self.model)

        if std_init_concentrations is not None:
            for conc in self.model.distribution_init_concentrations:
                conc.std = std_init_concentrations

        self.misconfigured_dynamic_submodels = {}
        self.dynamic_submodels = {}
        for lang_submodel in self.model.get_submodels():
            self.dynamic_submodels[lang_submodel.id] = DynamicSubmodel(
                *make_dynamic_submodel_params(self.model, lang_submodel))

            # create dynamic submodels that lack a dynamic compartment
            id, dynamic_model, reactions, species, dynamic_compartments, local_species_pop = \
                make_dynamic_submodel_params(self.model, lang_submodel)
            dynamic_compartments.popitem()
            self.misconfigured_dynamic_submodels[lang_submodel.id] = DynamicSubmodel(
                id, None, reactions, species, dynamic_compartments, local_species_pop)

    def test_get_state(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            self.assertEqual(dynamic_submodel.get_state(), DynamicSubmodel.GET_STATE_METHOD_MESSAGE)

    def test_get_num_submodels(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            self.assertEqual(dynamic_submodel.get_num_submodels(), 2)

    def expected_molar_conc(self, dynamic_submodel, species_id):
        species = list(filter(lambda s: s.id == species_id, dynamic_submodel.species))[0]
        init_volume = species.compartment.init_volume.mean
        copy_num = ModelUtilities.sample_copy_num_from_concentration(species, init_volume, RandomStateManager.instance())
        volume = dynamic_submodel.dynamic_compartments[species.compartment.id].volume()
        return copy_num / (volume * Avogadro)

    def test_calc_reaction_rates(self):
        # set standard deviation of initial conc. to 0
        self.setUp(std_init_concentrations=0.)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, {'dfba_time_step': 1})
        _, dynamic_model = multialgorithm_simulation.build_simulation()

        # rate law for reaction_4-forward: k_cat_4_for * max(species_4[c], p_4)
        k_cat_4_for = 1
        p_4 = 2
        species_4_c_pop = \
            multialgorithm_simulation.local_species_population.read_one(0, 'species_4[c]')
        expected_rate_reaction_4_forward = k_cat_4_for * max(species_4_c_pop, p_4)
        expected_rates = {
            'reaction_2': 0.0,
            'reaction_4': expected_rate_reaction_4_forward
        }
        for dynamic_submodel in multialgorithm_simulation.dynamic_model.dynamic_submodels.values():
            rates = dynamic_submodel.calc_reaction_rates()
            for index, rxn in enumerate(dynamic_submodel.reactions):
                if rxn.id in expected_rates:
                    self.assertAlmostEqual(list(rates)[index], expected_rates[rxn.id])

    expected_enabled = {
        'submodel_1': set([
            'reaction_1',
            'reaction_2',
            '__dfba_ex_submodel_1_species_1_e',
            '__dfba_ex_submodel_1_species_2_e',
        ]),
        'submodel_2': set([
            'reaction_4',
            'reaction_3_forward',
            'reaction_3_backward',
        ]),
    }

    def test_enabled_reaction(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            enabled = set()
            for rxn in dynamic_submodel.reactions:
                if dynamic_submodel.enabled_reaction(rxn):
                    enabled.add(rxn.id)
            self.assertEqual(TestDynamicSubmodelStatically.expected_enabled[dynamic_submodel.id], enabled)

    def test_identify_enabled_reactions(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            expected_set = TestDynamicSubmodelStatically.expected_enabled[dynamic_submodel.id]
            expected_array =\
                numpy.asarray([r.id in expected_set for r in dynamic_submodel.reactions]).astype(int)
            enabled = dynamic_submodel.identify_enabled_reactions()
            self.assertTrue(numpy.array_equal(enabled, expected_array))

    def test_execute_disabled_reactions(self):
        # test exception by executing reactions that aren't enabled
        enabled_rxn_ids = []
        for set_rxn_ids in TestDynamicSubmodelStatically.expected_enabled.values():
            enabled_rxn_ids.extend(list(set_rxn_ids))
        enabled_reactions = [get_component_by_id(self.model.get_reactions(), rxn_id)
                             for rxn_id in enabled_rxn_ids]
        for dynamic_submodel in self.dynamic_submodels.values():
            for reaction in dynamic_submodel.reactions:
                if reaction not in enabled_reactions:
                    with self.assertRaisesRegex(DynamicMultialgorithmError,
                                                "dynamic submodel .* cannot execute reaction"):
                        dynamic_submodel.execute_reaction(reaction)

    def do_test_execute_reaction(self, reaction_id, expected_adjustments):
        rxn = self.model.get_reactions(id=reaction_id)[0]
        dynamic_submodel = self.dynamic_submodels[rxn.submodel.id]
        before = dynamic_submodel.get_species_counts()
        dynamic_submodel.execute_reaction(rxn)
        after = dynamic_submodel.get_species_counts()
        for species_id, change in expected_adjustments.items():
            self.assertEqual(change, after[species_id] - before[species_id])

    def test_execute_reaction(self):
        # test reactions 'by hand'
        # reversible reactions have been split in two
        self.do_test_execute_reaction('reaction_3_forward',
                                      {'species_2[c]': -1, 'species_4[c]': -2, 'species_5[c]': 1})
        # test reaction in which a species appears multiple times
        self.do_test_execute_reaction('reaction_2', {'species_1[c]': 1, 'species_3[c]': 1})


class TestSkeletonSubmodel(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME)[Model][0]
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
        for conc in self.model.distribution_init_concentrations:
            conc.std = 0

        time_max = 100
        skeleton_submodel = self.make_sim_w_skeleton_submodel(lang_submodel, behavior)
        self.assertEqual(self.simulator.simulate(time_max),
                         time_max / behavior[SkeletonSubmodel.INTER_REACTION_TIME])

        behavior = {SkeletonSubmodel.INTER_REACTION_TIME: 2,
                    SkeletonSubmodel.REACTION_TO_EXECUTE: 0}    # reaction #0 makes one more 'species_1[c]'
        skeleton_submodel = self.make_sim_w_skeleton_submodel(lang_submodel, behavior)
        interval = 10
        pop_before = skeleton_submodel.local_species_population.read_one(0, 'species_1[c]')
        for time_max in range(interval, 5 * interval, interval):
            self.simulator.simulate(time_max)
            pop_after = skeleton_submodel.local_species_population.read_one(time_max, 'species_1[c]')
            delta = pop_after - pop_before
            self.assertEqual(delta, interval / behavior[SkeletonSubmodel.INTER_REACTION_TIME])
            pop_before = pop_after


class TestDsaSubmodel(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME)[Model][0]

    @staticmethod
    def transform_model_for_dsa_simulation(model):
        # change the framework of the SSA submodel to experimental deterministic simulation algorithm
        for submodel in model.submodels:
            if are_terms_equivalent(submodel.framework, onto['WC:stochastic_simulation_algorithm']):
                submodel.framework = onto['WC:deterministic_simulation_algorithm']

        # to make deterministic initial conditions, set variances of distributions to 0
        for conc in model.distribution_init_concentrations:
            conc.std = 0.
        for compartment in model.compartments:
            compartment.init_volume.std = 0.

    def test_deterministic_simulation_algorithm_submodel_statics(self):
        self.transform_model_for_dsa_simulation(self.model)
        prepare_model(self.model)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, dict(dfba_time_step=1))
        simulation_engine, _ = multialgorithm_simulation.build_simulation()
        simulation_engine.initialize()
        dsa_submodel_name = 'submodel_2'
        dsa_submodel = multialgorithm_simulation.dynamic_model.dynamic_submodels[dsa_submodel_name]
        self.assertTrue(isinstance(dsa_submodel, DsaSubmodel))

        # test init: is reaction_table correct?
        self.assertEqual(len(dsa_submodel.reaction_table), len(dsa_submodel.reactions))
        for rxn_id, rxn_index in dsa_submodel.reaction_table.items():
            # map reaction id to index
            self.assertEqual(dsa_submodel.reactions[rxn_index].id, rxn_id)

        # test send_initial_events(), schedule_next_reaction_execution() & schedule_ExecuteDsaReaction()
        # all of dsa_submodel's reactions should be scheduled to execute
        events = simulation_engine.event_queue.render(sim_obj=dsa_submodel, as_list=True)
        reaction_indices = set()
        send_time_idx, _, sender_idx, receiver_idx, event_type_idx, reaction_idx = list(range(6))
        for event_record in events[1:]:
            self.assertEqual(event_record[send_time_idx], (0.0,))
            self.assertEqual(event_record[sender_idx], dsa_submodel_name)
            self.assertEqual(event_record[receiver_idx], dsa_submodel_name)
            self.assertEqual(event_record[event_type_idx], ExecuteDsaReaction.__name__)
            reaction_indices.add(event_record[reaction_idx])
        self.assertEqual(reaction_indices, set([str(i) for i in range(len(dsa_submodel.reactions))]))

        # test handle_ExecuteDsaReaction_msg(): execute next reaction
        # reaction_3_forward has the highest reaction rate
        events = simulation_engine.event_queue.next_events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(dsa_submodel.reactions[event.message.reaction_index].id, 'reaction_3_forward')
        # reaction_3_forward: [c]: species_2 + (2) species_4 ==> species_5
        # check population changes
        species = ['species_2[c]', 'species_4[c]', 'species_5[c]']
        pops_before = {}
        populations = multialgorithm_simulation.local_species_population
        for species_id in species:
            pops_before[species_id] = populations.read_one(event.event_time, species_id)
        expected_pop_changes = dict(zip(species, [-1, -2, +1]))
        # set time of dsa_submodel to time of the event
        dsa_submodel.time = event.event_time
        dsa_submodel.handle_ExecuteDsaReaction_msg(event)
        for s_id, expected_pop_change in expected_pop_changes.items():
            self.assertEqual(pops_before[s_id] + expected_pop_changes[s_id],
                             populations.read_one(event.event_time, s_id))

        # zero populations and test exception
        for species_id in species:
            pop = populations.read_one(event.event_time, species_id)
            populations.adjust_discretely(event.event_time, {species_id: -pop})
        with self.assertRaises(DynamicMultialgorithmError):
            dsa_submodel.handle_ExecuteDsaReaction_msg(event)

        # test DsaSubmodel options
        expected = dict(a=1)
        options = dict(DsaSubmodel=dict(optiona=expected))
        options = {'DsaSubmodel': {'options': expected
                                  }
                  }
        multialgorithm_simulation = MultialgorithmSimulation(self.model, dict(dfba_time_step=1), options)
        multialgorithm_simulation.build_simulation()
        dsa_submodel = multialgorithm_simulation.dynamic_model.dynamic_submodels['submodel_2']
        self.assertEqual(dsa_submodel.options, expected)

    def test_simulate_deterministic_simulation_algorithm_submodel(self):
        model = MakeModel.make_test_model('1 species, 1 reaction')
        self.transform_model_for_dsa_simulation(model)
        simulation = Simulation(model)
        num_events, _ = simulation.run(time_max=100)
        self.assertGreater(num_events, 0)
