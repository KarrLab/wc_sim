"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
import copy
import numpy
import os
import unittest
import warnings

from de_sim.simulation_config import SimulationConfig
from de_sim.simulation_engine import SimulationEngine
from obj_tables.utils import get_component_by_id
from wc_lang import Model, Species, Validator
from wc_lang.io import Reader
from wc_lang.transform import PrepForWcSimTransform
from wc_onto import onto
from wc_sim.dynamic_components import DynamicModel, DynamicFunction
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.simulation import Simulation
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.submodels.testing.deterministic_simulation_algorithm import DsaSubmodel, ExecuteDsaReaction
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import read_model_for_test, TempConfigFileModifier
from wc_utils.util.ontology import are_terms_equivalent
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.string import indent_forest


def prepare_model(model):
    PrepForWcSimTransform().run(model)
    errors = Validator().run(model)
    if errors:
        raise ValueError(indent_forest(['The model is invalid:', [errors]]))


def build_sim_from_model(model, time_max=10, dfba_time_step=1, ode_time_step=1, options=None):
    de_simulation_config = SimulationConfig(time_max=time_max)
    wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=dfba_time_step,
                                       ode_time_step=ode_time_step)
    options = {} if options is None else options
    multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config, options=options)
    simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
    return multialgorithm_simulation, simulation_engine, dynamic_model


class TestDynamicSubmodelStatically(unittest.TestCase):

    def setUp(self, std_init_concentrations=None):

        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME)[Model][0]
        prepare_model(self.model)

        if std_init_concentrations is not None:
            for conc in self.model.distribution_init_concentrations:
                conc.std = std_init_concentrations

        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, ode_time_step=2, dfba_time_step=5)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, wc_sim_config)
        _, self.dynamic_model = multialgorithm_simulation.build_simulation()
        self.dynamic_submodels = self.dynamic_model.dynamic_submodels

        self.config_file_modifier = TempConfigFileModifier()

    def tearDown(self):
        self.config_file_modifier.clean_up()

    def test_get_state(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            self.assertEqual(dynamic_submodel.get_state(), DynamicSubmodel.GET_STATE_METHOD_MESSAGE)

    def test_get_num_submodels(self):
        for dynamic_submodel in self.dynamic_submodels.values():
            self.assertEqual(dynamic_submodel.get_num_submodels(), 1)

    def expected_molar_conc(self, dynamic_submodel, species_id):
        species = list(filter(lambda s: s.id == species_id, dynamic_submodel.species))[0]
        init_volume = species.compartment.init_volume.mean
        copy_num = ModelUtilities.sample_copy_num_from_concentration(species, init_volume, RandomStateManager.instance())
        volume = dynamic_submodel.dynamic_compartments[species.compartment.id].volume()
        return copy_num / (volume * Avogadro)

    def test_calc_reaction_rates(self):
        # set standard deviation of initial conc. to 0
        self.setUp(std_init_concentrations=0.)
        multialgorithm_simulation, _, _ = build_sim_from_model(self.model)

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
        'submodel_2': set([
            'reaction_2',
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

    def test_flush_after_reaction(self):
        self.config_file_modifier.write_test_config_file([('expression_caching', 'True'),
                                                          ('cache_invalidation', "'reaction_dependency_based'")])
        dependencies_mdl_file = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'test_dependencies.xlsx')
        model = Reader().run(dependencies_mdl_file)[Model][0]
        _, _, dynamic_model = build_sim_from_model(model)

        # eval DynamicFunction function_4
        function_4 = dynamic_model.dynamic_functions['function_4']
        val = function_4.eval(0)
        self.assertEqual(dynamic_model.cache_manager.get(function_4), val)
        test_submodel = dynamic_model.dynamic_submodels['dsa_submodel']
        reactions = {rxn.id: rxn for rxn in test_submodel.reactions}
        test_submodel.dynamic_model.flush_after_reaction(reactions['reaction_1'])
        with self.assertRaisesRegex(MultialgorithmError, 'dynamic expression .* not in cache'):
            dynamic_model.cache_manager.get(function_4)

        # since reaction_10 has no dependencies, it tests the if statement in flush_after_reaction()
        cache_copy = copy.deepcopy(dynamic_model.cache_manager._cache)
        test_submodel.dynamic_model.flush_after_reaction(reactions['reaction_10'])
        self.assertEqual(cache_copy, dynamic_model.cache_manager._cache)


class TestDsaSubmodel(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_submodel_no_shared_species.xlsx')
        self.model = Reader().run(self.MODEL_FILENAME, validate=True)[Model][0]
        self.transform_model_for_dsa_simulation(self.model)
        prepare_model(self.model)
        self.multialgorithm_simulation, self.simulation_engine, _ = build_sim_from_model(self.model)
        self.simulation_engine.initialize()
        self.dsa_submodel_name = 'submodel_2'
        self.dsa_submodel = self.multialgorithm_simulation.dynamic_model.dynamic_submodels[self.dsa_submodel_name]

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
        self.assertTrue(isinstance(self.dsa_submodel, DsaSubmodel))

        # test init: is reaction_table correct?
        self.assertEqual(len(self.dsa_submodel.reaction_table), len(self.dsa_submodel.reactions))
        for rxn_id, rxn_index in self.dsa_submodel.reaction_table.items():
            # map reaction id to index
            self.assertEqual(self.dsa_submodel.reactions[rxn_index].id, rxn_id)

        # test init_before_run(), schedule_next_reaction_execution() & schedule_ExecuteDsaReaction()
        # all of self.dsa_submodel's reactions should be scheduled to execute
        events = self.simulation_engine.event_queue.render(sim_obj=self.dsa_submodel, as_list=True)
        reaction_indices = set()
        send_time_idx, _, sender_idx, receiver_idx, event_type_idx, reaction_idx = list(range(6))
        for event_record in events[1:]:
            self.assertEqual(event_record[send_time_idx], (0.0,))
            self.assertEqual(event_record[sender_idx], self.dsa_submodel_name)
            self.assertEqual(event_record[receiver_idx], self.dsa_submodel_name)
            self.assertEqual(event_record[event_type_idx], ExecuteDsaReaction.__name__)
            reaction_indices.add(event_record[reaction_idx])
        self.assertEqual(reaction_indices, set([str(i) for i in range(len(self.dsa_submodel.reactions))]))

        # test handle_ExecuteDsaReaction_msgs(): execute next reaction
        # reaction_3_forward has the highest reaction rate
        events = self.simulation_engine.event_queue.next_events()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(self.dsa_submodel.reactions[event.message.reaction_index].id, 'reaction_3_forward')
        # reaction_3_forward: [c]: species_2 + (2) species_4 ==> species_5
        # check population changes
        species = ['species_2[c]', 'species_4[c]', 'species_5[c]']
        pops_before = {}
        populations = self.multialgorithm_simulation.local_species_population
        for species_id in species:
            pops_before[species_id] = populations.read_one(event.event_time, species_id)
        expected_pop_changes = dict(zip(species, [-1, -2, +1]))
        # set time of dsa_submodel to time of the event
        self.dsa_submodel.time = event.event_time
        self.dsa_submodel.handle_ExecuteDsaReaction_msgs(event)
        for s_id, expected_pop_change in expected_pop_changes.items():
            self.assertEqual(pops_before[s_id] + expected_pop_changes[s_id],
                             populations.read_one(event.event_time, s_id))

        # zero populations and test exception
        for species_id in species:
            pop = populations.read_one(event.event_time, species_id)
            populations.adjust_discretely(event.event_time, {species_id: -pop})
        with self.assertRaises(DynamicMultialgorithmError):
            self.dsa_submodel.handle_ExecuteDsaReaction_msgs(event)

        # test DsaSubmodel options
        expected = dict(a=1)
        options = {'DsaSubmodel': {'options': expected
                                  }
                  }
        multialgorithm_simulation, _, _ = build_sim_from_model(self.model, options=options)
        dsa_submodel = multialgorithm_simulation.dynamic_model.dynamic_submodels['submodel_2']
        self.assertEqual(dsa_submodel.options, expected)

    def test_rate_eq_0(self):
        # Disable caching so Parameter values and RateLaws are not cached
        self.dsa_submodel.dynamic_model._stop_caching()
        # set rate constant for reaction_5's rate law to 0; the parameter is 'k_cat_5_for'
        k_cat_5_for = self.dsa_submodel.dynamic_model.dynamic_parameters['k_cat_5_for']
        k_cat_5_for.value = 0
        rxn_ids_to_rxn_indices = {rxn.id: idx for idx, rxn in enumerate(self.dsa_submodel.reactions)}
        index_reaction_5 = rxn_ids_to_rxn_indices['reaction_5']
        # empty the simulator's event queue
        self.simulation_engine.event_queue.reset()

        # check that event time for the initial event for reaction_5 is inf
        self.dsa_submodel.init_before_run()
        for event in self.simulation_engine.event_queue.event_heap:
            if event.message.reaction_index == index_reaction_5:
                self.assertEqual(event.event_time, float('inf'))

        # when the next execution of reaction_5 is scheduled the only event should be for reaction_5 at inf
        self.simulation_engine.event_queue.reset()
        self.dsa_submodel.schedule_next_reaction_execution(self.dsa_submodel.reactions[index_reaction_5])
        events = self.simulation_engine.event_queue.next_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_time, float('inf'))
        self.assertEqual(events[0].message.reaction_index, index_reaction_5)

    def test_simulate_deterministic_simulation_algorithm_submodel(self):
        model = MakeModel.make_test_model('1 species, 1 reaction')
        self.transform_model_for_dsa_simulation(model)
        simulation = Simulation(model)
        num_events = simulation.run(time_max=100).num_events
        self.assertGreater(num_events, 0)
