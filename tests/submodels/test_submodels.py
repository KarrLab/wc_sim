"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from obj_tables.utils import get_component_by_id
from wc_lang import Model, Species, Validator
from wc_lang.io import Reader
from wc_lang.transform import PrepForWcSimTransform
from de_sim.simulation_engine import SimulationEngine
from wc_sim.dynamic_components import DynamicModel
from wc_sim.testing.make_models import MakeModel
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.submodels.testing.skeleton_submodel import SkeletonSubmodel
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.string import indent_forest
from scipy.constants import Avogadro
import numpy
import os
import unittest
import warnings


def prepare_model(model):
    PrepForWcSimTransform().run(model)
    errors = Validator().run(model)
    if errors:
        raise ValueError(indent_forest(['The model is invalid:', [errors]]))


def make_dynamic_submodel_params(model, lang_submodel):
    multialgorithm_simulation = MultialgorithmSimulation(model, None)
    multialgorithm_simulation.initialize_components()
    multialgorithm_simulation.dynamic_model = \
        DynamicModel(multialgorithm_simulation.model,
                     multialgorithm_simulation.local_species_population,
                     multialgorithm_simulation.dynamic_compartments)

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
        copy_num = ModelUtilities.concentration_to_molecules(species, init_volume, RandomStateManager.instance())
        volume = dynamic_submodel.dynamic_compartments[species.compartment.id].volume()
        return copy_num / (volume * Avogadro)

    def test_calc_reaction_rates(self):
        # set standard deviation of initial conc. to 0
        self.setUp(std_init_concentrations=0.)

        # rate law for reaction_4-forward: k_cat_4_for * max(species_4[c], p_4)
        k_cat_4_for = 1
        p_4 = 2
        species_4_c_pop = \
            self.dynamic_submodels['submodel_2'].local_species_population.read_one(0, 'species_4[c]')
        expected_rate_reaction_4_forward = k_cat_4_for * max(species_4_c_pop, p_4)
        expected_rates = {
            'reaction_2': 0.0,
            'reaction_4': expected_rate_reaction_4_forward
        }
        for dynamic_submodel in self.dynamic_submodels.values():
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
                    with self.assertRaises(MultialgorithmError) as context:
                        dynamic_submodel.execute_reaction(reaction)
                    self.assertRegex(str(context.exception),
                                     "dynamic submodel .* cannot execute reaction")

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

        end_time = 100
        skeleton_submodel = self.make_sim_w_skeleton_submodel(lang_submodel, behavior)
        self.assertEqual(self.simulator.simulate(end_time),
                         end_time / behavior[SkeletonSubmodel.INTER_REACTION_TIME])

        behavior = {SkeletonSubmodel.INTER_REACTION_TIME: 2,
                    SkeletonSubmodel.REACTION_TO_EXECUTE: 0}    # reaction #0 makes one more 'species_1[c]'
        skeleton_submodel = self.make_sim_w_skeleton_submodel(lang_submodel, behavior)
        interval = 10
        pop_before = skeleton_submodel.local_species_population.read_one(0, 'species_1[c]')
        for end_time in range(interval, 5 * interval, interval):
            self.simulator.simulate(end_time)
            pop_after = skeleton_submodel.local_species_population.read_one(end_time, 'species_1[c]')
            delta = pop_after - pop_before
            self.assertEqual(delta, interval / behavior[SkeletonSubmodel.INTER_REACTION_TIME])
            pop_before = pop_after
