'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017, Karr Lab
:License: MIT
'''

import os, unittest
from argparse import Namespace
from six import iteritems

from wc_sim.multialgorithm.multialgorithm_simulation import (DynamicModel, MultialgorithmSimulation,
    CheckModel)
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from obj_model import utils
from wc_lang.io import Reader
from wc_lang.core import Reaction, RateLaw, RateLawEquation


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME)
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        self.dynamic_model = DynamicModel(self.model, self.multialgorithm_simulation)

    def test_initialize_dynamic_model(self):
        self.dynamic_model.initialize()
        self.assertEqual(self.dynamic_model.extracellular_volume, 1.00E-12)
        self.assertEqual(self.dynamic_model.volume, 4.58E-17)
        self.assertEqual(self.dynamic_model.fraction_dry_weight, 0.3)
        self.assertAlmostEqual(self.dynamic_model.mass, 1.56273063E-42)
        self.assertAlmostEqual(self.dynamic_model.dry_weight, 4.68819190E-43)
        self.assertAlmostEqual(self.dynamic_model.density, 3.41207562E-26)


class TestMultialgorithmSimulation(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME)
        args = Namespace(FBA_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        self.dynamic_model = DynamicModel(self.model, self.multialgorithm_simulation)

    @unittest.skip('')
    def test_initialize_simulation(self):
        self.multialgorithm_simulation.initialize()
        self.simulation_engine = self.multialgorithm_simulation.build_simulation()
        self.assertEqual(len(self.simulation_engine.simulation_objects.keys()), 3)
        for name,simulation_obj in iteritems(self.simulation_engine.simulation_objects):
            print("\n{}: {} event queue:".format(simulation_obj.__class__.__name__, name))
            print(simulation_obj.event_queue_to_str())
        # self.simulation_engine.simulate(10)

    def test_transcode_rate_laws(self):
        self.multialgorithm_simulation.transcode_rate_laws()
        num_transcoded_rate_law_equations = len(
            list(filter(lambda rl: hasattr(rl.equation, 'transcoded'), self.model.get_rate_laws())))
        self.assertEqual(num_transcoded_rate_law_equations, 5)



class TestCheckModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_check_model_model.xlsx')

    def setUp(self):
        # read a model
        self.model = Reader().run(self.MODEL_FILENAME)
        self.check_model = CheckModel(self.model)

    def test_verify_biomass_reaction(self):
        self.assertEqual(self.check_model.verify_biomass_reaction(), [])
        # remove all the reactions
        for submodel in self.model.get_submodels():
            submodel.reactions = set()
        self.assertIn("biomass reaction required", self.check_model.verify_biomass_reaction()[0])

    def test_test_rate_laws(self):
        # good laws
        self.assertEqual(self.check_model.test_rate_laws(), [])

        # test errors
        # redefine one reaction
        rate_law_equation = RateLawEquation(
            expression='',
            transcoded='',
        )
        rate_law = RateLaw(
            equation=rate_law_equation,
        )
        rate_law_equation.rate_law = rate_law
        a_reaction = self.model.get_reactions().pop()
        a_reaction.rate_laws = [rate_law]
        TEST_ID = 'test_id'
        a_reaction.id = TEST_ID

        # rate laws that fail transcoding
        rate_law_equation.expression='__ 0'
        self.assertIn("Security risk: rate law expression '__", self.check_model.test_rate_laws()[0])
        rate_law_equation.expression='not_a_specie[e]'
        self.assertIn("'not_a_specie[e]' not a known specie", self.check_model.test_rate_laws()[0])
        
        # rate laws that fail evaluation
        rate_law_equation.expression='foo foo'
        self.assertIn("reaction '{}' has syntax error".format(TEST_ID),
            self.check_model.test_rate_laws()[0])
        rate_law_equation.expression='cos(0)'
        self.assertIn("name 'cos' is not defined".format(TEST_ID),
            self.check_model.test_rate_laws()[0])
        rate_law_equation.expression='{{{*'
        self.assertIn("EOF in multi-line statement", self.check_model.test_rate_laws()[0])
        
    def test_verify_reactant_compartments(self):
        for actual,expected in zip(self.check_model.verify_reactant_compartments(), 
            [".*reaction_1 uses specie specie_1 in another compartment: e",
                ".*reaction_1 uses specie specie_2 in another compartment: e",
                ".*'submodel_2' must contain a compartment attribute"]):
            self.assertRegex(actual, expected)
