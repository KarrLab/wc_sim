'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017, Karr Lab
:License: MIT
'''

import os, unittest
from argparse import Namespace
import six
import math

from wc_sim.multialgorithm.multialgorithm_simulation import (DynamicModel, MultialgorithmSimulation,
    CheckModel)
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from obj_model import utils
from wc_lang.io import Reader
from wc_lang.core import (Reaction, RateLaw, RateLawEquation, Submodel, SubmodelAlgorithm,
    RateLawDirection, SpeciesType)


class TestDynamicModel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        for model in [Submodel, Reaction, SpeciesType]:
            model.objects.reset()
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
        for name,simulation_obj in six.iteritems(self.simulation_engine.simulation_objects):
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
        for model in [Submodel, Reaction, SpeciesType]:
            model.objects.reset()
        # read a wc model
        self.model = Reader().run(self.MODEL_FILENAME)
        self.check_model = CheckModel(self.model)

    def test_check_dfba_submodel_1(self):
        dfba_submodel = Submodel.objects.get_one(id='dfba_submodel')
        self.assertEqual(self.check_model.check_dfba_submodel(dfba_submodel), [])

        # delete a reaction's rate laws
        Reaction.objects.get_one(id='reaction_1').rate_laws = []
        errors = self.check_model.check_dfba_submodel(dfba_submodel)
        self.assertIn("Error: no rate law for reaction 'reaction_name_1' in submodel", errors[0])

    def test_check_dfba_submodel_2(self):
        dfba_submodel = Submodel.objects.get_one(id='dfba_submodel')

        # delete a min_flux
        reaction_2 = Reaction.objects.get_one(id='reaction_2')
        reaction_2.rate_laws[0].min_flux = float('NaN')
        errors = self.check_model.check_dfba_submodel(dfba_submodel)
        self.assertIn("Error: no min_flux for forward direction of reaction", errors[0])
        
        # remove all the reactions
        for submodel in self.model.get_submodels():
            submodel.reactions = []
        errors = self.check_model.check_dfba_submodel(dfba_submodel)
        self.assertIn("No reactions participating in objective function", errors[0])

    def test_check_dynamic_submodel(self):
        ssa_submodel = Submodel.objects.get_one(id='ssa_submodel')
        self.assertEqual(self.check_model.check_dynamic_submodel(ssa_submodel), [])

        reaction_4 = Reaction.objects.get_one(id='reaction_4')
        # add reaction_4 backward ratelaw -> not reversible but has backward error
        reaction_4_ratelaw = reaction_4.rate_laws[0]
        reaction_4.rate_laws[0].direction = RateLawDirection.backward
        errors = self.check_model.check_dynamic_submodel(ssa_submodel)
        self.assertIn("is not reversible but has a 'backward' rate law specified", errors[0])

        # remove reaction_4 forward ratelaw -> no rate law error
        reaction_4.rate_laws = []
        errors = self.check_model.check_dynamic_submodel(ssa_submodel)
        self.assertIn("has no rate law specified", errors[0])

        # put back the good rate law for reaction_4
        reaction_4_ratelaw.direction = RateLawDirection.forward
        reaction_4.rate_laws = [reaction_4_ratelaw]
        self.assertEqual(self.check_model.check_dynamic_submodel(ssa_submodel), [])

        # remove reaction_3 backward ratelaw -> reversible but only forward error
        reaction_3 = Reaction.objects.get_one(id='reaction_3')
        del reaction_3.rate_laws[1:]
        errors = self.check_model.check_dynamic_submodel(ssa_submodel)
        self.assertIn("is reversible but has only a 'forward' rate law specified", errors[0])

    def test_check_rate_law_equations(self):
        # good laws
        self.assertEqual(self.check_model.check_rate_law_equations(), [])

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
        self.assertIn("Security risk: rate law expression '__", self.check_model.check_rate_law_equations()[0])
        rate_law_equation.expression='not_a_specie[e]'
        self.assertIn("'not_a_specie[e]' not a known specie", self.check_model.check_rate_law_equations()[0])
        
        # rate laws that fail evaluation
        rate_law_equation.expression='foo foo'
        self.assertIn("reaction '{}' has syntax error".format(TEST_ID),
            self.check_model.check_rate_law_equations()[0])
        rate_law_equation.expression='cos(0)'
        self.assertIn("name 'cos' is not defined".format(TEST_ID),
            self.check_model.check_rate_law_equations()[0])
        rate_law_equation.expression='{{{*'
        self.assertIn("EOF in multi-line statement", self.check_model.check_rate_law_equations()[0])
        
    def test_verify_reactant_compartments(self):
        for actual,expected in zip(self.check_model.verify_reactant_compartments(), 
            [".*reaction_1 uses specie specie_1 in another compartment: e",
                ".*reaction_1 uses specie specie_2 in another compartment: e",
                ".*'ssa_submodel' must contain a compartment attribute"]):
            six.assertRegex(self, actual, expected)
