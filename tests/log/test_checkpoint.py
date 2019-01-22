""" Checkpointing log tests

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-08-30
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from numpy import random
import numpy
import os
import shutil
import tempfile
import unittest
import copy

from wc_sim.core.sim_metadata import SimulationMetadata
from wc_sim.log.checkpoint import Checkpoint, CheckpointLogger
from wc_utils.util.ontology import wcm_ontology
from wc_utils.util.units import unit_registry
import wc_sim.core.sim_config
import wc_lang
import wc_utils.util.types


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.empty_checkpoint1 = Checkpoint(None, None, None)
        self.empty_checkpoint2 = Checkpoint(None, None, None)

        # Checkpoint(time, state, random_state)
        time = 2
        state = [[1, 2], 3]
        random_state = random.RandomState(seed=0)
        attrs = dict(time=time, state=state, random_state=random_state)
        self.non_empty_checkpoint1 = Checkpoint(time, state, random_state)
        self.non_empty_checkpoint2 = self.non_empty_checkpoint1

        # make Checkpoints that differ
        diff_time = 3
        diff_state = [[1, 2], 3, 4]
        diff_random_state = random.RandomState(seed=1)
        diff_attrs = dict(time=diff_time, state=diff_state,
            random_state=diff_random_state)
        self.diff_checkpoints = []
        for attr in ['time', 'state', 'random_state']:
            args = copy.deepcopy(attrs)
            args[attr] = copy.deepcopy(diff_attrs[attr])
            self.diff_checkpoints.append(Checkpoint(**args))

    def test_equality(self):
        obj = object()
        self.assertEqual(self.empty_checkpoint1, self.empty_checkpoint1)
        self.assertEqual(self.empty_checkpoint1, self.empty_checkpoint2)
        self.assertNotEqual(self.empty_checkpoint1, obj)
        self.assertTrue(self.empty_checkpoint1 != None)

        self.assertEqual(self.non_empty_checkpoint1, self.non_empty_checkpoint1)
        self.assertEqual(self.non_empty_checkpoint1, self.non_empty_checkpoint2)
        self.assertNotEqual(self.non_empty_checkpoint1, obj)
        for ckpt in self.diff_checkpoints:
            self.assertNotEqual(self.non_empty_checkpoint1, ckpt)


class CheckpointLogTest(unittest.TestCase):

    def setUp(self):
        self.checkpoint_dir = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.checkpoint_dir)
        shutil.rmtree(self.out_dir)

    def test_constructor_creates_checkpoint_dir(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, 'checkpoint')
        checkpoint_step = 2
        init_time = 0
        CheckpointLogger(checkpoint_dir, checkpoint_step, init_time)
        self.assertTrue(os.path.isdir(checkpoint_dir))

    def test_mock_simulator(self):
        checkpoint_dir = self.checkpoint_dir
        checkpoint_step = 2

        # full simulation and check no checkpoints
        time_max = 20
        metadata = dict(time_max=time_max)
        final_time, final_state, final_random_state = mock_simulate(metadata=metadata, checkpoint_step=checkpoint_step)

        self.assertEqual([], Checkpoint.list_checkpoints(dirname=checkpoint_dir, error_if_empty=False))
        with self.assertRaises(ValueError):
            Checkpoint.list_checkpoints(dirname=checkpoint_dir)
        self.assertGreater(final_time, time_max)

        # run simulation to check checkpointing
        time_max = 10
        metadata = dict(time_max=time_max)
        time, state, random_state = mock_simulate(
            metadata=metadata,
            checkpoint_dir=checkpoint_dir, checkpoint_step=checkpoint_step)
        self.assertGreater(time, time_max)

        # check checkpoints created
        self.assertTrue(sorted(Checkpoint.list_checkpoints(dirname=checkpoint_dir)))
        numpy.testing.assert_array_almost_equal(
            Checkpoint.list_checkpoints(dirname=checkpoint_dir),
            numpy.linspace(checkpoint_step, time_max - checkpoint_step, time_max / checkpoint_step - 1),
            decimal=1)

        # check checkpoints have correct data
        checkpoint_time = 5
        chkpt = Checkpoint.get_checkpoint(dirname=checkpoint_dir, time=checkpoint_time)
        self.assertIn('time:', str(chkpt))
        self.assertIn('state:', str(chkpt))
        self.assertLessEqual(chkpt.time, checkpoint_time)

        chkpt = Checkpoint.get_checkpoint(dirname=checkpoint_dir)
        self.assertLessEqual(chkpt.time, time_max)

        # resume simulation
        chkpt = Checkpoint.get_checkpoint(dirname=checkpoint_dir)

        time_max = 20
        metadata = dict(time_max=time_max)
        time, state, random_state = mock_simulate(
            metadata=metadata,
            init_time=chkpt.time, init_state=chkpt.state, init_random_state=chkpt.random_state,
            checkpoint_dir=checkpoint_dir, checkpoint_step=checkpoint_step)
        self.assertEqual(time, final_time)
        self.assertEqual(state, final_state)
        wc_utils.util.types.assert_value_equal(random_state, final_random_state, check_iterable_ordering=True)

        # check checkpoints created
        self.assertTrue(sorted(Checkpoint.list_checkpoints(dirname=checkpoint_dir)))
        numpy.testing.assert_array_almost_equal(
            Checkpoint.list_checkpoints(dirname=checkpoint_dir),
            numpy.linspace(checkpoint_step, time_max - checkpoint_step, time_max / checkpoint_step - 1),
            decimal=1)

        # check checkpoints have correct data
        chkpt = Checkpoint.get_checkpoint(dirname=checkpoint_dir)
        self.assertLessEqual(chkpt.time, final_time)

        self.assertNotEqual(wc_utils.util.types.cast_to_builtins(chkpt.random_state),
                            wc_utils.util.types.cast_to_builtins(final_random_state))
        wc_utils.util.types.assert_value_not_equal(chkpt.random_state, final_random_state, check_iterable_ordering=True)


def build_mock_model():
    ''' Create test model:

        L --> R

    * 1 compartment
    * 2 species
    * 1 reaction
    * 1 submodel
    '''
    model = wc_lang.Model()

    submodel = model.submodels.create(id='submodel', framework=wcm_ontology['WCM:stochastic_simulation_algorithm'])

    compartment_c = model.compartments.create(id='c', mean_init_volume=1.)
    compartment_e = model.compartments.create(id='e', mean_init_volume=1.)

    species_type_L = model.species_types.create(id='L', molecular_weight=10)
    species_type_R = model.species_types.create(id='R', molecular_weight=10)

    species_L = wc_lang.Species(id='L[c]', species_type=species_type_L, compartment=compartment_c)
    species_R = wc_lang.Species(id='R[c]', species_type=species_type_R, compartment=compartment_c)
    species = [species_L, species_R]

    wc_lang.Concentration(species=species_L, value=1., units=unit_registry.parse_units('molecule'))
    wc_lang.Concentration(species=species_R, value=0., units=unit_registry.parse_units('molecule'))

    reaction = submodel.reactions.create(id='reaction')
    reaction.rate_laws.create(direction=wc_lang.RateLawDirection.forward,
                              equation=wc_lang.RateLawEquation(expression='0.0'))
    reaction.participants.create(species=species_L, coefficient=-1)
    reaction.participants.create(species=species_R, coefficient=1)

    model.parameters.create(id='mean_doubling_time', value=30. * 60, units=unit_registry.parse_units('s')),

    return model


def mock_simulate(metadata, init_time=0, init_state=None, init_random_state=None, checkpoint_dir=None,
                  checkpoint_step=None):
    # initialize
    if init_state is None:
        state = 0
    else:
        state = init_state

    random_state = random.RandomState(seed=0)
    if init_random_state is not None:
        random_state.set_state(init_random_state)

    # simulate temporal dynamics
    time = init_time

    if checkpoint_dir:
        logger = CheckpointLogger(checkpoint_dir, checkpoint_step, init_time)

    while time < metadata['time_max']:
        dt = random_state.exponential(1. / 100.)
        time += dt
        state += 1

        if time > metadata['time_max']:
            break

        if checkpoint_dir:
            logger.checkpoint_periodically(time, state, random_state)

    return (time, state, random_state.get_state())
