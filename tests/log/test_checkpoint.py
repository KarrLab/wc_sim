""" Checkpointing log tests

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-08-30
:Copyright: 2016, Karr Lab
:License: MIT
"""

from numpy import random
from wc_sim.log import checkpoint
import numpy
import os
import shutil
import tempfile
import unittest
import wc_sim.sim_config
import wc_sim.sim_metadata
import wc_lang.core
import wc_utils.util.types


class CheckpointLogTest(unittest.TestCase):

    def setUp(self):
        self.checkpoint_dir = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.checkpoint_dir)

    def test_constructor_creates_checkpoint_dir(self):
        checkpoint_dir = os.path.join(self.checkpoint_dir, 'checkpoint')
        checkpoint_step = 2
        init_time = 0
        metadata = {'time_max': 10}
        wc_sim.log.checkpoint.CheckpointLogger(checkpoint_dir, checkpoint_step, init_time, metadata)
        self.assertTrue(os.path.isdir(checkpoint_dir))

    def test_mock_simulator(self):
        checkpoint_dir = self.checkpoint_dir
        checkpoint_step = 2

        # full simulation and check no checkpoints
        time_max = 20
        metadata = dict(time_max=time_max)
        final_time, final_state, final_random_state = mock_simulate(metadata=metadata, checkpoint_step=checkpoint_step)
        self.assertEqual([], wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir))
        self.assertGreater(final_time, time_max)

        # run simulation to check checkpointing
        time_max = 10
        metadata = dict(time_max=time_max)
        time, state, random_state = mock_simulate(
            metadata=metadata,
            checkpoint_dir=checkpoint_dir, checkpoint_step=checkpoint_step)
        self.assertGreater(time, time_max)

        # check checkpoints created
        self.assertTrue(sorted(wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir)))
        numpy.testing.assert_array_almost_equal(
            wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir),
            numpy.linspace(checkpoint_step, time_max - checkpoint_step, time_max / checkpoint_step - 1),
            decimal=1)

        # check checkpoints have correct data
        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir, time=2)
        self.assertEqual(chkpt.metadata, dict(time_max=time_max))
        self.assertGreaterEqual(chkpt.time, 2)

        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir)
        self.assertEqual(chkpt.metadata, dict(time_max=time_max))
        self.assertLessEqual(chkpt.time, time_max)

        # resume simulation
        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir)

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
        self.assertTrue(sorted(wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir)))
        numpy.testing.assert_array_almost_equal(
            wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir),
            numpy.linspace(checkpoint_step, time_max - checkpoint_step, time_max / checkpoint_step - 1),
            decimal=1)

        # check checkpoints have correct data
        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir)
        self.assertEqual(chkpt.metadata, dict(time_max=time_max))
        self.assertLessEqual(chkpt.time, final_time)

        self.assertNotEqual(wc_utils.util.types.cast_to_builtins(chkpt.random_state),
                            wc_utils.util.types.cast_to_builtins(final_random_state))
        wc_utils.util.types.assert_value_not_equal(chkpt.random_state, final_random_state, check_iterable_ordering=True)

    @unittest.skip('Update to test wc_sim')
    def test_simulator(self):
        time_step = 1
        random_seed = 0
        out_dir = self.out_dir
        checkpoint_step = 2
        checkpoint_dir = self.checkpoint_dir

        model = build_mock_model()

        options = {
            'progress_bar': {
                'enabled': False,
            },
            'log': {
                'checkpoint': {
                    'enabled': True,
                    'step': checkpoint_step,
                    'dirname': checkpoint_dir,
                },
            },
        }

        # full simulation
        time_max = 20
        sim_config = wc_sim.sim_config.SimulationConfig(time_max=time_max, time_step=time_step, random_seed=random_seed)
        options['log']['checkpoint']['enabled'] = False
        simulator = wc.sim.core.Simulator(model, options)
        simulator.run(sim_config, out_dir=out_dir)
        final_time = simulator._time
        final_species_counts = simulator.model.state.species_counts
        final_growth = simulator.model.state.growth
        final_random_state = simulator.model.random_state.get_state()

        self.assertEqual([], wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir))
        self.assertEqual(final_time, time_max)

        # run simulation to check checkpointing
        time_max = 10
        sim_config = wc_sim.sim_config.SimulationConfig(time_max=time_max, time_step=time_step, random_seed=random_seed)
        options['log']['checkpoint']['enabled'] = True
        wc.sim.core.Simulator(model, options).run(sim_config, out_dir=out_dir)
        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir)
        self.assertEqual(chkpt.time, time_max)

        # check checkpoints created
        self.assertTrue(sorted(wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir)))
        numpy.testing.assert_equal(
            wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir),
            numpy.linspace(checkpoint_step, time_max, time_max / checkpoint_step))

        # check checkpoints have correct data
        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir, time=2.)
        wc_utils.util.types.assert_value_equal(
            chkpt.metadata,
            wc_sim.sim_metadata.Metadata(None, sim_config, None, None),
            check_iterable_ordering=True)
        self.assertEqual(chkpt.time, 2)

        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir)
        wc_utils.util.types.assert_value_equal(
            chkpt.metadata,
            wc_sim.sim_metadata.Metadata(None, sim_config, None, None),
            check_iterable_ordering=True)
        self.assertEqual(chkpt.time, time_max)

        # resume simulation
        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir)

        time_max = 20
        sim_config = wc_sim.sim_config.SimulationConfig(time_max=time_max, time_step=time_step, random_seed=random_seed)
        options['log']['checkpoint']['enabled'] = True
        simulator = wc.sim.core.Simulator(model, options)
        simulator.restart(chkpt, out_dir=out_dir, sim_config=sim_config)
        time = simulator._time
        species_counts = simulator.model.state.species_counts
        growth = simulator.model.state.growth
        random_state = simulator.model.random_state.get_state()

        self.assertEqual(time, final_time)
        numpy.testing.assert_equal(species_counts, final_species_counts)
        numpy.testing.assert_equal(growth, final_growth)
        numpy.testing.assert_equal(random_state, final_random_state)

        # check checkpoints created
        self.assertTrue(sorted(wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir)))
        numpy.testing.assert_equal(
            wc_sim.log.checkpoint.Checkpoint.list_checkpoints(dirname=checkpoint_dir),
            numpy.linspace(checkpoint_step, time_max, time_max / checkpoint_step))

        # check checkpoints have correct data
        chkpt = wc_sim.log.checkpoint.Checkpoint.get_checkpoint(dirname=checkpoint_dir)
        wc_utils.util.types.assert_value_equal(
            chkpt.metadata,
            wc_sim.sim_metadata.Metadata(None, sim_config, None, None),
            check_iterable_ordering=True)
        self.assertEqual(chkpt.time, final_time)
        numpy.testing.assert_equal(chkpt.random_state, final_random_state)


def build_mock_model():
    ''' Create test model:

        L --> R

    * 1 compartment
    * 2 species
    * 1 reaction
    * 1 submodel
    '''
    model = wc_lang.core.Model()

    submodel = model.submodels.create(id='submodel', algorithm=wc_lang.core.SubmodelAlgorithm.ssa)

    compartment_c = model.compartments.create(id='c', initial_volume=1.)
    compartment_e = model.compartments.create(id='e', initial_volume=1.)

    species_type_L = model.species_types.create(id='L', molecular_weight=10)
    species_type_R = model.species_types.create(id='R', molecular_weight=10)

    species_L = wc_lang.core.Species(species_type=species_type_L, compartment=compartment_c)
    species_R = wc_lang.core.Species(species_type=species_type_R, compartment=compartment_c)
    species = [species_L, species_R]

    wc_lang.core.Concentration(species=species_L, value=1.)
    wc_lang.core.Concentration(species=species_R, value=0.)

    reaction = submodel.reactions.create(id='reaction')
    reaction.rate_laws.create(direction=wc_lang.core.RateLawDirection.forward,
                              equation=wc_lang.core.RateLawEquation(expression='0.0'))
    reaction.participants.create(species=species_L, coefficient=-1)
    reaction.participants.create(species=species_R, coefficient=1)

    model.parameters.create(id='fraction_dry_weight', value=1.)
    model.parameters.create(id='cell_cycle_length', value=30. * 60),  # s

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
        logger = wc_sim.log.checkpoint.CheckpointLogger(checkpoint_dir, checkpoint_step, init_time, metadata)

    while time < metadata['time_max']:
        dt = random_state.exponential(1. / 100.)
        time += dt
        state += 1

        if time > metadata['time_max']:
            break

        if checkpoint_dir:
            logger.checkpoint_periodically(time, state, random_state)

    return (time, state, random_state.get_state())
