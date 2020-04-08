""" Take periodic checkpoints in a multialgorithmic simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-05-08
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import pandas

from de_sim.checkpoint import Checkpoint
from de_sim.simulation_checkpoint_object import CheckpointSimulationObject, AccessStateObjectInterface
from wc_sim.species_populations import LocalSpeciesPopulation
from wc_sim.submodels.ssa import SsaSubmodel
from wc_utils.util.misc import obj_to_str


class MultialgorithmCheckpoint(Checkpoint):
    """ Checkpoint class that holds multialgorithmic checkpoints
    """
    pass


class AccessState(AccessStateObjectInterface):
    """ Obtain checkpoints of a multialgorithm simulation's biological state and random state

    Attributes:
        local_species_population (:obj:`LocalSpeciesPopulation`): provide a simulation's species populations
        dynamic_model (:obj:`DynamicModel`): provide the cell's aggregate state in a simulation
        multialgorithm_simulation (:obj:`MultialgorithmSimulation`): the `MultialgorithmSimulation`
    """

    def __init__(self, local_species_population, dynamic_model, multialgorithm_simulation):
        self.local_species_population = local_species_population
        self.dynamic_model = dynamic_model
        self.multialgorithm_simulation = multialgorithm_simulation

    def get_checkpoint_state(self, time):
        """ Obtain a checkpoint of the biological state

        Args:
            time (:obj:`float`): the simulation time of the checkpoing being created

        Returns:
            :obj:`dict` of `dict`: dictionaries containing the simulation's state: `population` has
                the simulation's species populations, `aggregate_state` contains its aggregrate compartment
                states, and `observables` contains all of its observables
        """
        state = {
            'population': self.local_species_population.read(time, round=False),
            'aggregate_state': self.dynamic_model.get_aggregate_state(),
            'observables': self.dynamic_model.eval_dynamic_observables(time),
            'functions': self.dynamic_model.eval_dynamic_functions(time),
        }
        return state

    def get_random_state(self):
        """ Obtain a checkpoint of the random state

        Provides a dictionary that maps components of the simulation to their random states, which
        are all instances of `numpy.random.RandomState`

        Returns:
            :obj:`dict`: a dictionary of the random states in the simulation
        """
        random_states = {}
        random_states['local_species_population'] = self.local_species_population.random_state.get_state()
        random_states['submodels'] = {}
        for submodel in self.multialgorithm_simulation.dynamic_model.dynamic_submodels.values():
            if isinstance(submodel, SsaSubmodel):
                # only SSA submodels use random numbers
                random_states['submodels'][submodel.id] = submodel.random_state.get_state()
        return random_states


class MultialgorithmicCheckpointingSimObj(CheckpointSimulationObject):
    """ A checkpointing simulation object for a multialgorithmic simulation

    Attributes:
        access_state_object (:obj:`AccessState`): an object that provides checkpoints
    """
    def __init__(self, name, checkpoint_period, checkpoint_dir, local_species_population,
                 dynamic_model, multialgorithm_simulation):
        """ Create a MultialgorithmicCheckpointingSimObj

        Args:
            name (:obj:`str`): name
            checkpoint_period (:obj:`float`): checkpoint period
            checkpoint_dir (:obj:`str`): checkpoint directory
            local_species_population (:obj:`LocalSpeciesPopulation`): the `LocalSpeciesPopulation`
            dynamic_model (:obj:`DynamicModel`): the `DynamicModel`
            multialgorithm_simulation (:obj:`MultialgorithmSimulation`): the `MultialgorithmSimulation`
        """

        self.access_state_object = AccessState(local_species_population, dynamic_model,
                                               multialgorithm_simulation)
        super().__init__(name, checkpoint_period, checkpoint_dir, self.access_state_object)

    def __str__(self):
        """ Provide a readable representation of this `MultialgorithmicCheckpointingSimObj`

        Returns:
            :obj:`str`: a readable representation of this `MultialgorithmicCheckpointingSimObj`
        """

        return obj_to_str(self, ['name', 'period', 'checkpoint_dir'])
