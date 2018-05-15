""" Take periodic checkpoints in a multialgorithmic simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-08
:Copyright: 2018, Karr Lab
:License: MIT
"""
import numpy as np
import pandas

from wc_sim.log.checkpoint import Checkpoint
from wc_sim.core.simulation_checkpoint_object import CheckpointSimulationObject, AccessStateObjectInterface


class MultialgorithmCheckpoint(Checkpoint):
    """ Checkpoint class that holds multialgorithmic checkpoints
    """

    def __init__(self, metadata, time, state, random_state):
        super().__init__(metadata, time, state, random_state)

    @staticmethod
    def convert_checkpoints(dirname):
        """ Convert the species population in saved checkpoints into a pandas dataframe

        Args:
            dirname (:obj:`str`): directory containing the checkpoint data

        Returns:
            :obj:`pandas.DataFrame`: the species popuation in a simulation checkpoint history
        """
        # create an empty DataFrame
        checkpoints = Checkpoint.list_checkpoints(dirname)
        checkpoint = Checkpoint.get_checkpoint(dirname, time=0)
        species_pop, _ = checkpoint.state
        species_ids = species_pop.keys()
        pred_species_pops = pandas.DataFrame(index=checkpoints, columns=species_ids, dtype=np.float64)

        # load the DataFrame
        for time in Checkpoint.list_checkpoints(dirname):
            species_populations, _ = Checkpoint.get_checkpoint(dirname, time=time).state
            for species_id,population in species_populations.items():
                pred_species_pops.loc[time, species_id] = population
        return pred_species_pops


class AccessStateObject(AccessStateObjectInterface):
    """ Get a checkpoint for a multialgorithm simulation

    Attributes:
        local_species_population (:obj:`LocalSpeciesPopulation`): provide a simulation's species populations
        dynamic_model (:obj:`DynamicModel`): provide the cell's aggregate state in a simulation
    """

    def __init__(self, local_species_population, dynamic_model):
        self.local_species_population = local_species_population
        self.dynamic_model = dynamic_model

    def get_checkpoint_state(self, time):
        """ Obtain a checkpoint

        Returns:
            :obj:`tuple` of (`dict`, `dict`): dictionaries with the species populations and the
                cell's aggregate state
        """
        return (self.local_species_population.read(time), self.dynamic_model.get_aggregate_state())


class MultialgorithmicCheckpointingSimObj(CheckpointSimulationObject):
    """ A checkpointing simulation object for a multialgorithmic simulatino

    Attributes:
        access_state_object (:obj:`AccessStateObject`): an object that provides checkpoints
    """

    def __init__(self, name, checkpoint_period, checkpoint_dir, metadata, local_species_population,
        dynamic_model):
        self.access_state_object = AccessStateObject(local_species_population, dynamic_model)
        super().__init__(name, checkpoint_period, checkpoint_dir, metadata, self.access_state_object)
